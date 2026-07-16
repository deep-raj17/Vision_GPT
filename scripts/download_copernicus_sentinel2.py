"""Download Sentinel-2 L2A products for satellite-caption dataset building."""

from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def project_path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else PROJECT_ROOT / default
    return path if path.is_absolute() else PROJECT_ROOT / path

USERNAME = os.getenv("CDSE_USERNAME")
PASSWORD = os.getenv("CDSE_PASSWORD")

START_DATE = os.getenv("CDSE_START_DATE", "2025-01-01")
END_DATE = os.getenv("CDSE_END_DATE", "2025-01-05")
WKT = os.getenv(
    "CDSE_WKT",
    "POLYGON((88.20 22.40,88.60 22.40,88.60 22.80,88.20 22.80,88.20 22.40))",
)
MAX_PRODUCTS = int(os.getenv("CDSE_MAX_PRODUCTS", "5"))
DOWNLOAD_DIR = project_path_from_env(
    "CDSE_DOWNLOAD_DIR",
    "satellite_caption_dataset/raw_images/copernicus",
)
COLLECTIONS = ["SENTINEL-2"]

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_token() -> str:
    if not USERNAME or not PASSWORD:
        raise ValueError("Set CDSE_USERNAME and CDSE_PASSWORD in your .env file.")

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": USERNAME,
        "password": PASSWORD,
        "grant_type": "password",
    }

    response = requests.post(url, data=data, timeout=60)
    response.raise_for_status()
    return response.json()["access_token"]


def search_products(collection: str) -> list[dict]:
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    query = (
        f"{base_url}?$filter="
        f"Collection/Name eq '{collection}' "
        f"and ContentDate/Start ge {START_DATE}T00:00:00.000Z "
        f"and ContentDate/Start le {END_DATE}T23:59:59.999Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{WKT}') "
        f"and contains(Name,'MSIL2A')"
        f"&$orderby=ContentDate/Start desc"
        f"&$top={MAX_PRODUCTS}"
    )

    response = requests.get(query, timeout=60)
    response.raise_for_status()
    return response.json().get("value", [])


def download_product(product: dict, token: str) -> None:
    product_id = product["Id"]
    product_name = product["Name"].replace("/", "_")
    output_path = DOWNLOAD_DIR / f"{product_name}.zip"

    if output_path.exists():
        print("Already downloaded:", output_path)
        return

    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    print("Downloading:", product_name)
    with requests.get(url, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with output_path.open("wb") as file, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=product_name[:40],
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

    print("Saved:", output_path)


def main() -> None:
    token = get_token()

    for collection in COLLECTIONS:
        print("\nSearching collection:", collection)
        products = search_products(collection)
        print("Products found:", len(products))

        for product in products:
            print("-", product["Name"])
            download_product(product, token)


if __name__ == "__main__":
    main()
