"""Search and download Bhoonidhi products for satellite-caption dataset building."""

from __future__ import annotations

import os
import time
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


BASE_URL = os.getenv("BHOONIDHI_BASE_URL", "https://bhoonidhi-api.nrsc.gov.in")
TOKEN_URL = f"{BASE_URL}/auth/token"
SEARCH_URL = f"{BASE_URL}/data/search"
DOWNLOAD_URL = f"{BASE_URL}/download"

USER_ID = os.getenv("BHOONIDHI_USER_ID")
PASSWORD = os.getenv("BHOONIDHI_PASSWORD")

DATE_RANGE = os.getenv("BHOONIDHI_DATE_RANGE", "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z")
LIMIT = int(os.getenv("BHOONIDHI_LIMIT", "500"))
REQUEST_DELAY_SECONDS = float(os.getenv("BHOONIDHI_REQUEST_DELAY_SECONDS", "1"))
DOWNLOAD_DIR = project_path_from_env(
    "BHOONIDHI_DOWNLOAD_DIR",
    "satellite_caption_dataset/raw_images/bhoonidhi",
)

COLLECTION = os.getenv(
    "BHOONIDHI_COLLECTION",
    os.getenv("BHOONIDHI_COLLECTIONS", "Sentinel-1A_SAR-IW_GRD").split(",")[0].strip(),
)

AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [88.20, 22.40],
            [88.60, 22.40],
            [88.60, 22.80],
            [88.20, 22.80],
            [88.20, 22.40],
        ]
    ],
}

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_access_token() -> str:
    if not USER_ID or not PASSWORD:
        raise ValueError("Set BHOONIDHI_USER_ID and BHOONIDHI_PASSWORD in your .env file.")

    payload = {
        "userId": USER_ID,
        "password": PASSWORD,
        "grant_type": "password",
    }

    response = requests.post(TOKEN_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["access_token"]


def search_products(token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "collections": [COLLECTION],
        "datetime": DATE_RANGE,
        "intersects": AOI,
        "filter": {
            "args": [
                {"property": "Online"},
                "Y",
            ],
            "op": "eq",
        },
        "filter-lang": "cql2-json",
        "limit": LIMIT,
    }

    response = requests.post(SEARCH_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json().get("features", [])


def download_product(token: str, product: dict) -> None:
    product_id = product["id"]
    collection_name = product.get("collection", COLLECTION)
    headers = {"Authorization": f"Bearer {token}"}
    safe_product_id = str(product_id).replace("/", "_")
    output_path = DOWNLOAD_DIR / f"{collection_name}_{safe_product_id}.zip"

    if output_path.exists():
        print("Already downloaded:", output_path)
        return

    params = {
        "id": product_id,
        "collection": collection_name,
    }

    with requests.get(
        DOWNLOAD_URL,
        params=params,
        headers=headers,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with output_path.open("wb") as file, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=str(product_id)[:30],
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

    print("Downloaded:", output_path)


def main() -> None:
    print("Logging in...")
    token = get_access_token()
    print("Login successful")

    print("Searching online products:", COLLECTION)
    features = search_products(token)
    print("Found online products:", len(features))

    for product in features:
        product_id = product.get("id")
        print("Product ID:", product_id)

        if not product_id:
            continue

        try:
            download_product(token, product)
            time.sleep(REQUEST_DELAY_SECONDS)
        except Exception as exc:
            print("Download failed:", product_id, exc)


if __name__ == "__main__":
    main()
