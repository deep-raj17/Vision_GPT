"""Download selected USGS TNM Access products for an area of interest."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def project_path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else PROJECT_ROOT / default
    return path if path.is_absolute() else PROJECT_ROOT / path


def datasets_from_env() -> list[str]:
    raw = os.getenv(
        "USGS_TNM_DATASETS",
        "Digital Elevation Model (DEM) 1 meter,"
        "US Topo,"
        "National Hydrography Dataset (NHD),"
        "National Transportation Dataset (NTD)",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


API_URL = os.getenv("USGS_TNM_API_URL", "https://tnmaccess.nationalmap.gov/api/v1/products")
BBOX = os.getenv("USGS_TNM_BBOX", "-77.12,38.80,-76.90,39.00")
MAX_PER_DATASET = int(os.getenv("USGS_TNM_MAX_PER_DATASET", "50"))
REQUEST_DELAY_SECONDS = float(os.getenv("USGS_TNM_REQUEST_DELAY_SECONDS", "1"))
DATASETS = datasets_from_env()
DOWNLOAD_DIR = project_path_from_env(
    "USGS_TNM_DOWNLOAD_DIR",
    "satellite_caption_dataset/raw_images/usgs_tnm",
)

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def safe_folder_name(name: str) -> str:
    cleaned = name.replace("/", "_")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    return cleaned.strip("_") or "dataset"


def search_products(dataset: str) -> list[dict]:
    params = {
        "datasets": dataset,
        "bbox": BBOX,
        "outputFormat": "JSON",
        "max": MAX_PER_DATASET,
    }

    response = requests.get(API_URL, params=params, timeout=90)
    response.raise_for_status()
    return response.json().get("items", [])


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    return filename or "usgs_tnm_product"


def download_file(url: str, folder: Path) -> Path:
    filename = filename_from_url(url)
    output_path = folder / filename

    if output_path.exists():
        print("Already exists:", filename)
        return output_path

    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with output_path.open("wb") as file, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=filename[:35],
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

    print("Downloaded:", output_path)
    return output_path


def main() -> None:
    records: list[dict[str, str | None]] = []

    print("BBOX:", BBOX)
    print("Datasets:", len(DATASETS))
    print("Max per dataset:", MAX_PER_DATASET)

    for dataset in DATASETS:
        print("\nSearching:", dataset)

        try:
            products = search_products(dataset)
        except Exception as exc:
            print("Search failed:", dataset, exc)
            continue

        print("Found:", len(products))
        dataset_folder = DOWNLOAD_DIR / safe_folder_name(dataset)
        dataset_folder.mkdir(parents=True, exist_ok=True)

        for item in products:
            title = item.get("title")
            download_url = item.get("downloadURL")
            downloaded_path = None

            records.append(
                {
                    "dataset": dataset,
                    "title": title,
                    "download_url": download_url,
                    "downloaded_path": downloaded_path,
                }
            )

            if not download_url:
                continue

            try:
                downloaded_path = str(download_file(download_url, dataset_folder))
                records[-1]["downloaded_path"] = downloaded_path
                time.sleep(REQUEST_DELAY_SECONDS)
            except Exception as exc:
                print("Download failed:", title, exc)

    manifest_path = DOWNLOAD_DIR / "download_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print("\nDone. Manifest saved:", manifest_path)


if __name__ == "__main__":
    main()
