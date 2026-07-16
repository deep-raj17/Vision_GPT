"""Download NASA LANCE NRT MODIS/VIIRS products for dataset building."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def project_path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name)
    path = Path(value) if value else PROJECT_ROOT / default
    return path if path.is_absolute() else PROJECT_ROOT / path


NASA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN")
COLLECTION_VERSION = os.getenv("NASA_LANCE_COLLECTION_VERSION", "61")
BASE_URL = os.getenv(
    "NASA_LANCE_BASE_URL",
    f"https://nrt3.modaps.eosdis.nasa.gov/archive/allData/{COLLECTION_VERSION}",
)
PRODUCT = os.getenv("NASA_LANCE_PRODUCT", "MOD021KM")
MAX_FILES = int(os.getenv("NASA_LANCE_MAX_FILES", "5"))
DATA_URL = os.getenv("NASA_LANCE_DATA_URL", f"{BASE_URL}/{PRODUCT}/Recent/")
DOWNLOAD_DIR = project_path_from_env(
    "NASA_LANCE_DOWNLOAD_DIR",
    f"satellite_caption_dataset/raw_images/nasa_lance/{PRODUCT}",
)

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def auth_headers() -> dict[str, str]:
    if not NASA_TOKEN:
        raise ValueError(
            "Set NASA_EARTHDATA_TOKEN in your .env file. Generate one from "
            "https://ladsweb.modaps.eosdis.nasa.gov/profile/#generate-token"
        )
    return {"Authorization": f"Bearer {NASA_TOKEN}"}


def list_files(url: str, headers: dict[str, str]) -> list[str]:
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    files = []

    for link in soup.find_all("a"):
        href = link.get("href", "")
        if href.endswith((".hdf", ".nc", ".h5")):
            files.append(urljoin(url, href))

    return files


def download_file(file_url: str, headers: dict[str, str]) -> None:
    filename = file_url.rstrip("/").split("/")[-1]
    output_path = DOWNLOAD_DIR / filename

    if output_path.exists():
        print("Already downloaded:", filename)
        return

    with requests.get(file_url, headers=headers, stream=True, timeout=300) as response:
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

    print("Saved:", output_path)


def main() -> None:
    headers = auth_headers()
    files = list_files(DATA_URL, headers)
    print("Data URL:", DATA_URL)
    print("Found files:", len(files))
    print("Downloading files:", min(len(files), MAX_FILES))

    for file_url in files[:MAX_FILES]:
        download_file(file_url, headers)


if __name__ == "__main__":
    main()
