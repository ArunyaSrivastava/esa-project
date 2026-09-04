"""
WESAD Dataset Downloader & Extractor Script.
Fetches the official WESAD (Wearable Stress and Affect Detection) dataset archive
from the University of Siegen / UCI Machine Learning Repository.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import WESAD_DIR

WESAD_DOWNLOAD_URL = "https://uni-siegen.sciebo.de/s/mcrNsj7645odNqP/download"


def download_wesad(target_dir: Path = WESAD_DIR):
    """Download and extract official WESAD dataset archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "WESAD.zip"

    print(f"[WESAD Downloader] Target directory: {target_dir}")
    print(f"[WESAD Downloader] Source: {WESAD_DOWNLOAD_URL}")

    if zip_path.exists():
        print(f"[WESAD Downloader] Found existing archive at {zip_path}")
    else:
        print("[WESAD Downloader] Connecting to repository (this may take a few minutes)...")
        try:
            response = requests.get(WESAD_DOWNLOAD_URL, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f, tqdm(
                desc="Downloading WESAD.zip",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            print("[WESAD Downloader] Download complete!")
        except Exception as e:
            print(f"[WESAD Downloader] Direct download failed ({e}).")
            print("Note: You can manually place WESAD subject folders (S2, S3, etc.) into:")
            print(f"  {target_dir.resolve()}")
            return

    if zip_path.exists():
        print("[WESAD Downloader] Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print("[WESAD Downloader] Extraction complete!")


if __name__ == "__main__":
    download_wesad()
