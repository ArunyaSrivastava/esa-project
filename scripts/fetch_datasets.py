"""
External Dataset Fetcher.

Pulls pre-collected physiological datasets from the internet so you do NOT have
to collect/record your own training data. Supports several sources:

  WESAD (Wearable Stress and Affect Detection)
    - 15 subjects; chest (RespiBAN: ECG, EDA, EMG, Resp, Temp, ACC @700 Hz) and
      wrist (Empatica E4: BVP/PPG @64 Hz, EDA @4 Hz, Temp @4 Hz, ACC @32 Hz).
    - Affective states: neutral (baseline), stress, amusement.
    - Two download paths:
        1) --source sciebo    : the original University of Siegen archive (.zip)
        2) --source uci       : the UCI Machine Learning Repository (robust API)

  Usage:
    python scripts/fetch_datasets.py --source sciebo
    python scripts/fetch_datasets.py --source uci
    python scripts/fetch_datasets.py --source uci --limit-subjects S2 S3 S4

After fetching, run `python scripts/preprocess_wesad.py` then
`python scripts/train_model.py` (or just launch `python run.py`, which does it
automatically on first run if the model checkpoint is missing).
"""

import argparse
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import WESAD_DIR  # noqa: E402

# Original University of Siegen archive (full 15 subjects, ~2 GB).
SCIEBO_WESAD_URL = "https://uni-siegen.sciebo.de/s/mcrNsj7645odNqP/download"

# UCI Machine Learning Repository identifier for WESAD.
UCI_WESAD_ID = 465


def _fetch_sciebo() -> None:
    """Mirror of the original download_wesad.py (full archive over HTTP)."""
    import requests
    from tqdm import tqdm

    WESAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = WESAD_DIR / "WESAD.zip"

    print(f"[fetch] Target: {WESAD_DIR.resolve()}")
    print(f"[fetch] Pulling WESAD archive ({SCIEBO_WESAD_URL}) ...")

    if zip_path.exists():
        print("[fetch] Archive already present, skipping download.")
    else:
        try:
            with requests.get(SCIEBO_WESAD_URL, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(zip_path, "wb") as f, tqdm(
                    desc="WESAD.zip", total=total, unit="iB",
                    unit_scale=True, unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
        except Exception as e:
            print(f"[fetch] sciebo download failed: {e}")
            print("[fetch] Falling back to the UCI API route:")
            _fetch_uci()
            return

    with zipfile.ZipFile(zip_path, "r") as zf:
        print("[fetch] Extracting archive...")
        zf.extractall(WESAD_DIR)
def _fetch_uci(limit_subjects=None) -> None:
    """
    Pull WESAD via the UCI Machine Learning Repository using `ucimlrepo`.

    The `ucimlrepo` package downloads and caches the raw dataset. We then locate
    any subject .pkl files it cached and copy them (optionally filtered) into
    this project's data/wesad/ directory.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("[fetch] The 'ucimlrepo' package is required for this source.")
        print("[fetch] Install it with:  pip install ucimlrepo")
        sys.exit(1)

    print(f"[fetch] Fetching WESAD from UCI repo (id={UCI_WESAD_ID})...")
    fetch_ucirepo(id=UCI_WESAD_ID)

    import os
    from pathlib import Path as _P
    import shutil

    data_dir = WESAD_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    cache = _P(os.path.expanduser("~")) / ".cache" / "ucimlrepo"
    print(f"[fetch] UCI cached archive located at: {cache}")

    copied = 0
    if cache.exists():
        for pkl in cache.rglob("*.pkl"):
            name = pkl.stem.upper()
            if limit_subjects and name not in limit_subjects:
                continue
            try:
                shutil.copyfile(pkl, data_dir / f"{name}.pkl")
                print(f"[fetch]   -> copied {name}.pkl")
                copied += 1
            except Exception as e:
                print(f"[fetch]   x failed to copy {name}: {e}")

    if copied:
        print(f"[fetch] Copied {copied} subject file(s) into {data_dir}")
    else:
        print("[fetch] No subject .pkl found in the UCI cache.")
        print("[fetch] The UCI package caches the raw archive; ensure the subject")
        print("[fetch] files are placed under data/wesad/ as S2.pkl, S3.pkl, ...")

    print("[fetch] Next steps:")
    print("  python scripts/preprocess_wesad.py")
    print("  python scripts/train_model.py")
    print("  (or just `python run.py` on first launch)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch pre-collected datasets from the internet.")
    parser.add_argument(
        "--source", choices=["sciebo", "uci"], default="uci",
        help="Which download route to use (uci is the robust default).",
    )
    parser.add_argument(
        "--limit-subjects", nargs="*", default=None,
        help="Only keep these subjects (e.g. S2 S3 S4). Default: all found.",
    )
    args = parser.parse_args()

    if args.source == "sciebo":
        _fetch_sciebo()
    else:
        _fetch_uci(limit_subjects=args.limit_subjects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    print(f"[fetch] Done. Subjects now in: {WESAD_DIR.resolve()}")