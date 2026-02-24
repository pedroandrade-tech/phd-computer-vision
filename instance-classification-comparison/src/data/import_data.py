"""Download the Human Face Emotions dataset from Kaggle and organize it into data/raw/."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES, RAW_DIR

log = logging.getLogger(__name__)

KAGGLE_DATASET = "samithsachidanandan/human-face-emotions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def download() -> Path:
    """Download the dataset via kagglehub and return the local path."""
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub is required. Install it with: pip install kagglehub")

    log.info("Downloading %s ...", KAGGLE_DATASET)
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    log.info("Downloaded to %s", path)
    return Path(path)


def organize(download_path: Path, overwrite: bool = False) -> None:
    """Copy class folders from the downloaded dataset into data/raw/."""
    if RAW_DIR.exists():
        if not overwrite:
            log.info("data/raw already exists. Use --overwrite to replace it.")
            return
        shutil.rmtree(RAW_DIR)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        class_name = cls.capitalize()
        src = download_path / "Data" / class_name
        dst = RAW_DIR / class_name

        if not src.is_dir():
            raise FileNotFoundError(f"Expected source directory not found: {src}")

        shutil.copytree(src, dst)
        log.info("%s: %d images", class_name, count_images(dst))


def verify() -> bool:
    """Check that data/raw/ contains the expected class directories with images."""
    ok = True
    for cls in CLASSES:
        class_dir = RAW_DIR / cls.capitalize()
        if not class_dir.is_dir():
            log.warning("Missing directory: %s", class_dir)
            ok = False
            continue
        n = count_images(class_dir)
        log.info("%s: %d images", cls.capitalize(), n)
        if n == 0:
            log.warning("%s has no images", cls.capitalize())
            ok = False
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download and organize the face emotions dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing data/raw/ directory.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data without downloading.")
    args = parser.parse_args()

    if args.verify_only:
        sys.exit(0 if verify() else 1)

    path = download()
    organize(path, overwrite=args.overwrite)
    sys.exit(0 if verify() else 1)