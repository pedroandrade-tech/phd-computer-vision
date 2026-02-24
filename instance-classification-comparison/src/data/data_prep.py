"""Create simulation splits from the raw dataset.

Reads images from data/raw/{Happy,Sad} and produces NUM_SIMULATIONS
balanced splits under data/simulations/SIM01..SIM30, each containing
IMAGES_PER_CLASS images per class with no overlap between splits.
"""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    IMAGES_PER_CLASS,
    NUM_SIMULATIONS,
    RAW_DIR,
    SIMULATIONS_DIR,
)

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_DIR_MAP = {"Happy": "happy", "Sad": "sad"}


def collect_images() -> dict[str, list[Path]]:
    """Return a dict mapping each raw class name to its list of image paths."""
    images: dict[str, list[Path]] = {}
    for raw_name in CLASS_DIR_MAP:
        class_dir = RAW_DIR / raw_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {class_dir}")
        files = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        images[raw_name] = files
        log.info("%s: %d images", raw_name, len(files))
    return images


def verify_capacity(images: dict[str, list[Path]]) -> None:
    """Raise if there aren't enough images for all simulations without overlap."""
    required = NUM_SIMULATIONS * IMAGES_PER_CLASS
    for raw_name, paths in images.items():
        if len(paths) < required:
            raise ValueError(
                f"{raw_name}: need {required} unique images but only {len(paths)} available"
            )


def create_simulation(
    sim_number: int,
    images: dict[str, list[Path]],
    used: dict[str, set[int]],
    base_seed: int = 42,
) -> None:
    """Create a single simulation directory with randomly sampled images."""
    rng = random.Random(base_seed + sim_number)
    sim_dir = SIMULATIONS_DIR / f"SIM{sim_number:02d}"

    for raw_name, dest_name in CLASS_DIR_MAP.items():
        available = [i for i in range(len(images[raw_name])) if i not in used[raw_name]]
        selected = rng.sample(available, IMAGES_PER_CLASS)
        used[raw_name].update(selected)

        dest_dir = sim_dir / dest_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for idx in selected:
            src = images[raw_name][idx]
            shutil.copy2(src, dest_dir / src.name)


def create_all_simulations(overwrite: bool = False) -> None:
    """Generate all simulation splits."""
    if SIMULATIONS_DIR.exists():
        if not overwrite:
            log.info("Simulations directory already exists. Use --overwrite to recreate.")
            return
        log.info("Removing existing simulations directory")
        shutil.rmtree(SIMULATIONS_DIR)

    SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)

    images = collect_images()
    verify_capacity(images)
    used: dict[str, set[int]] = {name: set() for name in CLASS_DIR_MAP}

    for sim_num in range(1, NUM_SIMULATIONS + 1):
        create_simulation(sim_num, images, used)
        if sim_num % 10 == 0 or sim_num == 1:
            log.info("SIM%02d created", sim_num)

    log.info("All %d simulations created under %s", NUM_SIMULATIONS, SIMULATIONS_DIR)


def verify_simulations() -> bool:
    """Check that every simulation directory has the expected image counts."""
    ok = True
    for sim_num in range(1, NUM_SIMULATIONS + 1):
        sim_dir = SIMULATIONS_DIR / f"SIM{sim_num:02d}"
        for dest_name in CLASS_DIR_MAP.values():
            class_dir = sim_dir / dest_name
            if not class_dir.is_dir():
                log.warning("Missing: %s", class_dir)
                ok = False
                continue
            count = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if count != IMAGES_PER_CLASS:
                log.warning("SIM%02d/%s: expected %d images, found %d", sim_num, dest_name, IMAGES_PER_CLASS, count)
                ok = False

    if ok:
        log.info("Verification passed: %d simulations, %d images/class each", NUM_SIMULATIONS, IMAGES_PER_CLASS)
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare simulation splits from raw data.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing simulations before creating new ones.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing simulations without creating new ones.")
    args = parser.parse_args()

    if args.verify_only:
        sys.exit(0 if verify_simulations() else 1)

    create_all_simulations(overwrite=args.overwrite)
    sys.exit(0 if verify_simulations() else 1)
