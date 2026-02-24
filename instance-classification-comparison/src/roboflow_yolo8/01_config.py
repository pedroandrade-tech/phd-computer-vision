"""Verify environment and save model configuration for the Roboflow/YOLOv8 pipeline."""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLASSES,
    NUM_SIMULATIONS,
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL_DIR,
    ROBOFLOW_SIMS_DIR,
    get_simulation_path,
)

log = logging.getLogger(__name__)

WORKSPACE = "emotions-dectection"
PROJECT = "human-face-emotions"
VERSION = 28
CONFIDENCE_THRESHOLD = 40

REQUIRED_PACKAGES = {
    "roboflow": "roboflow",
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}


def check_packages() -> bool:
    ok = True
    for module, package in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            log.warning("Missing package: pip install %s", package)
            ok = False
    return ok


def check_api_key() -> bool:
    if not ROBOFLOW_API_KEY:
        log.warning("ROBOFLOW_API_KEY not set in .env")
        return False
    log.info("ROBOFLOW_API_KEY configured")
    return True


def check_simulations() -> bool:
    found = sum(1 for i in range(1, NUM_SIMULATIONS + 1) if get_simulation_path(i).exists())
    log.info("Simulations: %d/%d", found, NUM_SIMULATIONS)
    return found == NUM_SIMULATIONS


def save_model_config() -> None:
    ROBOFLOW_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ROBOFLOW_SIMS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "workspace": WORKSPACE,
        "project": PROJECT,
        "version": VERSION,
        "target_classes": CLASSES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }
    path = ROBOFLOW_MODEL_DIR / "roboflow_config.json"
    path.write_text(json.dumps(config, indent=2))
    log.info("Model config saved to %s", path)


def main() -> bool:
    ok = True
    if not check_packages():
        ok = False
    if not check_api_key():
        ok = False
    if not check_simulations():
        ok = False
    save_model_config()

    if ok:
        log.info("Environment ready. Next: python src/roboflow_yolo8/02_connector.py")
    else:
        log.warning("Issues found — review warnings above")
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(0 if main() else 1)
