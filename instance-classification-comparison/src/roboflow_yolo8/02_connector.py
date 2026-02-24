"""Connect to Roboflow, load the YOLOv8 model, and validate with a test prediction."""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ROBOFLOW_API_KEY, ROBOFLOW_MODEL_DIR, get_simulation_path

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 40


def load_config() -> dict:
    """Load model config saved by 01_config.py."""
    path = ROBOFLOW_MODEL_DIR / "roboflow_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}. Run 01_config.py first.")
    return json.loads(path.read_text())


def connect(config: dict | None = None):
    """Connect to Roboflow and return the model object, or None on failure."""
    if not ROBOFLOW_API_KEY:
        log.error("ROBOFLOW_API_KEY not set in .env")
        return None

    cfg = config or load_config()
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        model = rf.workspace(cfg["workspace"]).project(cfg["project"]).version(cfg["version"]).model
        log.info("Model loaded: %s v%d", cfg["project"], cfg["version"])
        return model
    except Exception as e:
        log.error("Failed to load model: %s", e)
        return None


def predict_emotion(model, image_path: str, confidence: int = CONFIDENCE_THRESHOLD) -> dict:
    """Classify a single image. Returns dict with predicted_class, confidence, detected, error."""
    try:
        result = model.predict(image_path, confidence=confidence, overlap=30).json()
        preds = result.get("predictions", [])
        if not preds:
            return {"predicted_class": None, "confidence": 0.0, "detected": False, "error": "no face detected"}
        det = preds[0]
        return {
            "predicted_class": det.get("class", "unknown"),
            "confidence": det.get("confidence", 0.0),
            "detected": True,
            "error": None,
        }
    except Exception as e:
        return {"predicted_class": None, "confidence": 0.0, "detected": False, "error": str(e)}


def smoke_test(model) -> bool:
    """Run a single prediction on the first image from SIM01/happy."""
    test_dir = get_simulation_path(1) / "happy"
    if not test_dir.is_dir():
        log.error("Test directory not found: %s", test_dir)
        return False

    images = sorted(test_dir.glob("*.jpg"))[:1] or sorted(test_dir.glob("*.png"))[:1]
    if not images:
        log.error("No images found in %s", test_dir)
        return False

    result = predict_emotion(model, str(images[0]))
    if result["detected"]:
        correct = result["predicted_class"] == "happy"
        log.info(
            "Prediction: %s (conf=%.2f, expected: happy) — %s",
            result["predicted_class"], result["confidence"], "correct" if correct else "wrong",
        )
        return True

    log.warning("Prediction failed: %s", result["error"])
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = connect()
    if model is None:
        sys.exit(1)
    sys.exit(0 if smoke_test(model) else 1)
