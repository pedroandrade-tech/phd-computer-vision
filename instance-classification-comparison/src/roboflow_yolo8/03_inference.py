"""Run YOLOv8/Roboflow inference on a single simulation and compute metrics."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLASSES,
    CLASS_MAPPING,
    IMAGES_PER_CLASS,
    ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL_DIR,
    ROBOFLOW_SIMS_DIR,
    get_simulation_path,
)

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 40


def _load_config() -> dict:
    path = ROBOFLOW_MODEL_DIR / "roboflow_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}. Run 01_config.py first.")
    return json.loads(path.read_text())


def _connect(config: dict):
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    return rf.workspace(config["workspace"]).project(config["project"]).version(config["version"]).model


def _predict(model, image_path: str, confidence: int = CONFIDENCE_THRESHOLD) -> dict:
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


def process_simulation(model, sim_number: int) -> pd.DataFrame:
    """Classify every image in a simulation. Returns results DataFrame."""
    sim_dir = get_simulation_path(sim_number)
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    rows = []
    total = IMAGES_PER_CLASS * len(CLASSES)

    for cls in CLASSES:
        images = sorted(p for p in (sim_dir / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for img in images:
            result = _predict(model, str(img))
            rows.append({
                "image": img.name,
                "true_class": cls,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "detected": result["detected"],
                "error": result["error"],
            })
            if len(rows) % 50 == 0 or len(rows) == total:
                log.info("SIM%02d  %d/%d", sim_number, len(rows), total)

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict | None:
    """Compute classification metrics from inference results."""
    valid = df[df["predicted_class"].isin(CLASSES)].copy()
    if valid.empty:
        log.error("No valid predictions")
        return None

    y_true = valid["true_class"].map(CLASS_MAPPING)
    y_pred = valid["predicted_class"].map(CLASS_MAPPING)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "total_images": len(df),
        "valid_predictions": len(valid),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }


def save_results(df: pd.DataFrame, metrics: dict, sim_number: int) -> None:
    """Persist detailed CSV and metrics JSON for a simulation."""
    ROBOFLOW_SIMS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = ROBOFLOW_SIMS_DIR / f"sim{sim_number:02d}_detailed.csv"
    df.to_csv(csv_path, index=False)

    json_path = ROBOFLOW_SIMS_DIR / f"sim{sim_number:02d}_metrics.json"
    payload = {
        "simulation": f"SIM{sim_number:02d}",
        "simulation_number": sim_number,
        "model": "YOLOv8_Roboflow",
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    log.info("Results saved: %s, %s", csv_path.name, json_path.name)


def run(sim_number: int) -> bool:
    if not ROBOFLOW_API_KEY:
        log.error("ROBOFLOW_API_KEY not set")
        return False

    config = _load_config()
    model = _connect(config)
    df = process_simulation(model, sim_number)
    metrics = compute_metrics(df)
    if metrics is None:
        return False

    save_results(df, metrics, sim_number)
    log.info("SIM%02d done — accuracy=%.4f, f1=%.4f", sim_number, metrics["accuracy"], metrics["f1_score"])
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on a single simulation.")
    parser.add_argument("--sim", type=int, default=1, help="Simulation number (default: 1)")
    args = parser.parse_args()

    sys.exit(0 if run(args.sim) else 1)
