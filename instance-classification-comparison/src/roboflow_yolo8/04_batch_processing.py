"""Process all simulations with YOLOv8/Roboflow and consolidate metrics."""

import argparse
import json
import logging
import sys
import time
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
    METRICS,
    NUM_SIMULATIONS,
    ROBOFLOW_API_KEY,
    ROBOFLOW_METRICS_FILE,
    ROBOFLOW_MODEL_DIR,
    ROBOFLOW_SIMS_DIR,
    ROBOFLOW_STATS_FILE,
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


def process_simulation(model, sim_number: int) -> dict | None:
    """Run inference on a single simulation and return its metrics dict.

    Skips if results already exist on disk (allows resuming).
    """
    json_path = ROBOFLOW_SIMS_DIR / f"sim{sim_number:02d}_metrics.json"
    if json_path.exists():
        log.info("SIM%02d already processed, skipping", sim_number)
        return json.loads(json_path.read_text())

    sim_dir = get_simulation_path(sim_number)
    if not sim_dir.exists():
        log.warning("SIM%02d directory not found", sim_number)
        return None

    rows = []
    total = IMAGES_PER_CLASS * len(CLASSES)

    for cls in CLASSES:
        images = sorted(p for p in (sim_dir / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for img in images:
            result = _predict(model, str(img))
            rows.append({
                "image": img.name, "true_class": cls,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "detected": result["detected"], "error": result["error"],
            })
            if len(rows) % 50 == 0:
                log.info("SIM%02d  %d/%d", sim_number, len(rows), total)

    df = pd.DataFrame(rows)
    valid = df[df["predicted_class"].isin(CLASSES)].copy()
    if valid.empty:
        log.warning("SIM%02d: no valid predictions", sim_number)
        return None

    y_true = valid["true_class"].map(CLASS_MAPPING)
    y_pred = valid["predicted_class"].map(CLASS_MAPPING)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "simulation": f"SIM{sim_number:02d}",
        "simulation_number": sim_number,
        "model": "YOLOv8_Roboflow",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "total_images": len(df),
        "valid_predictions": len(valid),
        "timestamp": datetime.now().isoformat(),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }

    ROBOFLOW_SIMS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROBOFLOW_SIMS_DIR / f"sim{sim_number:02d}_detailed.csv", index=False)
    json_path.write_text(json.dumps(metrics, indent=2))

    log.info("SIM%02d done — acc=%.4f f1=%.4f", sim_number, metrics["accuracy"], metrics["f1_score"])
    return metrics


def consolidate() -> None:
    """Build all_metrics.csv and summary_statistics.json from individual JSONs."""
    records = []
    for sim_num in range(1, NUM_SIMULATIONS + 1):
        p = ROBOFLOW_SIMS_DIR / f"sim{sim_num:02d}_metrics.json"
        if p.exists():
            records.append(json.loads(p.read_text()))

    if not records:
        log.warning("No simulation results found to consolidate")
        return

    cols = ["simulation_number", "simulation", "accuracy", "precision", "recall", "f1_score", "total_images", "valid_predictions"]
    df = pd.DataFrame(records)[cols].sort_values("simulation_number")
    df.to_csv(ROBOFLOW_METRICS_FILE, index=False)
    log.info("Consolidated %d simulations -> %s", len(df), ROBOFLOW_METRICS_FILE.name)

    stats = {"model": "YOLOv8_Roboflow", "n": len(df), "timestamp": datetime.now().isoformat(), "metrics": {}}
    for m in METRICS:
        v = df[m]
        stats["metrics"][m] = {
            "mean": float(v.mean()), "std": float(v.std()), "median": float(v.median()),
            "min": float(v.min()), "max": float(v.max()),
        }
    ROBOFLOW_STATS_FILE.write_text(json.dumps(stats, indent=2))
    log.info("Summary statistics -> %s", ROBOFLOW_STATS_FILE.name)


def run(start: int = 1, end: int = NUM_SIMULATIONS) -> bool:
    if not ROBOFLOW_API_KEY:
        log.error("ROBOFLOW_API_KEY not set")
        return False

    ROBOFLOW_SIMS_DIR.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    model = _connect(config)
    t0 = time.time()

    for sim_num in range(start, end + 1):
        process_simulation(model, sim_num)

    total = time.time() - t0
    log.info("Batch complete: %d simulations in %.1f min", end - start + 1, total / 60)

    consolidate()
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Batch-process all simulations with YOLOv8/Roboflow.")
    parser.add_argument("--start", type=int, default=1, help="First simulation number (default: 1)")
    parser.add_argument("--end", type=int, default=NUM_SIMULATIONS, help=f"Last simulation number (default: {NUM_SIMULATIONS})")
    args = parser.parse_args()

    sys.exit(0 if run(args.start, args.end) else 1)
