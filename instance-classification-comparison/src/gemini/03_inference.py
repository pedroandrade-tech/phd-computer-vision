"""Run Gemini inference on a single simulation and compute metrics."""

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
    GEMINI_API_KEY,
    GEMINI_DELAY,
    GEMINI_RESULTS_DIR,
    GEMINI_SIMS_DIR,
    IMAGES_PER_CLASS,
    get_simulation_path,
)

# Reuse classifier from 02_connector
from importlib import import_module
_connector = import_module("02_connector")
GeminiClassifier = _connector.GeminiClassifier

log = logging.getLogger(__name__)


def process_simulation(classifier: GeminiClassifier, sim_number: int) -> tuple[pd.DataFrame, float]:
    """Classify every image in a simulation. Returns (results_df, elapsed_seconds)."""
    sim_dir = get_simulation_path(sim_number)
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    rows = []
    t0 = time.time()
    total = IMAGES_PER_CLASS * len(CLASSES)

    for cls in CLASSES:
        images = sorted((sim_dir / cls).glob("*"))
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        for i, img_path in enumerate(images, 1):
            result = classifier.predict(str(img_path))
            rows.append({
                "image": img_path.name,
                "true_class": cls,
                "predicted_class": result["predicted_class"],
                "detected": result["detected"],
                "error": result["error"],
            })

            done = len(rows)
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) / 60
                log.info("SIM%02d  %d/%d  (%.0f min remaining)", sim_number, done, total, eta)

            time.sleep(GEMINI_DELAY)

    return pd.DataFrame(rows), time.time() - t0


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


def save_results(df: pd.DataFrame, metrics: dict, sim_number: int, elapsed: float) -> None:
    """Persist detailed CSV and metrics JSON for a simulation."""
    GEMINI_SIMS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = GEMINI_SIMS_DIR / f"sim{sim_number:02d}_detailed.csv"
    df.to_csv(csv_path, index=False)

    json_path = GEMINI_SIMS_DIR / f"sim{sim_number:02d}_metrics.json"
    payload = {
        "simulation": f"SIM{sim_number:02d}",
        "simulation_number": sim_number,
        "model_id": "gemini-2.0-flash",
        "processing_time_minutes": round(elapsed / 60, 2),
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    log.info("Results saved: %s, %s", csv_path.name, json_path.name)


def run(sim_number: int) -> bool:
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set")
        return False

    classifier = GeminiClassifier(api_key=GEMINI_API_KEY)
    df, elapsed = process_simulation(classifier, sim_number)
    metrics = compute_metrics(df)
    if metrics is None:
        return False

    save_results(df, metrics, sim_number, elapsed)
    log.info("SIM%02d done — accuracy=%.4f, f1=%.4f (%.1f min)",
             sim_number, metrics["accuracy"], metrics["f1_score"], elapsed / 60)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run Gemini inference on a single simulation.")
    parser.add_argument("--sim", type=int, default=1, help="Simulation number (default: 1)")
    args = parser.parse_args()

    sys.exit(0 if run(args.sim) else 1)