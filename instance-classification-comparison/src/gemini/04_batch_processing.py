"""Process all simulations with Gemini and consolidate metrics."""

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
    GEMINI_METRICS_FILE,
    GEMINI_SIMS_DIR,
    GEMINI_STATS_FILE,
    IMAGES_PER_CLASS,
    METRICS,
    NUM_SIMULATIONS,
    get_simulation_path,
)

log = logging.getLogger(__name__)

PROMPT = (
    "Look at this face carefully. "
    "Classify the emotion as either 'Happy' or 'Sad'. "
    "Answer with ONLY ONE WORD: either 'Happy' or 'Sad'. "
    "Do not add any explanation."
)


class GeminiClassifier:
    """Emotion classifier backed by Google Gemini (multimodal)."""

    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        self._pil = __import__("PIL.Image", fromlist=["Image"])
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
        self.model_id = model_id

    def predict(self, image_path: str) -> dict:
        try:
            img = self._pil.open(image_path).resize((224, 224), self._pil.LANCZOS)
            response = self.model.generate_content([PROMPT, img])
            if not response.text:
                return {"predicted_class": None, "detected": False, "error": "empty response", "raw_response": None}
            text = response.text.strip().lower()
            if "happy" in text and "sad" not in text:
                cls = "happy"
            elif "sad" in text and "happy" not in text:
                cls = "sad"
            else:
                return {"predicted_class": None, "detected": False, "error": f"ambiguous: {text}", "raw_response": text}
            return {"predicted_class": cls, "detected": True, "error": None, "raw_response": text}
        except Exception as e:
            return {"predicted_class": None, "detected": False, "error": str(e), "raw_response": None}


def process_simulation(classifier: GeminiClassifier, sim_number: int) -> dict | None:
    """Run inference on a single simulation and return its metrics dict.

    Skips if results already exist on disk (allows resuming).
    """
    json_path = GEMINI_SIMS_DIR / f"sim{sim_number:02d}_metrics.json"
    if json_path.exists():
        log.info("SIM%02d already processed, skipping", sim_number)
        return json.loads(json_path.read_text())

    sim_dir = get_simulation_path(sim_number)
    if not sim_dir.exists():
        log.warning("SIM%02d directory not found", sim_number)
        return None

    rows = []
    t0 = time.time()
    total = IMAGES_PER_CLASS * len(CLASSES)

    for cls in CLASSES:
        images = sorted(p for p in (sim_dir / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, img in enumerate(images, 1):
            result = classifier.predict(str(img))
            rows.append({
                "image": img.name, "true_class": cls,
                "predicted_class": result["predicted_class"],
                "detected": result["detected"], "error": result["error"],
            })
            if len(rows) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / len(rows) * (total - len(rows)) / 60
                log.info("SIM%02d  %d/%d  (%.0f min left)", sim_number, len(rows), total, eta)
            time.sleep(GEMINI_DELAY)

    elapsed = time.time() - t0
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
        "model_id": classifier.model_id,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "total_images": len(df),
        "valid_predictions": len(valid),
        "processing_time_minutes": round(elapsed / 60, 2),
        "timestamp": datetime.now().isoformat(),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }

    GEMINI_SIMS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GEMINI_SIMS_DIR / f"sim{sim_number:02d}_detailed.csv", index=False)
    json_path.write_text(json.dumps(metrics, indent=2))

    log.info("SIM%02d done — acc=%.4f f1=%.4f (%.1f min)", sim_number, metrics["accuracy"], metrics["f1_score"], elapsed / 60)
    return metrics


def consolidate() -> None:
    """Build all_metrics.csv and summary_statistics.json from individual JSONs."""
    records = []
    for sim_num in range(1, NUM_SIMULATIONS + 1):
        p = GEMINI_SIMS_DIR / f"sim{sim_num:02d}_metrics.json"
        if p.exists():
            records.append(json.loads(p.read_text()))

    if not records:
        log.warning("No simulation results found to consolidate")
        return

    cols = ["simulation_number", "simulation", "accuracy", "precision", "recall", "f1_score", "total_images", "valid_predictions"]
    df = pd.DataFrame(records)[cols].sort_values("simulation_number")
    df.to_csv(GEMINI_METRICS_FILE, index=False)
    log.info("Consolidated %d simulations -> %s", len(df), GEMINI_METRICS_FILE.name)

    stats = {"model": "gemini_flash", "n": len(df), "timestamp": datetime.now().isoformat(), "metrics": {}}
    for m in METRICS:
        v = df[m]
        stats["metrics"][m] = {
            "mean": float(v.mean()), "std": float(v.std()), "median": float(v.median()),
            "min": float(v.min()), "max": float(v.max()),
        }
    GEMINI_STATS_FILE.write_text(json.dumps(stats, indent=2))
    log.info("Summary statistics -> %s", GEMINI_STATS_FILE.name)


def run(start: int = 1, end: int = NUM_SIMULATIONS) -> bool:
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set")
        return False

    GEMINI_SIMS_DIR.mkdir(parents=True, exist_ok=True)
    classifier = GeminiClassifier(api_key=GEMINI_API_KEY)
    t0 = time.time()

    for sim_num in range(start, end + 1):
        process_simulation(classifier, sim_num)

    total = time.time() - t0
    log.info("Batch complete: %d simulations in %.1f hours", end - start + 1, total / 3600)

    consolidate()
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Batch-process all simulations with Gemini.")
    parser.add_argument("--start", type=int, default=1, help="First simulation number (default: 1)")
    parser.add_argument("--end", type=int, default=NUM_SIMULATIONS, help=f"Last simulation number (default: {NUM_SIMULATIONS})")
    args = parser.parse_args()

    sys.exit(0 if run(args.start, args.end) else 1)