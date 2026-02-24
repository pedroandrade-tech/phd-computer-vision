"""Centralized project configuration: paths, API keys, and experiment constants."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_HAPPY_DIR = RAW_DIR / "Happy"
RAW_SAD_DIR = RAW_DIR / "Sad"
SIMULATIONS_DIR = DATA_DIR / "simulations"

MODELS_DIR = BASE_DIR / "models"
GEMINI_MODEL_DIR = MODELS_DIR / "gemini_flash"
ROBOFLOW_MODEL_DIR = MODELS_DIR / "roboflow_yolo8"

RESULTS_DIR = BASE_DIR / "results"
GEMINI_RESULTS_DIR = RESULTS_DIR / "gemini"
GEMINI_SIMS_DIR = GEMINI_RESULTS_DIR / "gemini_sims"
GEMINI_METRICS_FILE = GEMINI_RESULTS_DIR / "all_metrics.csv"
GEMINI_STATS_FILE = GEMINI_RESULTS_DIR / "summary_statistics.json"
ROBOFLOW_RESULTS_DIR = RESULTS_DIR / "roboflow_yolo8"
ROBOFLOW_SIMS_DIR = ROBOFLOW_RESULTS_DIR / "roboflow_sims"
ROBOFLOW_METRICS_FILE = ROBOFLOW_RESULTS_DIR / "all_metrics.csv"
ROBOFLOW_STATS_FILE = ROBOFLOW_RESULTS_DIR / "summary_statistics.json"
COMPARISON_DIR = RESULTS_DIR / "comparison"
COMPARISON_PLOTS_DIR = COMPARISON_DIR / "plots"
WILCOXON_RESULTS_FILE = COMPARISON_DIR / "wilcoxon_test_results.json"

# API keys
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Experiment
NUM_SIMULATIONS = 30
IMAGES_PER_CLASS = 100
BATCH_SIZE = 32
CLASSES = ["happy", "sad"]
CLASS_MAPPING = {"sad": 0, "happy": 1}
METRICS = ["accuracy", "precision", "recall", "f1_score"]
METRIC_LABELS = ["Acurácia", "Precisão", "Recall", "F1-Score"]
WILCOXON_ALPHA = 0.05

# Gemini rate limiting (free tier: 15 req/min)
GEMINI_REQUESTS_PER_MINUTE = 15
GEMINI_DELAY = 60 / GEMINI_REQUESTS_PER_MINUTE


def get_simulation_path(sim_number: int) -> Path:
    """Return the data directory for a simulation (1-indexed)."""
    return SIMULATIONS_DIR / f"SIM{sim_number:02d}"


def get_metrics_path(sim_number: int, model: str = "gemini") -> Path:
    """Return the per-simulation metrics file for a given model."""
    parent = {"gemini": GEMINI_SIMS_DIR, "roboflow": ROBOFLOW_SIMS_DIR}.get(model)
    if parent is None:
        raise ValueError(f"Unknown model '{model}'. Expected 'gemini' or 'roboflow'.")
    return parent / f"sim{sim_number:02d}_metrics.json"


def validate_api_keys() -> bool:
    """Check that required API keys are present. Logs a warning for any missing key."""
    missing = [k for k, v in {"ROBOFLOW_API_KEY": ROBOFLOW_API_KEY, "GEMINI_API_KEY": GEMINI_API_KEY}.items() if not v]
    if missing:
        log.warning("Missing API keys in .env: %s", ", ".join(missing))
        return False
    return True


def ensure_directories() -> None:
    """Create output directories if they don't exist yet."""
    for d in [GEMINI_SIMS_DIR, ROBOFLOW_SIMS_DIR, COMPARISON_PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)