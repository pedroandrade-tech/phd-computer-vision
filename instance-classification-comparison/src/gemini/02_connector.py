"""Connect to the Gemini API and validate with a test prediction."""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GEMINI_API_KEY, get_simulation_path

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
        import PIL.Image  # noqa: F401 – imported for later use in predict

        self._genai = genai
        self._pil = __import__("PIL.Image", fromlist=["Image"])
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
        self.model_id = model_id
        log.info("Model loaded: %s", model_id)

    def predict(self, image_path: str) -> dict:
        """Classify a single image. Returns dict with predicted_class, detected, error, raw_response."""
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


def connect() -> GeminiClassifier | None:
    """Create a GeminiClassifier instance, or None on failure."""
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set in .env")
        return None
    try:
        return GeminiClassifier(api_key=GEMINI_API_KEY)
    except Exception as e:
        log.error("Failed to connect: %s", e)
        return None


def smoke_test(classifier: GeminiClassifier) -> bool:
    """Run a single prediction on the first image from SIM01/happy."""
    test_dir = get_simulation_path(1) / "happy"
    if not test_dir.is_dir():
        log.error("Test directory not found: %s", test_dir)
        return False

    images = sorted(test_dir.glob("*.jpg"))[:1] or sorted(test_dir.glob("*.png"))[:1]
    if not images:
        log.error("No images found in %s", test_dir)
        return False

    t0 = time.time()
    result = classifier.predict(str(images[0]))
    elapsed = time.time() - t0

    if result["detected"]:
        correct = result["predicted_class"] == "happy"
        log.info(
            "Prediction: %s (expected: happy) — %s — %.2fs",
            result["predicted_class"], "correct" if correct else "wrong", elapsed,
        )
        return True

    log.warning("Prediction failed: %s", result["error"])
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    clf = connect()
    if clf is None:
        sys.exit(1)

    sys.exit(0 if smoke_test(clf) else 1)
