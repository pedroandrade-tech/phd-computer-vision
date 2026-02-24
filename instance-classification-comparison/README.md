# Instance Classification Comparison

Comparative evaluation of instance classification approaches for binary facial emotion detection (happy vs. sad), using repeated random sampling across 30 independent trials.

## Overview

This project benchmarks two fundamentally different strategies for image-level emotion classification:

- **YOLOv8 via Roboflow** — a specialized object detection model fine-tuned on facial emotion data.
- **Gemini Flash** — a general-purpose multimodal LLM guided through prompt engineering.

Each model is evaluated over 30 independent trials of 200 randomly sampled images (100 per class), with performance compared through standard classification metrics and non-parametric statistical testing.

## Project Structure

```
instance-classification-comparison/
├── config.py
├── requirements.txt
├── .env
│
├── data/
│   ├── raw/
│   └── simulations/            # SIM01–SIM30
│
├── src/
│   ├── data/
│   │   ├── import_data.py
│   │   └── data_prep.py
│   │
│   ├── roboflow_yolo8/
│   │   ├── 01_config.py
│   │   ├── 02_connector.py
│   │   ├── 03_inference.py
│   │   └── 04_batch_processing.py
│   │
│   ├── gemini/
│   │   ├── 01_config.py
│   │   ├── 02_connector.py
│   │   ├── 03_inference.py
│   │   └── 04_batch_processing.py
│   │
│   └── evaluation/
│       └── comparison.py
│
├── results/
│   ├── roboflow_yolo8/
│   ├── gemini/
│   └── comparison/
│
└── models/
    ├── roboflow_yolo8/
    └── gemini_flash/
```

## Setup

### Requirements

- Python 3.10+
- API keys for Roboflow, Google Gemini, and optionally Kaggle

### Installation

```bash
git clone https://github.com/yourusername/instance-classification-comparison.git
cd instance-classification-comparison

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root:

```
ROBOFLOW_API_KEY=your_key
GEMINI_API_KEY=your_key
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

### Data Preparation

```bash
python src/data/import_data.py    # download dataset from Kaggle
python src/data/data_prep.py      # generate 30 simulation splits
```

## Usage

### YOLOv8 + Roboflow

```bash
python src/roboflow_yolo8/01_config.py
python src/roboflow_yolo8/02_connector.py
python src/roboflow_yolo8/03_inference.py         # single trial
python src/roboflow_yolo8/04_batch_processing.py  # all 30 trials
```

### Gemini Flash

```bash
python src/gemini/01_config.py
python src/gemini/02_connector.py
python src/gemini/03_inference.py                 # single trial (~13 min)
python src/gemini/04_batch_processing.py          # all 30 trials (~6–8 h)
```

> Gemini's free tier is limited to 15 requests/min, which is the main bottleneck for batch processing.

### Evaluation

```bash
python src/evaluation/comparison.py
```

## Methodology

Each of the 30 trials uses a balanced random sample of 200 images (100 happy, 100 sad) drawn from the full dataset. Both models are evaluated on the same splits to ensure a fair paired comparison.

**Metrics:** Accuracy, Precision, Recall, F1-Score.

**Statistical testing:** Wilcoxon Signed-Rank Test (alpha = 0.05) applied to each metric across the 30 paired observations.

**Visualizations:** box plots, line charts across trials, and confusion matrices.

## Estimated Processing Time

| Scope | YOLOv8 | Gemini |
|-------|--------|--------|
| 1 trial (200 images) | ~2 min | ~13–15 min |
| 30 trials | ~60 min | ~6–8 h |

## Configuration

All paths, class labels, and experiment parameters are centralized in `config.py`:

```python
from config import PATHS, CLASSES, NUM_SIMULATIONS, IMAGES_PER_CLASS
```

## Tech Stack

Python, YOLOv8 (Ultralytics), Roboflow API, Google Gemini API, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, SciPy.

## License

This project is part of ongoing PhD research. If you use or reference it, please cite appropriately.

## Author

Pedro Fonseca de Andrade