# 🎯 Instance Classification Comparison

**PhD Computer Vision Project** - Comparing instance classification models for emotion detection (Happy vs Sad).

## 📋 Overview

This project compares two different approaches for facial emotion classification:

| Model | Type | Approach |
|-------|------|----------|
| **YOLOv8 + Roboflow** | Specialized Object Detection | Pre-trained on facial emotions |
| **Gemini Flash** | Multimodal LLM | General-purpose with prompt engineering |

### Key Features

- 🔬 **30 Monte Carlo simulations** (200 images each)
- 📊 **4 metrics**: Accuracy, Precision, Recall, F1-Score
- 📈 **Statistical comparison**: Wilcoxon Signed-Rank Test (α = 0.05)
- 📉 **Visualizations**: BoxPlots, Line Charts, Confusion Matrices

## 🗂️ Project Structure

```
instance-classification-comparison/
├── config.py                    # Centralized configuration
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not in repo)
│
├── data/
│   ├── raw/                     # Original dataset
│   └── simulations/             # SIM01-SIM30 folders
│       ├── SIM01/
│       │   ├── happy/           # 100 images
│       │   └── sad/             # 100 images
│       └── ...
│
├── src/
│   ├── data/
│   │   ├── import_data.py       # Download from Kaggle
│   │   └── data_prep.py         # Create simulations
│   │
│   ├── roboflow_yolo8/
│   │   ├── 01_config.py         # Environment setup
│   │   ├── 02_connector.py      # Model connection
│   │   ├── 03_inference.py      # Single simulation
│   │   └── 04_batch_processing.py  # All 30 simulations
│   │
│   ├── gemini/
│   │   ├── 01_config.py         # Environment setup
│   │   ├── 02_connector.py      # API connection
│   │   ├── 03_inference.py      # Single simulation
│   │   └── 04_batch_processing.py  # All 30 simulations
│   │
│   └── evaluation/
│       └── comparison.py        # Statistical comparison
│
├── results/
│   ├── roboflow_yolo8/
│   │   ├── roboflow_sims/       # Individual results
│   │   ├── all_metrics.csv      # Consolidated metrics
│   │   └── summary_statistics.json
│   │
│   ├── gemini/
│   │   ├── gemini_sims/         # Individual results
│   │   ├── all_metrics.csv      # Consolidated metrics
│   │   └── summary_statistics.json
│   │
│   └── comparison/
│       ├── boxplot_*.png        # BoxPlot visualizations
│       ├── line_*.png           # Line charts
│       ├── wilcoxon_test_results.json
│       └── comparison_report.txt
│
└── models/
    ├── roboflow_yolo8/
    │   └── roboflow_config.json
    └── gemini_flash/
        └── gemini_config.json
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/instance-classification-comparison.git
cd instance-classification-comparison
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
# Roboflow API Key
# Get yours at: https://app.roboflow.com/settings/api
ROBOFLOW_API_KEY=your_roboflow_api_key_here

# Google Gemini API Key
# Get yours at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Kaggle (optional - for dataset download)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### 5. Download and prepare data

```bash
# Download dataset from Kaggle
python src/data/import_data.py

# Create 30 simulations
python src/data/data_prep.py
```

## 📊 Running the Pipelines

### YOLOv8 + Roboflow Pipeline

```bash
# 1. Verify environment
python src/roboflow_yolo8/01_config.py

# 2. Connect and test model
python src/roboflow_yolo8/02_connector.py

# 3. Process single simulation (SIM01)
python src/roboflow_yolo8/03_inference.py

# 4. Process all 30 simulations
python src/roboflow_yolo8/04_batch_processing.py
```

### Gemini Flash Pipeline

```bash
# 1. Verify environment
python src/gemini/01_config.py

# 2. Connect and test API
python src/gemini/02_connector.py

# 3. Process single simulation (~13-15 min)
python src/gemini/03_inference.py

# 4. Process all 30 simulations (~6-8 hours)
python src/gemini/04_batch_processing.py
```

### Compare Models

```bash
python src/evaluation/comparison.py
```

## ⏱️ Estimated Processing Times

| Task | YOLOv8 | Gemini |
|------|--------|--------|
| Single simulation (200 images) | ~2 min | ~13-15 min |
| All 30 simulations | ~60 min | ~6-8 hours |

> **Note**: Gemini has a rate limit of 15 requests/minute on the free tier.

## 📈 Output Examples

### Metrics CSV Format

```csv
simulation_number,simulation,accuracy,precision,recall,f1_score,total_images,valid_predictions
1,SIM01,0.8950,0.9100,0.8800,0.8947,200,200
2,SIM02,0.9050,0.9000,0.9100,0.9050,200,200
...
```

### Wilcoxon Test Results

```json
{
  "test": "Wilcoxon Signed-Rank Test",
  "significance_level": 0.05,
  "results": {
    "accuracy": {
      "p_value": 0.0023,
      "is_significant": true,
      "mean_difference": 0.0450
    }
  }
}
```

## 🔧 Configuration

All paths and constants are centralized in `config.py`:

```python
from config import (
    PATHS,              # All project paths
    CLASSES,            # ['happy', 'sad']
    NUM_SIMULATIONS,    # 30
    IMAGES_PER_CLASS,   # 100
    ROBOFLOW_API_KEY,   # From .env
    GEMINI_API_KEY,     # From .env
)
```

## 📝 Interactive Menus

All scripts have interactive menus with verification options:

```
📋 OPTIONS:
   1. Execute full process
   2. Verify existing results only
   3. Cancel

❓ Choose an option (1/2/3):
```

## 🛠️ Tech Stack

- **Python** 3.10+
- **YOLOv8** (Ultralytics)
- **Roboflow** API
- **Google Gemini** API
- **Pandas** / **NumPy** - Data processing
- **Scikit-learn** - Metrics calculation
- **Matplotlib** / **Seaborn** - Visualizations
- **SciPy** - Statistical tests

## 📄 License

This project is part of a PhD research. Please cite appropriately if used.

## 👤 Author

**Pedro Fonseca de Andrade**

PhD Candidate - Computer Vision Research

---

⭐ If this project helped you, please give it a star!
