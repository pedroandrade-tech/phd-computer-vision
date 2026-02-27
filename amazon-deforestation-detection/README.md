# Amazon Deforestation Detection

Semantic segmentation of deforested areas in Amazon rainforest using Sentinel-2 satellite imagery.

## Results

| Metric | Score |
|--------|-------|
| Precision | 0.94 |
| Recall | 0.92 |
| F1-Score | 0.93 |
| IoU | 0.87 |

## Dataset

[Deforestation Detection Dataset](https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset) — 16 Sentinel-2 tiles with binary masks (forest/deforestation).

## Model

U-Net with ResNet34 encoder (ImageNet pretrained) using [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

## Usage

```bash
pip install -r requirements.txt
jupyter notebook deforestation_detection.ipynb
```

## Structure

```
├── deforestation_detection.ipynb
├── requirements.txt
└── assets/results.png
```
