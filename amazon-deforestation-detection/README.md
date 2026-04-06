# Amazon Deforestation Detection

Semantic segmentation of deforested areas in Amazon rainforest using Sentinel-2 satellite imagery. This project compares CNN-based and Transformer-based encoder architectures for the segmentation task.

## Dataset

[Deforestation Detection Dataset](https://www.kaggle.com/datasets/akhilchibber/deforestation-detection-dataset) — 16 Sentinel-2 tiles (2816×2816 pixels) with binary masks (forest/deforestation), processed into 1936 chips of 256×256 pixels.

## Models

Both models use the U-Net decoder architecture with different encoders, implemented via [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

| Model | Encoder | Type | Parameters |
|-------|---------|------|------------|
| U-Net | ResNet34 | CNN | 24.4M |
| U-Net | MiT-B2 | Vision Transformer | 27.5M |

## Results

### Evaluation Metrics

| Metric | U-Net (CNN) | U-Net (ViT) |
|--------|-------------|-------------|
| Precision | 0.9312 | 0.9282 |
| Recall | 0.9263 | 0.9270 |
| F1-Score | 0.9287 | 0.9276 |
| IoU | 0.8670 | 0.8650 |

### Training Dynamics

| Aspect | U-Net (CNN) | U-Net (ViT) |
|--------|-------------|-------------|
| Best Val Loss | 0.0937 | 0.0937 |
| Best Epoch | 7 | 9 |
| Convergence | Faster | Slower |
| Stability | Overfits after epoch 7 | Stable through epoch 10 |

### Analysis

Both architectures achieved equivalent final performance on this dataset. The CNN encoder converged faster (best result at epoch 7 vs epoch 9), while the ViT encoder showed more stable training without overfitting in later epochs. For this dataset size (1936 samples), the encoder architecture is not the limiting factor — both approaches are equally effective for deforestation detection.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook deforestation_detection_unet.ipynb  # CNN version
jupyter notebook deforestation_detection_vit.ipynb   # ViT version
```

## Structure

```
├── deforestation_detection_unet.ipynb  # U-Net with ResNet34 encoder
├── deforestation_detection_vit.ipynb   # U-Net with MiT-B2 encoder
├── requirements.txt
└── data/
    ├── raw/archive/                    # Original dataset
    └── processed/chips/                # 256×256 chips
```

## Requirements

- Python 3.8+
- PyTorch
- segmentation-models-pytorch
- rioxarray
