# Chest X-Ray Pneumonia Detection

Binary classification of chest X-ray images into Normal vs Pneumonia using EfficientNet-B0 with model explainability via Grad-CAM.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 81.89% |
| Test Recall | 99.23% |
| Test F1-Score | 87.26% |
| Best Val F1 | 0.9961 |

High recall ensures minimal false negatives — crucial for medical diagnosis.

## Model Explainability

### Grad-CAM Visualization

The model's predictions are interpreted through EigenCAM heatmaps:

**Pneumonia Case** — Strong localized activation in consolidation regions:

![Pneumonia Grad-CAM](https://github.com/S-am-ir/pneumonia-detection/blob/main/grads_cam/penumonia_sample.png)

**Normal Case** — Diffused, minimal activation across healthy lungs:

![Normal Grad-CAM](https://github.com/S-am-ir/pneumonia-detection/blob/main/grads_cam/normal_sample.png)

The model correctly identifies pneumonia by focusing on affected lung areas while ignoring healthy tissue.

## Architecture

- **Base Model**: EfficientNet-B0 (ImageNet pre-trained)
- **Input Size**: 224×224 RGB images
- **Output**: Binary classification (Normal / Pneumonia)
- **Loss**: Weighted CrossEntropyLoss (handles class imbalance)
- **Optimizer**: Adam (lr=1e-4)

## Project Structure

```
pneumonia_detection/
├── config.py        # Configuration & constants
├── data.py          # Data loading, preprocessing, datasets
├── training.py      # Model setup, training, evaluation, Grad-CAM
├── main.py          # Full pipeline entry point
├── requirements.txt # Python dependencies
├── .gitignore       # Git exclusion rules
└── grads_cam/       # Visualization outputs
    ├── pneumonia_gradcam.png
    └── normal_gradcam.png
```


## Technologies

- PyTorch, EfficientNet-B0, Albumentations, pytorch-grad-cam (EigenCAM)
- Scikit-learn for metrics

## Dataset

Chest X-Ray images from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):
- ~5,216 training images
- ~1,040 validation images  
- ~624 test images
- 2 classes: Normal, Pneumonia
