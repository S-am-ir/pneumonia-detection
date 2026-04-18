"""
Configuration and constants for pneumonia detection project.
"""
from pathlib import Path

# Data path (Kaggle dataset)
DATA_DIR = Path("/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray")

# Model checkpoint
MODEL_CHECKPOINT = "/kaggle/working/best_pneumonia_efficientnet.pth"

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30

# Grad-CAM settings
TARGET_IDX = 7  # Index of test image for visualization
