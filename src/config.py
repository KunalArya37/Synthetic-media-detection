"""
Central Configuration Module for AI-Driven Synthetic Media Detection & Authentication Engine
"""

import os
import torch
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset_published"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
EXTERNAL_TEST_DIR = DATASET_DIR / "external_test"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAMPLES_DIR = PROJECT_ROOT / "sample_images"

# Ensure runtime directories exist
for d in [DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, EXTERNAL_TEST_DIR, MODELS_DIR, REPORTS_DIR, SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hardware & Acceleration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2 if os.name == 'nt' else 4
PIN_MEMORY = True if DEVICE == "cuda" else False

# Model Hyperparameters
DEFAULT_MODEL_NAME = "efficientnet_b4"  # Options: 'efficientnet_b4', 'resnet50', 'convnext_tiny'
NUM_CLASSES = 2  # [0: Authentic / Real, 1: Synthetic / AI-Generated]
CLASS_NAMES = ["Authentic (Real)", "Synthetic (AI-Generated)"]

# Image Input Dimensions
IMG_SIZE = 256  # 256x256 preserves fine-grained texture & frequency artifacts better than 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

# Saved Checkpoints
BEST_MODEL_PATH = MODELS_DIR / "best_synthetic_detector.pth"
LATEST_MODEL_PATH = MODELS_DIR / "latest_checkpoint.pth"

# Forensic Signal Analysis Thresholds
ELA_SCALE = 15.0
ELA_QUALITY = 90
FFT_ANOMALY_THRESHOLD = 0.55
NOISE_RESIDUAL_THRESHOLD = 0.45
COLOR_ANOMALY_THRESHOLD = 0.50

# Composite Engine Weights
# Combines: Deep Learning Confidence (55%), FFT Spectral (20%), ELA Artifacts (15%), Noise Discrepancy (10%)
WEIGHT_DL = 0.55
WEIGHT_FFT = 0.20
WEIGHT_ELA = 0.15
WEIGHT_NOISE = 0.10
