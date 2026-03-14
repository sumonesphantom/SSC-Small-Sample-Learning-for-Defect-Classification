"""Configuration for the defect classification pipeline."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Class names (folder names in Data/)
CLASS_NAMES = [
    "good", "defect1", "defect2", "defect3", "defect4",
    "defect5", "defect8", "defect9", "defect10",
]
NUM_CLASSES = len(CLASS_NAMES)

# Image settings
IMAGE_SIZE = 224  # ConvNeXt-Tiny default input
INPUT_CHANNELS = 3  # replicate grayscale to 3 channels

# Backbone
BACKBONE = "convnext_tiny.fb_in22k"  # pretrained on ImageNet-22k
EMBEDDING_DIM = 768  # ConvNeXt-Tiny feature dim

# Prototypical network training
N_WAY = NUM_CLASSES        # use all classes per episode
K_SHOT_SUPPORT = 5         # support samples per class
Q_QUERY = 5                # query samples per class
NUM_EPISODES_TRAIN = 100   # episodes per epoch
NUM_EPISODES_VAL = 20
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Fine-tuning (stage 2)
FINETUNE_EPOCHS = 20
FINETUNE_LR = 5e-5
FINETUNE_BATCH_SIZE = 32

# Data split
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Device — auto-detect best available
import torch as _torch
if _torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Callbacks
EARLY_STOPPING_PATIENCE = 7    # stop if no improvement for N epochs
MIN_DELTA = 1e-4               # minimum improvement to count as progress
CHECKPOINT_SAVE_TOP_K = 3      # keep top K model checkpoints
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Inference
INFERENCE_BATCH_SIZE = 1
