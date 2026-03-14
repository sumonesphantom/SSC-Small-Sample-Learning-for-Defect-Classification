# Small Sample Learning for Defect Classification

**Semiconductor Solutions Challenge 2026 — Problem A**
Developed in partnership with Intel Corporation

## Problem Overview

Defect classification is essential to semiconductor manufacturing, where accurate and timely detection directly impacts yield, product quality, and time-to-market. In real production environments, labeled defect data is often limited, creating a gap between practical needs and traditional machine learning approaches that rely on large datasets.

This project builds a model capable of detecting and classifying defects from limited data, reflecting how a human expert rapidly learns and adapts from just a few examples.

### Objectives

1. Enable solution scalability via automation
2. Improve insights into defect categories
3. Improve yield and product quality

### Scope

| Requirement | Specification |
|---|---|
| Input | Grayscale images up to ~1500 x 2500 pixels |
| Output | Classification into 8 defect classes + 1 good class |
| Inference Speed | ~1 second per image |
| Accuracy Target | ~85% overall classification accuracy |
| Data Challenge | Highly imbalanced classes (9 to 7,135 samples per class) |

---

## Dataset

### Downloading the Data

The dataset is provided by Intel as part of the Semiconductor Solutions Challenge 2026. It is included in the challenge repository as both a `dataset` folder and a `dataset.zip` file (both contain identical data — use only one).

1. Download the dataset from the challenge repository provided by ASU/Intel.
2. Place the images in a `Data/` folder at the project root, organized by class:

```
Data/
├── good/          # 7,135 images — normal / no defect
├── defect1/       #   253 images
├── defect2/       #   178 images
├── defect3/       #     9 images
├── defect4/       #    14 images
├── defect5/       #   411 images
├── defect8/       #   803 images
├── defect9/       #   319 images
└── defect10/      #   674 images
```

3. If you downloaded `dataset.zip`, unzip it and rename the folder to `Data/`:

```bash
unzip dataset.zip -d Data/
```

> The `Data/` directory is excluded from version control via `.gitignore`.

### Dataset Summary

| Class | Samples | Description |
|---|---|---|
| good | 7,135 | Normal / no defect |
| defect8 | 803 | Defect type 8 |
| defect10 | 674 | Defect type 10 |
| defect5 | 411 | Defect type 5 |
| defect9 | 319 | Defect type 9 |
| defect1 | 253 | Defect type 1 |
| defect2 | 178 | Defect type 2 |
| defect4 | 14 | Defect type 4 |
| defect3 | 9 | Defect type 3 |

**Total: 9,796 images** with extreme class imbalance (793:1 ratio between largest and smallest class).

---

## Approach

### Why Prototypical Networks?

Traditional deep learning classifiers require large, balanced datasets. With only 9 samples in the rarest class, standard fine-tuning would overfit immediately. Our approach uses **Prototypical Networks** — a metric-learning framework designed for few-shot classification — combined with a two-stage training pipeline.

Key advantages:
- **Learns from few examples**: Designed to classify with as few as 1-5 examples per class
- **Handles imbalance naturally**: Episodic training gives equal representation to all classes
- **Fast adaptation**: New classes or examples can be incorporated without retraining
- **Demonstrates learning speed**: Can show accuracy improvement as more examples are provided

### Architecture

```
Input Image (224x224, RGB)
    |
    v
ConvNeXt-Tiny (pretrained on ImageNet-22k)
    |
    v
Global Average Pooling -> 768-dim features
    |
    v
Projection Head (768 -> 512 -> 256)
    |
    v
L2-Normalized 256-dim Embedding
    |
    v
Classification via Nearest Class Prototype (cosine similarity)
```

**Backbone**: ConvNeXt-Tiny was chosen for its strong transfer learning performance, efficiency, and compatibility with Intel OpenVINO for deployment on Intel hardware.

### Two-Stage Training Pipeline

**Stage 1 — Episodic Prototypical Training (up to 30 epochs)**
- Trains using N-way K-shot episodes that simulate few-shot scenarios
- Each episode: sample K=5 support + Q=5 query images per class
- Loss: Cross-entropy on query predictions against prototype-based logits
- Differential learning rates: backbone at 0.1x, projection head at 1x
- 100 training episodes + 20 validation episodes per epoch
- Cosine annealing learning rate schedule
- Early stopping with patience of 7 epochs

**Stage 2 — Fine-tuning with Focal Loss (up to 20 epochs)**
- Adds a linear classification head on top of the learned embeddings
- Focal Loss (gamma=2) with per-class alpha weights for class imbalance handling
- WeightedRandomSampler ensures rare classes are seen proportionally
- Cosine annealing schedule with lower learning rate (5e-5)
- Early stopping with patience of 7 epochs

### Training Callbacks

Both stages use the following callbacks:
- **ModelCheckpoint**: Saves the top-3 best models by validation accuracy; automatically removes worse checkpoints
- **EarlyStopping**: Stops training if validation accuracy does not improve for 7 consecutive epochs
- **CSVLogger**: Logs all metrics (loss, accuracy, learning rates, epoch time) to CSV files in `outputs/logs/`
- **LRTracker**: Records learning rates per parameter group each epoch

### Class Imbalance Handling

Multiple strategies are combined to address the extreme imbalance:

1. **Episodic training** (Stage 1): Each episode samples equally from all classes
2. **Focal Loss** (Stage 2): Down-weights easy/frequent examples, focuses on hard/rare ones
3. **Inverse-frequency class weights**: Alpha weights inversely proportional to class frequency
4. **WeightedRandomSampler**: Oversamples rare classes during fine-tuning
5. **Data augmentation**: Random flips, rotations, affine transforms, color jitter, and random erasing expand the effective training set for all classes

### Data Augmentation

Training transforms:
- Resize to 224x224
- Random horizontal and vertical flips
- Random rotation (up to 15 degrees)
- Random affine translation (10%) and scaling (90%-110%)
- Color jitter (brightness and contrast)
- Random erasing (20% probability)

---

## Project Structure

```
├── config.py          # Hyperparameters, paths, and device settings
├── callbacks.py       # ModelCheckpoint, EarlyStopping, CSVLogger, LRTracker
├── dataset.py         # Data loading, image caching, episodic sampling, augmentation
├── model.py           # PrototypicalNet, FinetuneClassifier, FocalLoss
├── train.py           # Two-stage training pipeline with callbacks
├── evaluate.py        # Evaluation metrics and plot generation
├── inference.py       # Single-image inference CLI
├── requirements.txt   # Python dependencies
├── Data/              # Dataset (not tracked in git)
│   ├── good/
│   ├── defect1/
│   ├── ...
│   └── defect10/
└── outputs/
    ├── models/        # Saved model checkpoints
    ├── logs/          # Training CSV logs
    └── plots/         # Evaluation plots
```

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- **Windows**, **macOS**, or **Linux**

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SSC-Small-Sample-Learning-for-Defect-Classification
```

### 2. Install Dependencies

**Windows (with NVIDIA GPU):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Windows (CPU only):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**macOS (Apple Silicon):**

```bash
pip install -r requirements.txt
```

**Linux:**

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the Intel-provided dataset into the `Data/` directory as described in the [Dataset](#dataset) section above. Verify the structure:

```bash
# macOS / Linux
ls Data/

# Windows (PowerShell)
dir Data\

# Expected: defect1  defect10  defect2  defect3  defect4  defect5  defect8  defect9  good
```

### 4. Device Configuration

The device is **auto-detected** in `config.py` — no manual edits needed:

| Hardware | Auto-detected device |
|---|---|
| NVIDIA GPU (Windows/Linux) | `cuda` |
| Apple Silicon (macOS) | `mps` |
| No GPU | `cpu` |

To override, edit `config.py`:

```python
DEVICE = "cuda"   # Force NVIDIA GPU
DEVICE = "mps"    # Force Apple Silicon
DEVICE = "cpu"    # Force CPU
```

---

## Training

### Run Training

```bash
python train.py
```

This runs the full two-stage pipeline:

1. **Pre-loads** all 9,796 images into RAM for fast training
2. **Stage 1**: Prototypical episodic training (up to 30 epochs, early stopping)
3. **Stage 2**: Fine-tuning with Focal Loss (up to 20 epochs, early stopping)

### What Gets Saved

Training automatically saves:

| File | Description |
|---|---|
| `outputs/models/proto_model.pth` | Best prototypical model weights (Stage 1) |
| `outputs/models/finetune_model.pth` | Best fine-tuned model weights (Stage 2) |
| `outputs/models/proto_epoch*_val_acc=*.pth` | Top-3 Stage 1 checkpoints with metrics |
| `outputs/models/finetune_epoch*_val_acc=*.pth` | Top-3 Stage 2 checkpoints with metrics |
| `outputs/logs/proto_log.csv` | Stage 1 per-epoch metrics (loss, acc, LR, time) |
| `outputs/logs/finetune_log.csv` | Stage 2 per-epoch metrics |
| `outputs/test_samples.pth` | Train/val/test data splits for reproducibility |

### Training Output

You will see progress bars and per-epoch metrics:

```
Pre-loading 9796 images into RAM... 1000 2000 ... Done.

============================================================
STAGE 1: Prototypical (Episodic) Training
============================================================
  Epoch  1 |████████████████████████████████| 100/100 loss=0.523 acc=0.834
  Epoch   1/30 | Loss: 0.6234 | Train Acc: 0.7821 | Val Acc: 0.8150 | Time: 45.2s
  [Checkpoint] New best val_acc=0.8150 saved to proto_epoch001_val_acc=0.8150.pth
  ...
```

### Resume from Checkpoint

To load a specific checkpoint and continue or use it:

```python
import torch
from model import PrototypicalNet, FinetuneClassifier

# Load prototypical model
proto = PrototypicalNet(embedding_dim=256)
checkpoint = torch.load("outputs/models/proto_epoch015_val_acc=0.9200.pth", weights_only=True)
proto.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, metrics: {checkpoint['metrics']}")
```

### Estimated Training Time

| Hardware | Stage 1 | Stage 2 | Total |
|---|---|---|---|
| NVIDIA GPU (RTX 3090/4090) | ~15-25 min | ~3-5 min | ~20-30 min |
| Apple Silicon (M1/M2/M3) | ~45-70 min | ~8-12 min | ~55-80 min |
| CPU only (Windows/Linux/Mac) | ~3-4 hours | ~30-45 min | ~4-5 hours |

---

## Testing and Evaluation

### Run Full Evaluation

After training, generate all evaluation metrics and plots:

```bash
python evaluate.py
```

This produces:

| Output | Description |
|---|---|
| `outputs/plots/accuracy_vs_occurrence.png` | Per-class accuracy vs number of samples |
| `outputs/plots/confusion_matrix.png` | Full 9x9 confusion matrix |
| `outputs/plots/per_class_f1.png` | Per-class F1 scores |
| `outputs/plots/learning_curve.png` | Accuracy vs K-shot (1, 2, 3, 5, 10, 15, 20 examples) |
| Terminal output | Classification report with precision, recall, F1, and inference time |

### Single-Image Inference

**Using the fine-tuned classifier** (recommended):

```bash
python inference.py --image Data/defect1/sample.PNG
```

Output:
```
Image: Data/defect1/sample.PNG
Mode: finetune
Prediction: defect1 (confidence: 0.9432)
Inference time: 12.3 ms

All class probabilities:
       good: 0.0021
    defect1: 0.9432
    defect2: 0.0156
    ...
```

**Using the prototypical model** (few-shot mode):

```bash
python inference.py --image Data/defect3/sample.PNG --mode proto --k_shot 5
```

This builds class prototypes from K random training examples per class, then classifies the query image by nearest prototype.

### Test on a Batch of Images

```python
import torch
from model import PrototypicalNet, FinetuneClassifier
from dataset import get_val_transform
from PIL import Image
import config

device = torch.device(config.DEVICE)

# Load model
proto = PrototypicalNet(embedding_dim=256)
classifier = FinetuneClassifier(proto)
classifier.load_state_dict(torch.load("outputs/models/finetune_model.pth", map_location=device, weights_only=True))
classifier.to(device).eval()

# Classify
transform = get_val_transform()
img = Image.open("path/to/image.png").convert("RGB")
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = classifier(tensor)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()

print(f"Predicted: {config.CLASS_NAMES[pred]} ({probs[0][pred]:.4f})")
```

---

## Deliverables Mapping

| Deliverable (from Problem Brief) | Implementation |
|---|---|
| 1. Application for detecting and classifying defects from grayscale images | `inference.py` — CLI accepting any image, outputs class + confidence |
| 2. Evaluation plots: accuracy vs defect class occurrence | `evaluate.py` generates accuracy vs occurrence, confusion matrix, F1 plots |
| 3. ~85% overall classification accuracy | Targeted via two-stage training with imbalance handling |
| 4. Demonstrate how quickly the model learns | Learning curve plot: accuracy vs K-shot (1, 2, 3, 5, 10, 15, 20 examples) |
| 5. Documentation of approach, assumptions, and hardware | This README + inline code documentation |

---

## Assumptions

1. **Image-level labels only**: The dataset contains folder-level class labels without bounding boxes or segmentation masks. The task is treated as whole-image classification.
2. **Grayscale input**: Images are converted to 3-channel RGB (replicated grayscale) to leverage ImageNet-pretrained backbones.
3. **Variable image sizes**: All images are resized to 224x224 for the model. Aspect ratio distortion is acceptable given the nature of semiconductor inspection images.
4. **No defect localization**: The "detection" aspect is addressed at the image level (defect present or not + class), not spatial localization within the image.
5. **Closed-world assumption**: Only the 9 provided classes exist; no unknown/novel defect types at inference time.

---

## Hardware

### Supported Platforms

| Platform | GPU Support | Notes |
|---|---|---|
| **Windows 10/11** | NVIDIA CUDA | Install PyTorch with CUDA index URL (see Setup) |
| **macOS** | Apple Silicon MPS | Works on M1/M2/M3/M4 chips |
| **Linux** | NVIDIA CUDA | Standard PyTorch install |
| **Any (CPU)** | None | Slower but fully functional |

- **Recommended for production**: Intel hardware with OpenVINO export for optimized inference
- **Inference time target**: < 1 second per image (achievable on all modern hardware)

### Exporting for Intel Deployment

The model can be exported to ONNX/OpenVINO format for deployment on Intel CPUs and GPUs:

```python
import torch

model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "defect_classifier.onnx", opset_version=17)
```
