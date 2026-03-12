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

## Dataset

Intel-provided grayscale semiconductor inspection images organized into 9 classes:

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

The dataset should be placed in the `Data/` directory with one subfolder per class. This folder is excluded from version control via `.gitignore`.

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
Global Average Pooling → 768-dim features
    |
    v
Projection Head (768 → 512 → 256)
    |
    v
L2-Normalized 256-dim Embedding
    |
    v
Classification via Nearest Class Prototype (cosine similarity)
```

**Backbone**: ConvNeXt-Tiny was chosen for its strong transfer learning performance, efficiency, and compatibility with Intel OpenVINO for deployment on Intel hardware.

### Two-Stage Training Pipeline

**Stage 1 — Episodic Prototypical Training (30 epochs)**
- Trains using N-way K-shot episodes that simulate few-shot scenarios
- Each episode: sample K=5 support + Q=5 query images per class
- Loss: Cross-entropy on query predictions against prototype-based logits
- Differential learning rates: backbone at 0.1x, projection head at 1x
- 500 training episodes + 100 validation episodes per epoch
- Cosine annealing learning rate schedule

**Stage 2 — Fine-tuning with Focal Loss (20 epochs)**
- Adds a linear classification head on top of the learned embeddings
- Focal Loss (gamma=2) with per-class alpha weights for class imbalance handling
- WeightedRandomSampler ensures rare classes are seen proportionally
- Cosine annealing schedule with lower learning rate (5e-5)

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

## Project Structure

```
├── config.py          # Hyperparameters, paths, and device settings
├── dataset.py         # Data loading, episodic sampling, augmentation, splits
├── model.py           # PrototypicalNet, FinetuneClassifier, FocalLoss
├── train.py           # Two-stage training pipeline
├── evaluate.py        # Evaluation metrics and plot generation
├── inference.py       # Single-image inference CLI
├── Data/              # Dataset (not tracked in git)
│   ├── good/
│   ├── defect1/
│   ├── defect2/
│   ├── defect3/
│   ├── defect4/
│   ├── defect5/
│   ├── defect8/
│   ├── defect9/
│   └── defect10/
└── outputs/
    ├── models/        # Saved model weights
    └── plots/         # Evaluation plots
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- macOS (Apple Silicon with MPS) / Linux / Windows with CUDA GPU

### Install Dependencies

```bash
pip install torch torchvision timm scikit-learn matplotlib pillow numpy
```

### Device Configuration

Edit `config.py` to set your device:
```python
DEVICE = "mps"    # Apple Silicon (default)
DEVICE = "cuda"   # NVIDIA GPU
DEVICE = "cpu"    # CPU fallback
```

## Usage

### 1. Training

Run the full two-stage training pipeline:

```bash
python train.py
```

This will:
- Load and stratified-split the dataset (70% train / 20% val / 10% test)
- Run Stage 1: Prototypical episodic training (30 epochs)
- Run Stage 2: Fine-tuning with Focal Loss (20 epochs)
- Save model weights to `outputs/models/`
- Save data splits to `outputs/test_samples.pth`

### 2. Evaluation

Generate all evaluation plots and metrics:

```bash
python evaluate.py
```

This produces:
- **`outputs/plots/accuracy_vs_occurrence.png`** — Per-class accuracy vs sample count
- **`outputs/plots/confusion_matrix.png`** — Full confusion matrix
- **`outputs/plots/per_class_f1.png`** — Per-class F1 scores
- **`outputs/plots/learning_curve.png`** — Accuracy vs number of support examples (K-shot)
- Classification report with precision, recall, F1 per class
- Inference time measurement

### 3. Single-Image Inference

Classify a single image using the fine-tuned model:

```bash
python inference.py --image path/to/image.png
```

Classify using the prototypical model with K-shot support:

```bash
python inference.py --image path/to/image.png --mode proto --k_shot 5
```

## Deliverables Mapping

| Deliverable (from Problem Brief) | Implementation |
|---|---|
| 1. Application for detecting and classifying defects from grayscale images | `inference.py` — CLI application accepting any grayscale image, outputs class + confidence |
| 2. Evaluation plots: accuracy vs defect class occurrence | `evaluate.py` generates accuracy vs occurrence, confusion matrix, F1 plots |
| 3. ~85% overall classification accuracy | Targeted via two-stage training with imbalance handling |
| 4. Demonstrate how quickly the model learns | Learning curve plot: accuracy vs K-shot (1, 2, 3, 5, 10, 15, 20 examples) |
| 5. Documentation of approach, assumptions, and hardware | This README + inline code documentation |

## Assumptions

1. **Image-level labels only**: The dataset contains folder-level class labels without bounding boxes or segmentation masks. The task is treated as whole-image classification.
2. **Grayscale input**: Images are converted to 3-channel RGB (replicated grayscale) to leverage ImageNet-pretrained backbones.
3. **Variable image sizes**: All images are resized to 224x224 for the model. Aspect ratio distortion is acceptable given the nature of semiconductor inspection images.
4. **No defect localization**: The "detection" aspect is addressed at the image level (defect present or not + class), not spatial localization within the image.
5. **Closed-world assumption**: Only the 9 provided classes exist; no unknown/novel defect types at inference time.

## Hardware

- **Development**: Apple Silicon (M-series) with MPS acceleration
- **Recommended for production**: Intel hardware with OpenVINO export for optimized inference
- **Inference time target**: < 1 second per image (achievable on all modern hardware)

The model can be exported to ONNX/OpenVINO format for deployment on Intel CPUs and GPUs using standard PyTorch export utilities.
