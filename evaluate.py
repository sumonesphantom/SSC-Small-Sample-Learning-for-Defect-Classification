"""
Evaluation and plotting:
    1. Per-class accuracy vs class occurrence
    2. Confusion matrix
    3. Learning curve (accuracy vs number of examples seen)
    4. Overall metrics report
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score,
)
from collections import defaultdict

import config
from dataset import (
    get_all_samples, get_val_transform, DefectDataset,
    load_episode_images, EpisodicSampler,
)
from model import PrototypicalNet, FinetuneClassifier
from torch.utils.data import DataLoader


def load_finetune_model(device):
    """Load the fine-tuned classifier."""
    proto = PrototypicalNet(embedding_dim=256)
    classifier = FinetuneClassifier(proto)
    path = os.path.join(config.MODEL_DIR, "finetune_model.pth")
    classifier.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    classifier.to(device)
    classifier.eval()
    return classifier


def load_proto_model(device):
    """Load the prototypical model."""
    model = PrototypicalNet(embedding_dim=256)
    path = os.path.join(config.MODEL_DIR, "proto_model.pth")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def evaluate_classifier(classifier, test_samples, device):
    """Run fine-tuned classifier on test set, return predictions and labels."""
    transform = get_val_transform()
    dataset = DefectDataset(test_samples, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = classifier(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_accuracy_vs_occurrence(preds, labels, save_path):
    """Plot per-class accuracy vs number of samples (class occurrence)."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Compute per-class accuracy and counts
    classes = sorted(set(labels))
    class_acc = []
    class_counts = []
    class_names_used = []

    for c in classes:
        mask = labels == c
        acc = (preds[mask] == labels[mask]).mean()
        class_acc.append(acc * 100)
        class_counts.append(mask.sum())
        class_names_used.append(config.CLASS_NAMES[c])

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, class_acc, width, label="Accuracy (%)", color="#2196F3", alpha=0.8)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, class_counts, width, label="Sample Count", color="#FF9800", alpha=0.8)
    ax2.set_ylabel("Number of Test Samples", fontsize=12)

    ax1.set_xlabel("Defect Class", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names_used, rotation=45, ha="right")
    ax1.set_title("Classification Accuracy vs Defect Class Occurrence", fontsize=14)

    # Add value labels on bars
    for bar, val in zip(bars1, class_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, class_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(preds, labels, save_path):
    """Plot confusion matrix."""
    present_classes = sorted(set(labels) | set(preds))
    display_labels = [config.CLASS_NAMES[c] for c in present_classes]

    cm = confusion_matrix(labels, preds, labels=present_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curve(proto_model, test_samples, train_samples, device, save_path):
    """
    Show how quickly the model learns:
    Use increasing numbers of support examples per class, measure accuracy on test queries.
    """
    transform = get_val_transform()

    # Group train samples by class for support
    class_train = defaultdict(list)
    for path, label in train_samples:
        class_train[label].append(path)

    # Group test samples by class for queries
    class_test = defaultdict(list)
    for path, label in test_samples:
        class_test[label].append(path)

    k_values = [1, 2, 3, 5, 10, 15, 20]
    accuracies = []
    n_trials = 5  # average over random selections

    proto_model.eval()
    with torch.no_grad():
        for k in k_values:
            trial_accs = []
            for _ in range(n_trials):
                all_preds, all_labels = [], []

                # Build support set: k examples per class
                support_paths, support_labels = [], []
                available_classes = []
                for c in sorted(class_train.keys()):
                    paths = class_train[c]
                    actual_k = min(k, len(paths))
                    selected = random.sample(paths, actual_k)
                    support_paths.extend(selected)
                    support_labels.extend([c] * actual_k)
                    available_classes.append(c)

                support_imgs = load_episode_images(support_paths, transform).to(device)
                support_labels_t = torch.tensor(support_labels, device=device)
                support_emb = proto_model(support_imgs)

                # Remap labels to 0..n-1 for prototype computation
                label_map = {c: i for i, c in enumerate(available_classes)}
                mapped_support = torch.tensor([label_map[l] for l in support_labels], device=device)
                prototypes = proto_model.compute_prototypes(support_emb, mapped_support)

                # Classify all test samples
                for c in available_classes:
                    if c not in class_test or len(class_test[c]) == 0:
                        continue
                    test_imgs = load_episode_images(class_test[c], transform).to(device)
                    test_emb = proto_model(test_imgs)
                    logits = proto_model.classify(test_emb, prototypes)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    # Map predictions back to original class labels
                    reverse_map = {i: c for c, i in label_map.items()}
                    mapped_preds = [reverse_map[p] for p in preds]
                    all_preds.extend(mapped_preds)
                    all_labels.extend([c] * len(class_test[c]))

                acc = accuracy_score(all_labels, all_preds)
                trial_accs.append(acc)

            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs)
            accuracies.append((mean_acc, std_acc))
            print(f"  K={k:3d} shots: {mean_acc:.4f} +/- {std_acc:.4f}")

    means = [a[0] * 100 for a in accuracies]
    stds = [a[1] * 100 for a in accuracies]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=5,
                linewidth=2, markersize=8, color="#2196F3")
    ax.fill_between(k_values, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)], alpha=0.2, color="#2196F3")
    ax.set_xlabel("Number of Support Examples per Class (K-shot)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Learning Speed: Accuracy vs Number of Examples", fontsize=14)
    ax.axhline(y=85, color="r", linestyle="--", alpha=0.7, label="85% target")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_f1(preds, labels, save_path):
    """Plot per-class F1 scores."""
    present_classes = sorted(set(labels))
    display_labels = [config.CLASS_NAMES[c] for c in present_classes]

    from sklearn.metrics import f1_score as f1
    f1_per_class = f1(labels, preds, labels=present_classes, average=None)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(display_labels, f1_per_class * 100, color="#4CAF50", alpha=0.8)
    for bar, val in zip(bars, f1_per_class * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_xlabel("Defect Class", fontsize=12)
    ax.set_title("Per-Class F1 Score", fontsize=14)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def measure_inference_time(classifier, device):
    """Measure average inference time per image."""
    transform = get_val_transform()
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            classifier(dummy)

    import time
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            classifier(dummy)
            times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    print(f"Average inference time: {avg_time:.1f} ms per image")
    return avg_time


def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    device = torch.device(config.DEVICE)

    # Load data splits
    splits = torch.load(os.path.join(config.OUTPUT_DIR, "test_samples.pth"), weights_only=False)
    test_samples = splits["test"]
    train_samples = splits["train"]

    print(f"Test samples: {len(test_samples)}")

    # Load models
    classifier = load_finetune_model(device)
    proto_model = load_proto_model(device)

    # 1. Evaluate fine-tuned classifier
    print("\n--- Fine-tuned Classifier Evaluation ---")
    preds, labels = evaluate_classifier(classifier, test_samples, device)

    overall_acc = accuracy_score(labels, preds)
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc * 100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=[config.CLASS_NAMES[i] for i in sorted(set(labels))],
        digits=4,
    ))

    # 2. Plots
    plot_accuracy_vs_occurrence(
        preds, labels,
        os.path.join(config.PLOT_DIR, "accuracy_vs_occurrence.png"),
    )
    plot_confusion_matrix(
        preds, labels,
        os.path.join(config.PLOT_DIR, "confusion_matrix.png"),
    )
    plot_per_class_f1(
        preds, labels,
        os.path.join(config.PLOT_DIR, "per_class_f1.png"),
    )

    # 3. Learning curve (prototypical model)
    print("\n--- Learning Curve (Prototypical Model) ---")
    plot_learning_curve(
        proto_model, test_samples, train_samples, device,
        os.path.join(config.PLOT_DIR, "learning_curve.png"),
    )

    # 4. Inference time
    print("\n--- Inference Time ---")
    measure_inference_time(classifier, device)


if __name__ == "__main__":
    main()
