"""
Two-stage training pipeline:
    Stage 1: Episodic prototypical training (few-shot learning)
    Stage 2: Full fine-tuning with focal loss + balanced sampling

Callbacks:
    - ModelCheckpoint: saves top-K models by val_acc
    - EarlyStopping: halts training after patience epochs with no improvement
    - CSVLogger: logs all metrics per epoch to CSV
    - LRTracker: records learning rates per param group
"""

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import (
    get_all_samples, split_dataset, preload_images, EpisodicSampler,
    load_episode_images, get_train_transform, get_val_transform,
    get_finetune_loaders,
)
from model import PrototypicalNet, FinetuneClassifier, FocalLoss
from callbacks import CallbackRunner


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def progress_bar(current, total, prefix="", suffix="", length=30):
    """Simple inline progress bar."""
    pct = current / total
    filled = int(length * pct)
    bar = "█" * filled + "░" * (length - filled)
    sys.stdout.write(f"\r  {prefix} |{bar}| {current}/{total} {suffix}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def train_prototypical(model, train_samples, val_samples, device):
    """Stage 1: Episodic prototypical training."""
    print("\n" + "=" * 60)
    print("STAGE 1: Prototypical (Episodic) Training")
    print("=" * 60)

    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.LEARNING_RATE * 0.1, "name": "backbone"},
            {"params": model.projector.parameters(), "lr": config.LEARNING_RATE, "name": "projector"},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    callbacks = CallbackRunner(stage="proto")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss, epoch_acc, n_episodes = 0.0, 0.0, 0

        sampler = EpisodicSampler(
            train_samples, config.N_WAY, config.K_SHOT_SUPPORT,
            config.Q_QUERY, config.NUM_EPISODES_TRAIN,
        )

        total_episodes = config.NUM_EPISODES_TRAIN
        for support_paths, support_labels, query_paths, query_labels, _ in sampler:
            support_imgs = load_episode_images(support_paths, train_transform).to(device)
            query_imgs = load_episode_images(query_paths, train_transform).to(device)
            support_labels_t = torch.tensor(support_labels, device=device)
            query_labels_t = torch.tensor(query_labels, device=device)

            support_emb = model(support_imgs)
            query_emb = model(query_imgs)

            prototypes = model.compute_prototypes(support_emb, support_labels_t)
            logits = model.classify(query_emb, prototypes)

            loss = F.cross_entropy(logits, query_labels_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            acc = (logits.argmax(dim=1) == query_labels_t).float().mean().item()
            epoch_loss += loss.item()
            epoch_acc += acc
            n_episodes += 1

            progress_bar(n_episodes, total_episodes,
                         prefix=f"Epoch {epoch:2d}",
                         suffix=f"loss={loss.item():.3f} acc={acc:.3f}")

        scheduler.step()
        epoch_time = time.time() - epoch_start

        avg_loss = epoch_loss / n_episodes
        avg_acc = epoch_acc / n_episodes

        # Validation
        val_acc = evaluate_prototypical(model, val_samples, val_transform, device)

        print(f"  Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

        metrics = {
            "train_loss": round(avg_loss, 6),
            "train_acc": round(avg_acc, 6),
            "val_acc": round(val_acc, 6),
            "epoch_time": round(epoch_time, 2),
        }
        callbacks.on_epoch_end(epoch, model, metrics, optimizer)

        if callbacks.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    callbacks.load_best_model(model, device)
    return model


def evaluate_prototypical(model, val_samples, transform, device):
    """Evaluate prototypical accuracy on validation set."""
    model.eval()
    sampler = EpisodicSampler(
        val_samples, config.N_WAY, config.K_SHOT_SUPPORT,
        config.Q_QUERY, config.NUM_EPISODES_VAL,
    )
    total_acc, n = 0.0, 0
    with torch.no_grad():
        for support_paths, support_labels, query_paths, query_labels, _ in sampler:
            support_imgs = load_episode_images(support_paths, transform).to(device)
            query_imgs = load_episode_images(query_paths, transform).to(device)
            support_labels_t = torch.tensor(support_labels, device=device)
            query_labels_t = torch.tensor(query_labels, device=device)

            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            prototypes = model.compute_prototypes(support_emb, support_labels_t)
            logits = model.classify(query_emb, prototypes)

            acc = (logits.argmax(dim=1) == query_labels_t).float().mean().item()
            total_acc += acc
            n += 1
    return total_acc / max(n, 1)


def train_finetune(model, train_samples, val_samples, device):
    """Stage 2: Standard fine-tuning with focal loss and balanced sampling."""
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tuning with Focal Loss")
    print("=" * 60)

    classifier = FinetuneClassifier(model).to(device)
    train_loader, val_loader = get_finetune_loaders(train_samples, val_samples)

    # Compute class weights for focal loss
    labels = [s[1] for s in train_samples]
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
    alpha = torch.tensor(class_weights, device=device)

    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    optimizer = AdamW(
        [
            {"params": classifier.backbone.parameters(), "lr": config.FINETUNE_LR * 0.1, "name": "backbone"},
            {"params": classifier.projector.parameters(), "lr": config.FINETUNE_LR, "name": "projector"},
            {"params": classifier.classifier.parameters(), "lr": config.FINETUNE_LR, "name": "head"},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.FINETUNE_EPOCHS, eta_min=1e-6)

    callbacks = CallbackRunner(stage="finetune")
    total_batches = len(train_loader)

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        epoch_start = time.time()
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels_batch) in enumerate(train_loader, 1):
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            logits = classifier(images)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels_batch).sum().item()
            total += images.size(0)

            progress_bar(batch_idx, total_batches,
                         prefix=f"Epoch {epoch:2d}",
                         suffix=f"loss={loss.item():.3f}")

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Validation
        classifier.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                logits = classifier(images)
                val_correct += (logits.argmax(1) == labels_batch).sum().item()
                val_total += images.size(0)
                val_loss_sum += criterion(logits, labels_batch).item() * images.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total
        train_loss = total_loss / total
        val_loss = val_loss_sum / val_total

        print(f"  Epoch {epoch:3d}/{config.FINETUNE_EPOCHS} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

        metrics = {
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_time": round(epoch_time, 2),
        }
        callbacks.on_epoch_end(epoch, model=classifier, metrics=metrics, optimizer=optimizer)

        if callbacks.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    callbacks.load_best_model(classifier, device)
    return classifier


def main():
    set_seed()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load and split data
    samples, class_to_idx = get_all_samples()
    print(f"Total samples: {len(samples)}")
    for name, idx in class_to_idx.items():
        count = sum(1 for _, l in samples if l == idx)
        print(f"  {name}: {count}")

    train_samples, val_samples, test_samples = split_dataset(samples)
    print(f"\nSplit: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Pre-load all images into RAM (avoids repeated disk I/O during training)
    all_samples = train_samples + val_samples + test_samples
    preload_images(all_samples)

    # Stage 1: Prototypical training
    proto_model = PrototypicalNet(embedding_dim=256, freeze_backbone=False).to(device)
    proto_model = train_prototypical(proto_model, train_samples, val_samples, device)

    proto_path = os.path.join(config.MODEL_DIR, "proto_model.pth")
    torch.save(proto_model.state_dict(), proto_path)
    print(f"Saved prototypical model to {proto_path}")

    # Stage 2: Fine-tuning
    classifier = train_finetune(proto_model, train_samples, val_samples, device)

    ft_path = os.path.join(config.MODEL_DIR, "finetune_model.pth")
    torch.save(classifier.state_dict(), ft_path)
    print(f"Saved fine-tuned model to {ft_path}")

    # Save data splits for evaluation
    test_path = os.path.join(config.OUTPUT_DIR, "test_samples.pth")
    torch.save({"test": test_samples, "train": train_samples, "val": val_samples}, test_path)
    print(f"Saved data splits to {test_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best checkpoints and logs saved in: {config.OUTPUT_DIR}")
    print(f"  Run 'python evaluate.py' to generate plots and metrics.")
    print("=" * 60)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal training time: {time.time() - start:.1f}s")
