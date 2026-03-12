"""
Two-stage training pipeline:
    Stage 1: Episodic prototypical training (few-shot learning)
    Stage 2: Full fine-tuning with focal loss + balanced sampling
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import (
    get_all_samples, split_dataset, EpisodicSampler,
    load_episode_images, get_train_transform, get_val_transform,
    get_finetune_loaders,
)
from model import PrototypicalNet, FinetuneClassifier, FocalLoss


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_prototypical(model, train_samples, val_samples, device):
    """Stage 1: Episodic prototypical training."""
    print("\n" + "=" * 60)
    print("STAGE 1: Prototypical (Episodic) Training")
    print("=" * 60)

    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.LEARNING_RATE * 0.1},
            {"params": model.projector.parameters(), "lr": config.LEARNING_RATE},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_acc, n_episodes = 0.0, 0.0, 0

        sampler = EpisodicSampler(
            train_samples, config.N_WAY, config.K_SHOT_SUPPORT,
            config.Q_QUERY, config.NUM_EPISODES_TRAIN,
        )

        for support_paths, support_labels, query_paths, query_labels, _ in sampler:
            support_imgs = load_episode_images(support_paths, train_transform).to(device)
            query_imgs = load_episode_images(query_paths, train_transform).to(device)
            support_labels_t = torch.tensor(support_labels, device=device)
            query_labels_t = torch.tensor(query_labels, device=device)

            # Forward
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)

            # Compute prototypes and classify
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

        scheduler.step()
        avg_loss = epoch_loss / n_episodes
        avg_acc = epoch_acc / n_episodes

        # Validation
        val_acc = evaluate_prototypical(model, val_samples, val_transform, device)

        print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
              f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"\nBest prototypical val accuracy: {best_val_acc:.4f}")
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
            {"params": classifier.backbone.parameters(), "lr": config.FINETUNE_LR * 0.1},
            {"params": classifier.projector.parameters(), "lr": config.FINETUNE_LR},
            {"params": classifier.classifier.parameters(), "lr": config.FINETUNE_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.FINETUNE_EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels_batch in train_loader:
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

        scheduler.step()

        # Validation
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                logits = classifier(images)
                val_correct += (logits.argmax(1) == labels_batch).sum().item()
                val_total += images.size(0)

        train_acc = correct / total
        val_acc = val_correct / val_total
        avg_loss = total_loss / total

        print(f"Epoch {epoch:3d}/{config.FINETUNE_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

    if best_state:
        classifier.load_state_dict(best_state)
    print(f"\nBest fine-tune val accuracy: {best_val_acc:.4f}")
    return classifier


def main():
    set_seed()
    os.makedirs(config.MODEL_DIR, exist_ok=True)

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

    # Stage 1: Prototypical training
    proto_model = PrototypicalNet(embedding_dim=256, freeze_backbone=False).to(device)
    proto_model = train_prototypical(proto_model, train_samples, val_samples, device)

    # Save prototypical model
    proto_path = os.path.join(config.MODEL_DIR, "proto_model.pth")
    torch.save(proto_model.state_dict(), proto_path)
    print(f"Saved prototypical model to {proto_path}")

    # Stage 2: Fine-tuning
    classifier = train_finetune(proto_model, train_samples, val_samples, device)

    # Save fine-tuned model
    ft_path = os.path.join(config.MODEL_DIR, "finetune_model.pth")
    torch.save(classifier.state_dict(), ft_path)
    print(f"Saved fine-tuned model to {ft_path}")

    # Save test samples for evaluation
    test_path = os.path.join(config.OUTPUT_DIR, "test_samples.pth")
    torch.save({"test": test_samples, "train": train_samples, "val": val_samples}, test_path)
    print(f"Saved data splits to {test_path}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal training time: {time.time() - start:.1f}s")
