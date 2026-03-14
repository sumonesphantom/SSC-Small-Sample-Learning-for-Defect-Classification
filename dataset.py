"""Dataset utilities for defect classification."""

import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config


# ---------------------------------------------------------------------------
# Global image cache — loads all images into RAM once to avoid repeated disk I/O
# ---------------------------------------------------------------------------
_IMAGE_CACHE = {}


def _cache_image(path):
    """Load image into cache if not already there. Returns the PIL Image."""
    if path not in _IMAGE_CACHE:
        _IMAGE_CACHE[path] = Image.open(path).convert("RGB")
    return _IMAGE_CACHE[path]


def preload_images(samples):
    """Pre-load all images into RAM. Call once before training."""
    total = len(samples)
    print(f"Pre-loading {total} images into RAM...", end=" ", flush=True)
    for i, (path, _) in enumerate(samples):
        _cache_image(path)
        if (i + 1) % 1000 == 0:
            print(f"{i+1}/{total}", end=" ", flush=True)
    print("Done.")


def get_cached_image(path):
    """Get image from cache (must call preload_images first)."""
    if path in _IMAGE_CACHE:
        return _IMAGE_CACHE[path].copy()  # copy so transforms don't mutate cache
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_all_samples():
    """Load all image paths and labels from the Data directory."""
    samples = []
    class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        idx = class_to_idx[class_name]
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                samples.append((os.path.join(class_dir, fname), idx))

    return samples, class_to_idx


def split_dataset(samples, val_ratio=config.VAL_RATIO, test_ratio=config.TEST_RATIO, seed=config.RANDOM_SEED):
    """Stratified split into train/val/test."""
    paths, labels = zip(*samples)
    paths, labels = list(paths), list(labels)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )

    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))
    return train, val, test


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DefectDataset(Dataset):
    """Standard classification dataset (uses cache if available)."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = get_cached_image(path)
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Episodic sampling for prototypical training
# ---------------------------------------------------------------------------

class EpisodicSampler:
    """Generates N-way K-shot episodes for prototypical training."""

    def __init__(self, samples, n_way, k_shot, q_query, num_episodes):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes

        self.class_samples = defaultdict(list)
        for path, label in samples:
            self.class_samples[label].append(path)

        self.available_classes = [
            c for c, paths in self.class_samples.items()
            if len(paths) >= 2
        ]

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            if len(self.available_classes) <= self.n_way:
                episode_classes = self.available_classes[:]
            else:
                episode_classes = random.sample(self.available_classes, self.n_way)

            support_paths, support_labels = [], []
            query_paths, query_labels = [], []

            for i, cls in enumerate(episode_classes):
                cls_paths = self.class_samples[cls]
                n_available = len(cls_paths)

                k = min(self.k_shot, max(1, n_available // 2))
                q = min(self.q_query, n_available - k)
                if q < 1:
                    k = max(1, n_available - 1)
                    q = n_available - k

                selected = random.sample(cls_paths, k + q)
                support_paths.extend(selected[:k])
                support_labels.extend([i] * k)
                query_paths.extend(selected[k:k + q])
                query_labels.extend([i] * q)

            yield (support_paths, support_labels, query_paths, query_labels, episode_classes)


def load_episode_images(paths, transform):
    """Load and transform a batch of images from paths (uses cache)."""
    images = []
    for p in paths:
        img = get_cached_image(p)
        if transform:
            img = transform(img)
        images.append(img)
    return torch.stack(images)


# ---------------------------------------------------------------------------
# Fine-tuning data loaders
# ---------------------------------------------------------------------------

def get_weighted_sampler(samples):
    """Create a WeightedRandomSampler to handle class imbalance."""
    labels = [s[1] for s in samples]
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_finetune_loaders(train_samples, val_samples):
    """Get DataLoaders for fine-tuning stage with balanced sampling."""
    train_ds = DefectDataset(train_samples, get_train_transform())
    val_ds = DefectDataset(val_samples, get_val_transform())

    sampler = get_weighted_sampler(train_samples)
    train_loader = DataLoader(
        train_ds, batch_size=config.FINETUNE_BATCH_SIZE,
        sampler=sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.FINETUNE_BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader
