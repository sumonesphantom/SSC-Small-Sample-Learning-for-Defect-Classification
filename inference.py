"""
Single-image inference for the defect classification application.

Usage:
    python inference.py --image path/to/image.png
    python inference.py --image path/to/image.png --mode proto --k_shot 5
"""

import argparse
import os
import time

import torch
import numpy as np
from PIL import Image

import config
from dataset import get_val_transform, load_episode_images, get_all_samples
from model import PrototypicalNet, FinetuneClassifier
from collections import defaultdict


def load_models(device):
    """Load both prototypical and fine-tuned models."""
    proto = PrototypicalNet(embedding_dim=256)
    proto_path = os.path.join(config.MODEL_DIR, "proto_model.pth")
    proto.load_state_dict(torch.load(proto_path, map_location=device, weights_only=True))
    proto.to(device).eval()

    ft_classifier = FinetuneClassifier(proto)
    ft_path = os.path.join(config.MODEL_DIR, "finetune_model.pth")
    ft_classifier.load_state_dict(torch.load(ft_path, map_location=device, weights_only=True))
    ft_classifier.to(device).eval()

    return proto, ft_classifier


def classify_finetune(classifier, image_path, device):
    """Classify using the fine-tuned classifier."""
    transform = get_val_transform()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    elapsed = time.time() - start

    pred_idx = probs.argmax().item()
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    return pred_class, confidence, probs.cpu().numpy(), elapsed


def classify_prototypical(proto_model, image_path, device, k_shot=5):
    """Classify using prototypical model with k-shot support from training data."""
    transform = get_val_transform()

    # Build support set from training data
    samples, _ = get_all_samples()
    class_samples = defaultdict(list)
    for path, label in samples:
        class_samples[label].append(path)

    support_paths, support_labels = [], []
    import random
    for c in range(config.NUM_CLASSES):
        paths = class_samples[c]
        k = min(k_shot, len(paths))
        selected = random.sample(paths, k)
        support_paths.extend(selected)
        support_labels.extend([c] * k)

    # Compute prototypes
    support_imgs = load_episode_images(support_paths, transform).to(device)
    support_labels_t = torch.tensor(support_labels, device=device)

    img = Image.open(image_path).convert("RGB")
    query_tensor = transform(img).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        support_emb = proto_model(support_imgs)
        query_emb = proto_model(query_tensor)
        prototypes = proto_model.compute_prototypes(support_emb, support_labels_t)
        logits = proto_model.classify(query_emb, prototypes)
        probs = torch.softmax(logits, dim=1)[0]
    elapsed = time.time() - start

    pred_idx = probs.argmax().item()
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    return pred_class, confidence, probs.cpu().numpy(), elapsed


def main():
    parser = argparse.ArgumentParser(description="Defect Classification Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mode", default="finetune", choices=["finetune", "proto"],
                        help="Inference mode: finetune (default) or proto (prototypical)")
    parser.add_argument("--k_shot", type=int, default=5,
                        help="K-shot for prototypical mode (default: 5)")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    proto_model, classifier = load_models(device)

    if args.mode == "finetune":
        pred_class, confidence, probs, elapsed = classify_finetune(
            classifier, args.image, device)
    else:
        pred_class, confidence, probs, elapsed = classify_prototypical(
            proto_model, args.image, device, args.k_shot)

    print(f"\nImage: {args.image}")
    print(f"Mode: {args.mode}")
    print(f"Prediction: {pred_class} (confidence: {confidence:.4f})")
    print(f"Inference time: {elapsed * 1000:.1f} ms")
    print(f"\nAll class probabilities:")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name:>10s}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
