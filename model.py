"""Prototypical Network with ConvNeXt-Tiny backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


class PrototypicalNet(nn.Module):
    """
    Prototypical Network for few-shot defect classification.

    Architecture:
        ConvNeXt-Tiny (pretrained) → global pool → projection → L2-normalized embedding

    Classification:
        Compute class prototypes as mean embeddings of support set,
        then classify query samples by nearest prototype (Euclidean distance).
    """

    def __init__(self, backbone_name=config.BACKBONE, embedding_dim=256, freeze_backbone=False):
        super().__init__()

        # Load pretrained backbone (remove classifier head)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features  # 768 for convnext_tiny

        # Projection head: maps backbone features to embedding space
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Extract L2-normalized embeddings."""
        features = self.backbone(x)  # (B, backbone_dim)
        embeddings = self.projector(features)  # (B, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def compute_prototypes(self, support_embeddings, support_labels):
        """
        Compute class prototypes as the mean embedding per class.

        Args:
            support_embeddings: (N_support, embed_dim)
            support_labels: (N_support,) with values in 0..n_way-1

        Returns:
            prototypes: (n_way, embed_dim)
        """
        n_way = support_labels.max().item() + 1
        prototypes = torch.zeros(n_way, support_embeddings.size(1),
                                 device=support_embeddings.device)
        for c in range(n_way):
            mask = support_labels == c
            prototypes[c] = support_embeddings[mask].mean(dim=0)
        return F.normalize(prototypes, p=2, dim=-1)

    def classify(self, query_embeddings, prototypes, temperature=10.0):
        """
        Classify queries by distance to prototypes.

        Returns logits (negative distances scaled by temperature).
        """
        # Cosine similarity (since embeddings are L2-normalized)
        logits = temperature * torch.mm(query_embeddings, prototypes.t())
        return logits


class FinetuneClassifier(nn.Module):
    """
    Wraps the prototypical backbone with a standard classification head
    for stage-2 fine-tuning on all data with focal loss.
    """

    def __init__(self, proto_net, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.backbone = proto_net.backbone
        self.projector = proto_net.projector
        embed_dim = proto_net.projector[-1].out_features
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projector(features)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        logits = self.classifier(embeddings)
        return logits

    def get_embeddings(self, x):
        features = self.backbone(x)
        embeddings = self.projector(features)
        return F.normalize(embeddings, p=2, dim=-1)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights tensor

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
