"""Baseline CNN model definition."""

from __future__ import annotations

from torch import nn
from torchvision import models


def build_baseline_cnn(
    num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    """Create a ResNet18 baseline for binary collision prediction."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
