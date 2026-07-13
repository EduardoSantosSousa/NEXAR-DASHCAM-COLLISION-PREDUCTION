"""Baseline CNN model definition."""

from __future__ import annotations

from torch import nn
from torchvision import models


SUPPORTED_BASELINE_BACKBONES = ("resnet18", "efficientnet_b0", "convnext_tiny")


def build_baseline_cnn(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    """Create a CNN baseline for binary collision prediction."""
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported backbone: {backbone}. "
        f"Available backbones: {list(SUPPORTED_BASELINE_BACKBONES)}"
    )


def freeze_baseline_backbone(model: nn.Module, backbone: str) -> None:
    """Freeze feature layers and leave only the classifier head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if backbone == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if backbone in {"efficientnet_b0", "convnext_tiny"}:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(
        f"Unsupported backbone: {backbone}. "
        f"Available backbones: {list(SUPPORTED_BASELINE_BACKBONES)}"
    )


def unfreeze_last_feature_blocks(
    model: nn.Module,
    backbone: str,
    last_n_blocks: int,
) -> None:
    """Unfreeze the last N feature blocks after freezing the backbone."""
    if last_n_blocks <= 0:
        return

    if backbone == "resnet18":
        ordered_blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    elif backbone in {"efficientnet_b0", "convnext_tiny"}:
        ordered_blocks = list(model.features.children())
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Available backbones: {list(SUPPORTED_BASELINE_BACKBONES)}"
        )

    for block in ordered_blocks[-last_n_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def configure_trainable_baseline_layers(
    model: nn.Module,
    backbone: str,
    freeze_backbone: bool = False,
    unfreeze_last_n_blocks: int = 0,
) -> None:
    """Configure which baseline layers are trainable for a product experiment."""
    if unfreeze_last_n_blocks < 0:
        raise ValueError("unfreeze_last_n_blocks must be greater than or equal to 0.")

    if freeze_backbone or unfreeze_last_n_blocks > 0:
        freeze_baseline_backbone(model, backbone)
        unfreeze_last_feature_blocks(model, backbone, unfreeze_last_n_blocks)
