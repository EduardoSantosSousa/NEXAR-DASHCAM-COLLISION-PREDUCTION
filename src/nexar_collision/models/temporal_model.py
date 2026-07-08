"""CNN + recurrent model definitions for temporal collision prediction."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class CnnRnnCollisionModel(nn.Module):
    """Encode each frame with a CNN, then classify a causal frame sequence."""

    def __init__(
        self,
        num_classes: int = 2,
        cnn_backbone: str = "resnet18",
        pretrained: bool = True,
        rnn_type: str = "gru",
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        freeze_cnn: bool = True,
    ):
        super().__init__()
        if cnn_backbone != "resnet18":
            raise ValueError("Only resnet18 is currently supported.")
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be either 'gru' or 'lstm'.")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        cnn = models.resnet18(weights=weights)
        feature_dim = cnn.fc.in_features
        cnn.fc = nn.Identity()

        self.cnn = cnn
        self.freeze_cnn = freeze_cnn
        if freeze_cnn:
            for parameter in self.cnn.parameters():
                parameter.requires_grad = False

        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_class = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_multiplier, num_classes),
        )

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * sequence_length, channels, height, width)

        if self.freeze_cnn:
            with torch.no_grad():
                flat_features = self.cnn(flat_images)
        else:
            flat_features = self.cnn(flat_images)

        return flat_features.reshape(batch_size, sequence_length, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode_frames(images)
        sequence_output, _ = self.rnn(features)
        final_output = sequence_output[:, -1, :]
        return self.classifier(final_output)


def build_temporal_model(
    num_classes: int = 2,
    cnn_backbone: str = "resnet18",
    pretrained: bool = True,
    rnn_type: str = "gru",
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
    bidirectional: bool = False,
    freeze_cnn: bool = True,
) -> nn.Module:
    """Create a CNN + GRU/LSTM model for causal frame sequences."""
    return CnnRnnCollisionModel(
        num_classes=num_classes,
        cnn_backbone=cnn_backbone,
        pretrained=pretrained,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        freeze_cnn=freeze_cnn,
    )
