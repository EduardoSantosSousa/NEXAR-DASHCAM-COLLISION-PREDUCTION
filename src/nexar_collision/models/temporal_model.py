"""CNN + recurrent model definitions for temporal collision predictions"""

from __future__ import annotations
import torch
from torch import nn
from torchvision import models

CNN_TRAIN_POLICIES = {"frozen", "layer4", "full"}

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
            freeze_cnn: bool | None = None,
            cnn_train_policy: str = "frozen",
    ):
        super().__init__()
        if cnn_backbone != "resnet18":
            raise ValueError ("Only resnet18 is currently supported.")
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be either 'gru' or 'lstm'.")
        
        # Backward compatibility with older checkpoints/configs.
        if freeze_cnn is not None:
            cnn_train_policy = "frozen" if freeze_cnn else "full"
        if cnn_train_policy not in CNN_TRAIN_POLICIES:
            raise ValueError(
                "cnn_train_policy must be one of: "
                f"{sorted(CNN_TRAIN_POLICIES)}"
            )

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        cnn = models.resnet18(weights=weights)
        feature_dim = cnn.fc.in_features
        cnn.fc = nn.Identity()

        self.cnn = cnn 
        self.cnn_train_policy = cnn_train_policy
        self.freeze_cnn = cnn_train_policy == "frozen"
        self._configure_cnn_trainability()

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

    def _configure_cnn_trainability(self) -> None:
        """Set trainable CNN blocks according to the selected policy."""
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        if self.cnn_train_policy == "frozen":
            return

        if self.cnn_train_policy == "full":
            for parameter in self.cnn.parameters():
                parameter.requires_grad = True
            return

        if self.cnn_train_policy == "layer4":
            for parameter in self.cnn.layer4.parameters():
                parameter.requires_grad = True
            return

        raise ValueError(f"Unsupported CNN train policy: {self.cnn_train_policy}")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._set_cnn_training_mode()
        return self

    def _set_cnn_training_mode(self) -> None:
        if self.cnn_train_policy == "full":
            self.cnn.train()
            return

        if self.cnn_train_policy == "frozen":
            self.cnn.eval()
            return

        if self.cnn_train_policy == "layer4":
            self.cnn.eval()
            self.cnn.layer4.train()

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, sequece_lenght, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * sequece_lenght, channels, height, width)

        if self.freeze_cnn:
            with torch.no_grad():
                flat_features = self.cnn(flat_images)
        else:
            flat_features = self.cnn(flat_images)

        return flat_features.reshape(batch_size, sequece_lenght, -1)

    def forward(self, images:torch.Tensor) -> torch.Tensor:
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
        bidirectional : bool = False,
        freeze_cnn: bool | None = None,
        cnn_train_policy: str = "frozen",
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
        cnn_train_policy=cnn_train_policy,
    )                       
