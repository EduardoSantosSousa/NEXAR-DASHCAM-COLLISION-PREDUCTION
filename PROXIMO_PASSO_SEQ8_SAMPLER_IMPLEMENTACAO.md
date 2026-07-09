# Proximo passo: GRU seq8 com sampler e fine-tuning parcial

Este documento descreve o proximo passo recomendado para o projeto
`nexar-dashcam-collision-prediction`.

O objetivo e transformar a pipeline sequencial ja existente em uma rodada mais
forte de experimento:

```text
temporal_alert_224_pretrained_gru_seq8_sampler
```

Esse experimento deve atacar dois problemas observados ate aqui:

1. O dataset temporal e desbalanceado.
2. O primeiro CNN + GRU/LSTM usou encoder congelado e sequencias curtas
   (`sequence_length=4`).

O plano e implementar:

- `WeightedRandomSampler` para treino sequencial.
- Fine-tuning parcial da ResNet18, liberando apenas `layer4`.
- Learning rates separados para CNN e cabeca temporal.
- Opcoes de loss (`cross_entropy` e `focal`), mantendo o experimento principal
  com `weighted_sampler + cross_entropy`.
- Logging dos novos parametros no MLflow.

## Estado atual usado como baseline

O melhor ponto atual e:

```text
Modelo: temporal_alert_224_pretrained_best
Pos-processamento: 2 consecutive frames
Threshold: 0.13
Precision: 0.727
Recall: 0.800
False alarm rate: 0.300
Mean alert error: -7.991 s
```

O novo experimento deve ser comparado contra esse baseline.

## Arquivos que serao alterados

```text
src/nexar_collision/models/temporal_model.py
src/nexar_collision/models/train_sequence.py
scripts/train_sequence_model.py
```

Recomendacao: aplicar as mudancas abaixo em uma branch ou commit separado,
porque existem notebooks modificados localmente no repositorio.

## Passo 1 - Atualizar `temporal_model.py`

Substitua o conteudo de:

```text
src/nexar_collision/models/temporal_model.py
```

por:

```python
"""CNN + recurrent model definitions for temporal collision prediction."""

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
            raise ValueError("Only resnet18 is currently supported.")
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
```

## Passo 2 - Atualizar `train_sequence.py`

Substitua o conteudo de:

```text
src/nexar_collision/models/train_sequence.py
```

por:

```python
"""Training utilities for CNN + GRU/LSTM temporal frame classifiers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from nexar_collision.data.dataset import FrameSequenceCollisionDataset
from nexar_collision.evaluation.metrics import compute_metrics
from nexar_collision.models.temporal_model import build_temporal_model
from nexar_collision.models.train import (
    build_transforms,
    default_best_checkpoint_path,
    is_improvement,
    load_manifest_with_optional_split,
    monitor_value_from_metrics,
    resolve_device,
    save_figures,
    seed_everything,
    split_by_video,
)
from nexar_collision.tracking.mlflow_utils import (
    DEFAULT_EXPERIMENT_NAME,
    log_artifacts,
    log_metrics,
    log_params,
    mlflow_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMBALANCE_STRATEGIES = {"none", "class_weight", "weighted_sampler"}
LOSS_NAMES = {"cross_entropy", "focal"}


@dataclass
class SequenceTrainingConfig:
    manifest_path: Path = PROJECT_ROOT / "data" / "interim" / "temporal_frames_224_manifest.csv"
    predictions_path: Path = PROJECT_ROOT / "outputs" / "predictions" / "sequence_frame_predictions.csv"
    metrics_path: Path = PROJECT_ROOT / "models" / "reports" / "sequence_metrics.json"
    checkpoint_path: Path = PROJECT_ROOT / "models" / "checkpoints" / "sequence_model.pt"
    best_checkpoint_path: Path | None = None
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    split_manifest_path: Path | None = None
    split_column: str = "split"
    train_split: str = "train"
    val_split: str = "val"
    figure_prefix: str = "sequence"
    sequence_length: int = 4
    train_sequence_stride: int = 1
    val_sequence_stride: int = 1
    batch_size: int = 8
    epochs: int = 4
    learning_rate: float = 1e-4
    head_learning_rate: float | None = None
    cnn_learning_rate: float | None = None
    weight_decay: float = 1e-2
    val_size: float = 0.2
    random_state: int = 42
    num_workers: int = 0
    pretrained: bool = True
    rnn_type: str = "gru"
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    cnn_train_policy: str = "frozen"
    imbalance_strategy: str = "class_weight"
    loss_name: str = "cross_entropy"
    focal_gamma: float = 2.0
    device: str = "auto"
    monitor_metric: str = "roc_auc"
    monitor_mode: str = "max"
    patience: int = 2
    min_delta: float = 0.0
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = DEFAULT_EXPERIMENT_NAME
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None


class FocalLoss(nn.Module):
    """Multi-class focal loss for imbalanced classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        probabilities = torch.softmax(logits, dim=1)
        target_probabilities = probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)
        focal_factor = (1.0 - target_probabilities).clamp(min=1e-8) ** self.gamma
        return torch.mean(focal_factor * ce_loss)


def validate_config(config: SequenceTrainingConfig) -> None:
    if config.imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            "imbalance_strategy must be one of: "
            f"{sorted(IMBALANCE_STRATEGIES)}"
        )
    if config.loss_name not in LOSS_NAMES:
        raise ValueError(f"loss_name must be one of: {sorted(LOSS_NAMES)}")


def model_config_from_training_config(config: SequenceTrainingConfig) -> dict[str, object]:
    return {
        "num_classes": 2,
        "cnn_backbone": "resnet18",
        "pretrained": config.pretrained,
        "rnn_type": config.rnn_type,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "bidirectional": config.bidirectional,
        "cnn_train_policy": config.cnn_train_policy,
    }


def save_sequence_checkpoint(
    model: nn.Module,
    path: Path,
    config: SequenceTrainingConfig,
    epoch: int,
    monitor_value: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config_from_training_config(config),
            "sequence_length": config.sequence_length,
            "epoch": epoch,
            "monitor_metric": config.monitor_metric,
            "monitor_value": monitor_value,
        },
        path,
    )


def build_balanced_class_weights(
    train_df: pd.DataFrame,
    device: torch.device,
) -> torch.Tensor | None:
    class_counts = train_df["target"].value_counts().sort_index()
    if len(class_counts) != 2:
        return None

    total = float(class_counts.sum())
    return torch.tensor(
        [total / (2.0 * class_counts.get(index, 1)) for index in [0, 1]],
        dtype=torch.float32,
        device=device,
    )


def build_sequence_loss(
    train_df: pd.DataFrame,
    config: SequenceTrainingConfig,
    device: torch.device,
) -> nn.Module:
    class_weights = None
    if config.imbalance_strategy == "class_weight":
        class_weights = build_balanced_class_weights(train_df, device)

    if config.loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    if config.loss_name == "focal":
        return FocalLoss(gamma=config.focal_gamma, weight=class_weights)

    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def sequence_targets(dataset: FrameSequenceCollisionDataset) -> list[int]:
    targets = []
    for sequence_positions in dataset.sequence_index:
        end_position = sequence_positions[-1]
        target = int(dataset.manifest.iloc[end_position]["target"])
        targets.append(target)
    return targets


def build_weighted_sequence_sampler(
    dataset: FrameSequenceCollisionDataset,
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    targets = sequence_targets(dataset)
    class_counts = np.bincount(targets, minlength=2)

    if np.any(class_counts == 0):
        raise ValueError(
            "WeightedRandomSampler requires both classes in the training split. "
            f"Observed counts: {class_counts.tolist()}"
        )

    sample_weights = [1.0 / float(class_counts[target]) for target in targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    distribution = {str(index): int(count) for index, count in enumerate(class_counts)}
    return sampler, distribution


def build_optimizer(
    model: nn.Module,
    config: SequenceTrainingConfig,
) -> torch.optim.Optimizer:
    head_lr = config.head_learning_rate or config.learning_rate
    cnn_lr = config.cnn_learning_rate
    if cnn_lr is None:
        cnn_lr = config.learning_rate * 0.1

    cnn_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("cnn."):
            cnn_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups = []
    if cnn_parameters:
        parameter_groups.append(
            {
                "params": cnn_parameters,
                "lr": cnn_lr,
                "weight_decay": config.weight_decay,
            }
        )
    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": head_lr,
                "weight_decay": config.weight_decay,
            }
        )

    if not parameter_groups:
        raise ValueError("No trainable parameters found for the optimizer.")

    return torch.optim.AdamW(parameter_groups)


def count_trainable_parameters(model: nn.Module) -> dict[str, int]:
    cnn_trainable = 0
    head_trainable = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        if name.startswith("cnn."):
            cnn_trainable += count
        else:
            head_trainable += count

    return {
        "trainable_parameters": cnn_trainable + head_trainable,
        "trainable_cnn_parameters": cnn_trainable,
        "trainable_head_parameters": head_trainable,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def predict_sequences(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records = []

    for batch in loader:
        images = batch["images"].to(device)
        logits = model(images)
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (scores >= 0.5).astype(int)

        for idx, score in enumerate(scores):
            records.append(
                {
                    "id": batch["video_id"][idx],
                    "timestamp": float(batch["timestamp"][idx]),
                    "target": int(batch["target"][idx]),
                    "video_target": int(batch["video_target"][idx]),
                    "duration": float(batch["duration"][idx]),
                    "score": float(score),
                    "prediction": int(preds[idx]),
                }
            )

    return pd.DataFrame(records)


def train_sequence_model(config: SequenceTrainingConfig | None = None) -> dict[str, object]:
    config = config or SequenceTrainingConfig()
    validate_config(config)
    seed_everything(config.random_state)
    device = resolve_device(config.device)
    best_checkpoint_path = config.best_checkpoint_path or default_best_checkpoint_path(
        config.checkpoint_path
    )

    manifest_df = load_manifest_with_optional_split(config)  # type: ignore[arg-type]
    train_df, val_df = split_by_video(
        manifest_df=manifest_df,
        val_size=config.val_size,
        random_state=config.random_state,
        split_column=config.split_column,
        train_split=config.train_split,
        val_split=config.val_split,
    )
    train_transform, eval_transform = build_transforms()

    train_dataset = FrameSequenceCollisionDataset(
        train_df,
        sequence_length=config.sequence_length,
        transform=train_transform,
        sequence_stride=config.train_sequence_stride,
    )
    val_dataset = FrameSequenceCollisionDataset(
        val_df,
        sequence_length=config.sequence_length,
        transform=eval_transform,
        sequence_stride=config.val_sequence_stride,
    )

    train_sampler = None
    train_sequence_target_distribution = {
        str(key): int(value)
        for key, value in pd.Series(sequence_targets(train_dataset))
        .value_counts()
        .sort_index()
        .items()
    }
    shuffle_train = True
    if config.imbalance_strategy == "weighted_sampler":
        train_sampler, train_sequence_target_distribution = build_weighted_sequence_sampler(
            train_dataset
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_temporal_model(**model_config_from_training_config(config)).to(device)
    criterion = build_sequence_loss(train_df, config, device)
    optimizer = build_optimizer(model, config)
    parameter_counts = count_trainable_parameters(model)

    config.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_epoch: int | None = None
    best_monitor_value: float | None = None
    best_frame_metrics: dict[str, float | None] | None = None
    epochs_without_improvement = 0
    early_stopped = False
    stopped_epoch: int | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        predictions_df = predict_sequences(model, val_loader, device)
        frame_metrics = compute_metrics(
            predictions_df["target"],
            predictions_df["prediction"],
            predictions_df["score"],
        )
        monitor_value = monitor_value_from_metrics(
            config.monitor_metric,
            train_loss,
            frame_metrics,
        )
        improved = is_improvement(
            current_value=monitor_value,
            best_value=best_monitor_value,
            mode=config.monitor_mode,
            min_delta=config.min_delta,
        )
        if improved:
            best_epoch = epoch
            best_monitor_value = monitor_value
            best_frame_metrics = dict(frame_metrics)
            epochs_without_improvement = 0
            save_sequence_checkpoint(
                model,
                best_checkpoint_path,
                config,
                epoch,
                monitor_value,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **frame_metrics,
                "monitor_metric": config.monitor_metric,
                "monitor_value": monitor_value,
                "is_best": improved,
            }
        )
        best_marker = " best" if improved else ""
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_f1={frame_metrics['f1']:.4f} "
            f"val_{config.monitor_metric}={monitor_value:.4f}{best_marker}"
        )

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            early_stopped = True
            stopped_epoch = epoch
            print(
                "early_stopping="
                f"epoch={epoch} monitor={config.monitor_metric} "
                f"best_epoch={best_epoch} best_value={best_monitor_value:.4f}"
            )
            break

    predictions_df = predict_sequences(model, val_loader, device)
    frame_metrics = compute_metrics(
        predictions_df["target"],
        predictions_df["prediction"],
        predictions_df["score"],
    )

    predictions_df.to_csv(config.predictions_path, index=False)
    save_figures(predictions_df, config.figures_dir, config.figure_prefix)
    save_sequence_checkpoint(
        model,
        config.checkpoint_path,
        config,
        int(history[-1]["epoch"]),
        float(history[-1]["monitor_value"]),
    )

    report = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "model_config": model_config_from_training_config(config),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "train_frames": len(train_df),
        "val_frames": len(val_df),
        "train_videos": train_df["id"].nunique(),
        "val_videos": val_df["id"].nunique(),
        "parameter_counts": parameter_counts,
        "train_target_distribution": {
            str(key): int(value)
            for key, value in train_df["target"].value_counts().sort_index().items()
        },
        "val_target_distribution": {
            str(key): int(value)
            for key, value in val_df["target"].value_counts().sort_index().items()
        },
        "train_sequence_target_distribution": train_sequence_target_distribution,
        "history": history,
        "best_epoch": best_epoch,
        "best_monitor_metric": config.monitor_metric,
        "best_monitor_mode": config.monitor_mode,
        "best_monitor_value": best_monitor_value,
        "best_frame_metrics": best_frame_metrics,
        "best_checkpoint_path": str(best_checkpoint_path),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "frame_metrics": frame_metrics,
    }

    config.metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_sequence_training_run_to_mlflow(config, report)
    return report


def log_sequence_training_run_to_mlflow(
    config: SequenceTrainingConfig,
    report: dict[str, object],
) -> None:
    run_name = config.mlflow_run_name or f"{config.figure_prefix}_train"
    tags = {
        "stage": "train",
        "model_family": f"cnn_{config.rnn_type}",
        "experiment_name": config.figure_prefix,
        "imbalance_strategy": config.imbalance_strategy,
        "cnn_train_policy": config.cnn_train_policy,
    }
    figure_paths = [
        config.figures_dir / f"{config.figure_prefix}_frame_confusion_matrix.png",
        config.figures_dir / f"{config.figure_prefix}_frame_roc_curve.png",
    ]

    with mlflow_run(
        enabled=config.mlflow_enabled,
        experiment_name=config.mlflow_experiment_name,
        run_name=run_name,
        tracking_uri=config.mlflow_tracking_uri,
        tags=tags,
    ) as mlflow:
        log_params(mlflow, {**asdict(config), **model_config_from_training_config(config)})
        log_metrics(
            mlflow,
            {
                "train_sequences": report["train_sequences"],
                "val_sequences": report["val_sequences"],
                "train_frames": report["train_frames"],
                "val_frames": report["val_frames"],
                "train_videos": report["train_videos"],
                "val_videos": report["val_videos"],
            },
        )
        log_metrics(
            mlflow,
            report["parameter_counts"],  # type: ignore[arg-type]
            prefix="model_",
        )
        if report["best_epoch"] is not None:
            log_metrics(
                mlflow,
                {
                    "best_epoch": report["best_epoch"],
                    "best_monitor_value": report["best_monitor_value"],
                    "early_stopped": int(bool(report["early_stopped"])),
                    "stopped_epoch": report["stopped_epoch"],
                },
            )
        if report["best_frame_metrics"] is not None:
            log_metrics(
                mlflow,
                report["best_frame_metrics"],  # type: ignore[arg-type]
                prefix="best_val_",
            )
        for epoch_metrics in report["history"]:  # type: ignore[index]
            step = int(epoch_metrics["epoch"])
            log_metrics(mlflow, {"train_loss": epoch_metrics["train_loss"]}, step=step)
            log_metrics(
                mlflow,
                {
                    key: value
                    for key, value in epoch_metrics.items()
                    if key not in {"epoch", "train_loss"}
                },
                prefix="val_frame_",
                step=step,
            )
        log_metrics(mlflow, report["frame_metrics"], prefix="final_frame_")  # type: ignore[arg-type]
        log_artifacts(
            mlflow,
            [config.metrics_path, config.predictions_path],
            artifact_path="reports",
        )
        log_artifacts(
            mlflow,
            [config.checkpoint_path, Path(str(report["best_checkpoint_path"]))],
            artifact_path="checkpoints",
        )
        log_artifacts(mlflow, figure_paths, artifact_path="figures")
```

## Passo 3 - Atualizar `train_sequence_model.py`

Substitua o conteudo de:

```text
scripts/train_sequence_model.py
```

por:

```python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

from nexar_collision.models.train_sequence import SequenceTrainingConfig, train_sequence_model
from nexar_collision.tracking.mlflow_utils import DEFAULT_EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN + GRU/LSTM temporal model.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--cnn-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--cnn-train-policy",
        choices=["frozen", "layer4", "full"],
        default="frozen",
        help="CNN fine-tuning policy. Use layer4 for partial fine-tuning.",
    )
    parser.add_argument(
        "--fine-tune-cnn",
        action="store_true",
        help="Backward-compatible alias for --cnn-train-policy full.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--train-sequence-stride", type=int, default=1)
    parser.add_argument("--val-sequence-stride", type=int, default=1)
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument(
        "--imbalance-strategy",
        choices=["none", "class_weight", "weighted_sampler"],
        default="class_weight",
    )
    parser.add_argument(
        "--loss-name",
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "temporal_frames_224_manifest.csv",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional CSV with columns id and split.",
    )
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--experiment-name", default="cnn_gru_sequence")
    parser.add_argument("--monitor-metric", default="roc_auc")
    parser.add_argument("--monitor-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name
    cnn_train_policy = "full" if args.fine_tune_cnn else args.cnn_train_policy

    report = train_sequence_model(
        SequenceTrainingConfig(
            manifest_path=args.manifest,
            split_manifest_path=args.split_manifest,
            split_column=args.split_column,
            train_split=args.train_split,
            val_split=args.val_split,
            predictions_path=PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{experiment_name}_sequence_predictions.csv",
            metrics_path=PROJECT_ROOT
            / "models"
            / "reports"
            / f"{experiment_name}_sequence_metrics.json",
            checkpoint_path=PROJECT_ROOT
            / "models"
            / "checkpoints"
            / f"{experiment_name}_sequence.pt",
            best_checkpoint_path=PROJECT_ROOT
            / "models"
            / "checkpoints"
            / f"{experiment_name}_best_sequence.pt",
            figure_prefix=experiment_name,
            sequence_length=args.sequence_length,
            train_sequence_stride=args.train_sequence_stride,
            val_sequence_stride=args.val_sequence_stride,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            head_learning_rate=args.head_learning_rate,
            cnn_learning_rate=args.cnn_learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            pretrained=args.pretrained,
            cnn_train_policy=cnn_train_policy,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            imbalance_strategy=args.imbalance_strategy,
            loss_name=args.loss_name,
            focal_gamma=args.focal_gamma,
            num_workers=args.num_workers,
            monitor_metric=args.monitor_metric,
            monitor_mode=args.monitor_mode,
            patience=args.patience,
            min_delta=args.min_delta,
            mlflow_enabled=not args.no_mlflow,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )
    )
    print("Sequence training complete")
    print(f"Device: {report['device']}")
    print(f"Train sequences: {report['train_sequences']}")
    print(f"Validation sequences: {report['val_sequences']}")
    print(f"Parameter counts: {report['parameter_counts']}")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Best {report['best_monitor_metric']}: {report['best_monitor_value']}")
    print(f"Best checkpoint: {report['best_checkpoint_path']}")
    print(f"Frame metrics: {report['frame_metrics']}")


if __name__ == "__main__":
    main()
```

## Passo 4 - Validar que a CLI carregou

Rode:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py --help
```

Confirme que aparecem as novas opcoes:

```text
--cnn-train-policy
--head-learning-rate
--cnn-learning-rate
--imbalance-strategy
--loss-name
--focal-gamma
```

## Passo 5 - Rodar o novo treino principal

Esse e o experimento recomendado:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\temporal_frames_224_manifest.csv `
  --split-manifest data\interim\sample_100_videos_splits.csv `
  --experiment-name temporal_alert_224_pretrained_gru_seq8_sampler `
  --pretrained `
  --sequence-length 8 `
  --train-sequence-stride 2 `
  --val-sequence-stride 1 `
  --rnn-type gru `
  --epochs 6 `
  --batch-size 8 `
  --learning-rate 0.0001 `
  --head-learning-rate 0.0001 `
  --cnn-learning-rate 0.00001 `
  --hidden-size 128 `
  --num-workers 0 `
  --cnn-train-policy layer4 `
  --imbalance-strategy weighted_sampler `
  --loss-name cross_entropy `
  --monitor-metric roc_auc `
  --monitor-mode max `
  --patience 2
```

Notas:

- `sequence-length 8` aumenta o contexto temporal.
- `cnn-train-policy layer4` libera apenas o ultimo bloco da ResNet18.
- `cnn-learning-rate 0.00001` evita destruir os pesos pre-treinados.
- `head-learning-rate 0.0001` deixa GRU/classifier aprenderem mais rapido.
- `weighted_sampler` balanceia as sequencias amostradas no treino.
- `batch-size 8` e mais conservador para caber na GPU ao treinar `layer4`.

Se faltar memoria na GPU, tente:

```powershell
--batch-size 4
```

## Passo 6 - Avaliar o checkpoint escolhido

Depois do treino, rode:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_sequence_model.py `
  --checkpoint models\checkpoints\temporal_alert_224_pretrained_gru_seq8_sampler_best_sequence.pt `
  --experiment-name temporal_alert_224_pretrained_gru_seq8_sampler `
  --manifest data\interim\temporal_frames_224_manifest.csv `
  --sample-csv data\interim\sample_100_videos_splits.csv `
  --split-manifest data\interim\sample_100_videos_splits.csv `
  --split val `
  --threshold 0.5 `
  --batch-size 8
```

## Passo 7 - Rodar sweep de thresholds

Sweep raw:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py `
  --experiment-name temporal_alert_224_pretrained_gru_seq8_sampler `
  --sample-csv data\interim\sample_100_videos_splits.csv `
  --split val
```

Sweep com 2 frames consecutivos:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py `
  --experiment-name temporal_alert_224_pretrained_gru_seq8_sampler `
  --sample-csv data\interim\sample_100_videos_splits.csv `
  --split val `
  --min-consecutive-frames 2
```

## Passo 8 - Comparar contra o baseline

Arquivos esperados:

```text
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_sequence_metrics.json
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_alert_metrics.json
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_alert_threshold_sweep.csv
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_consecutive2_alert_threshold_sweep.csv
outputs/predictions/temporal_alert_224_pretrained_gru_seq8_sampler_sequence_predictions.csv
outputs/predictions/temporal_alert_224_pretrained_gru_seq8_sampler_temporal_risk_scores.csv
```

Use este criterio:

```text
Baseline:
precision = 0.727
recall = 0.800
false_alarm_rate = 0.300
mean_alert_time_error = -7.991 s
```

O novo experimento e melhor se atingir pelo menos uma destas condicoes:

1. `recall >= 0.80` e `false_alarm_rate < 0.30`.
2. `recall >= 0.80` e `precision > 0.727`.
3. `recall >= 0.70` e `false_alarm_rate <= 0.20`.

Se nao bater o baseline, ele ainda pode ser util se melhorar a calibracao do
score ou reduzir muito o erro medio de tempo do alerta.

## Passo 9 - Registrar o resultado em relatorio

Crie depois um relatorio novo:

```text
reports/gru_seq8_sampler_experiment.md
```

Estrutura sugerida:

```markdown
# GRU seq8 sampler experiment

Run date: YYYY-MM-DD

## Objective

Test whether longer causal context, partial CNN fine-tuning, and sequence-level
weighted sampling improve temporal collision alerts.

## Configuration

| Setting | Value |
| --- | --- |
| Experiment | temporal_alert_224_pretrained_gru_seq8_sampler |
| Sequence length | 8 |
| RNN | GRU |
| CNN | ResNet18 ImageNet pretrained |
| CNN train policy | layer4 |
| Imbalance strategy | weighted_sampler |
| Loss | cross_entropy |
| Head LR | 0.0001 |
| CNN LR | 0.00001 |
| Batch size | 8 |

## Frame-Level Results

Paste best epoch, best ROC-AUC, F1, precision, recall.

## Alert Results

Compare raw and 2-consecutive-frame sweep.

## Comparison With Current Baseline

Baseline:

- precision: 0.727
- recall: 0.800
- false alarm rate: 0.300
- mean alert error: -7.991 s

## Conclusion

State whether this should replace the current baseline or become a failed but
informative experiment.
```

## Opcional - Segundo experimento com focal loss

Se o sampler melhorar recall mas criar falsos positivos demais, rode uma segunda
variante:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\temporal_frames_224_manifest.csv `
  --split-manifest data\interim\sample_100_videos_splits.csv `
  --experiment-name temporal_alert_224_pretrained_gru_seq8_focal `
  --pretrained `
  --sequence-length 8 `
  --train-sequence-stride 2 `
  --val-sequence-stride 1 `
  --rnn-type gru `
  --epochs 6 `
  --batch-size 8 `
  --learning-rate 0.0001 `
  --head-learning-rate 0.0001 `
  --cnn-learning-rate 0.00001 `
  --hidden-size 128 `
  --num-workers 0 `
  --cnn-train-policy layer4 `
  --imbalance-strategy class_weight `
  --loss-name focal `
  --focal-gamma 2.0 `
  --monitor-metric roc_auc `
  --monitor-mode max `
  --patience 2
```

Depois avalie e rode os sweeps usando o mesmo padrao do experimento principal.

## Observacao importante

A validacao atual usa apenas 20 videos, e os thresholds tambem sao escolhidos
nessa mesma validacao. Mesmo que o resultado melhore, trate a conclusao como
experimental. Para uma afirmacao mais forte no artigo, o proximo passo depois
desse ciclo deve ser um holdout separado ou cross-validation por video.
