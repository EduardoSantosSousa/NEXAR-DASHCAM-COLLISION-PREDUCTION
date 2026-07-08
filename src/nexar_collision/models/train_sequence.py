"""Training utilities for CNN + GRU/LSTM temporal frame classifiers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from nexar_collision.data.dataset import FrameSequenceCollisionDataset
from nexar_collision.evaluation.metrics import compute_metrics
from nexar_collision.models.temporal_model import build_temporal_model
from nexar_collision.models.train import (
    build_class_weighted_loss,
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
    val_size: float = 0.2
    random_state: int = 42
    num_workers: int = 0
    pretrained: bool = True
    rnn_type: str = "gru"
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    freeze_cnn: bool = True
    device: str = "auto"
    monitor_metric: str = "roc_auc"
    monitor_mode: str = "max"
    patience: int = 2
    min_delta: float = 0.0
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = DEFAULT_EXPERIMENT_NAME
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None


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
        "freeze_cnn": config.freeze_cnn,
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_temporal_model(**model_config_from_training_config(config)).to(device)
    criterion = build_class_weighted_loss(train_df, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )

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
        "train_target_distribution": {
            str(key): int(value)
            for key, value in train_df["target"].value_counts().sort_index().items()
        },
        "val_target_distribution": {
            str(key): int(value)
            for key, value in val_df["target"].value_counts().sort_index().items()
        },
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
