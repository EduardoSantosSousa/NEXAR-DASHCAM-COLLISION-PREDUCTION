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

from nexar_collision.data.dataset import (
    ExplicitFrameWindowSequenceDataset,
    FrameSequenceCollisionDataset,
)
from nexar_collision.evaluation.evaluate import compute_alert_metrics
from nexar_collision.evaluation.metrics import compute_classification_metrics
from nexar_collision.models.temporal_model import build_temporal_model
from nexar_collision.models.train import (
    autocast_context,
    build_transforms,
    default_best_checkpoint_path,
    is_improvement,
    load_manifest_with_optional_split,
    monitor_value_from_metrics,
    resolve_device,
    save_figures,
    seed_everything,
    should_use_amp,
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
    num_classes: int = 2
    target_column: str = "target"
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
    sample_weight_column: str | None = None
    hard_negative_weight: float = 1.0
    negative_video_weight: float = 1.0
    positive_safe_weight: float = 1.0
    positive_event_weight: float = 1.0
    alert_metric_selection: bool = False
    alert_min_recall: float = 0.80
    alert_max_false_alarm_rate: float = 0.30
    alert_min_precision: float = 0.70
    alert_threshold_start: float = 0.10
    alert_threshold_stop: float = 0.99
    alert_threshold_step: float = 0.01
    alert_min_consecutive_frames: int = 1
    alert_class_indices: tuple[int, ...] | None = None
    amp: bool = False
    log_every_n_batches: int = 0
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
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
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
        loss = focal_factor * ce_loss
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return torch.mean(loss)
        raise ValueError(f"Unsupported focal loss reduction: {self.reduction}")


def validate_config(config: SequenceTrainingConfig) -> None:
    if config.imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            "imbalance_strategy must be one of: "
            f"{sorted(IMBALANCE_STRATEGIES)}"
        )
    if config.loss_name not in LOSS_NAMES:
        raise ValueError(f"loss_name must be one of: {sorted(LOSS_NAMES)}")
    if config.num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    for class_index in resolved_alert_class_indices(config):
        if class_index < 0 or class_index >= config.num_classes:
            raise ValueError(
                "alert_class_indices must be valid class indices for "
                f"num_classes={config.num_classes}. Got: "
                f"{resolved_alert_class_indices(config)}"
            )


def model_config_from_training_config(config: SequenceTrainingConfig) -> dict[str, object]:
    return {
        "num_classes": config.num_classes,
        "cnn_backbone": "resnet18",
        "pretrained": config.pretrained,
        "rnn_type": config.rnn_type,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "bidirectional": config.bidirectional,
        "cnn_train_policy": config.cnn_train_policy,
    }


def resolved_alert_class_indices(config: SequenceTrainingConfig) -> tuple[int, ...]:
    if config.alert_class_indices is not None:
        return tuple(int(index) for index in config.alert_class_indices)
    if config.num_classes == 2:
        return (1,)
    return tuple(range(1, config.num_classes))


def prepare_sequence_target_column(
    manifest_df: pd.DataFrame,
    config: SequenceTrainingConfig,
) -> pd.DataFrame:
    if config.target_column not in manifest_df.columns:
        raise ValueError(
            f"Target column {config.target_column!r} was not found in the manifest."
        )

    prepared_df = manifest_df.copy()
    if config.target_column != "target" and "target" in prepared_df.columns:
        prepared_df["binary_target"] = prepared_df["target"]
    prepared_df["target"] = prepared_df[config.target_column].astype(int)

    min_target = int(prepared_df["target"].min())
    max_target = int(prepared_df["target"].max())
    if min_target < 0 or max_target >= config.num_classes:
        raise ValueError(
            "Target values must be in the range "
            f"[0, {config.num_classes - 1}]. "
            f"Observed min={min_target}, max={max_target}."
        )

    return prepared_df


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
            "target_column": config.target_column,
            "num_classes": config.num_classes,
            "alert_class_indices": list(resolved_alert_class_indices(config)),
            "epoch": epoch,
            "monitor_metric": config.monitor_metric,
            "monitor_value": monitor_value,
        },
        path,
    )


def build_balanced_class_weights(
    train_df: pd.DataFrame,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor | None:
    class_counts = train_df["target"].value_counts().sort_index()
    if len(class_counts) < 2:
        return None

    total = float(class_counts.sum())
    return torch.tensor(
        [
            total / (float(num_classes) * float(class_counts.get(index, 1)))
            for index in range(num_classes)
        ],
        dtype=torch.float32,
        device=device,
    )


def build_sequence_loss(
    train_df: pd.DataFrame,
    config: SequenceTrainingConfig,
    device: torch.device,
    reduction: str = "mean",
) -> nn.Module:
    class_weights = None
    if config.imbalance_strategy == "class_weight":
        class_weights = build_balanced_class_weights(
            train_df,
            device,
            config.num_classes,
        )

    if config.loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    if config.loss_name == "focal":
        return FocalLoss(
            gamma=config.focal_gamma,
            weight=class_weights,
            reduction=reduction,
        )

    raise ValueError(f"Unsupported loss_name: {config.loss_name}")


def window_type_weight_mapping(config: SequenceTrainingConfig) -> dict[str, float]:
    return {
        "hard_negative_mined": config.hard_negative_weight,
        "negative_video": config.negative_video_weight,
        "positive_safe": config.positive_safe_weight,
        "positive_event": config.positive_event_weight,
    }


def sample_weights_enabled(config: SequenceTrainingConfig) -> bool:
    configured_weights = window_type_weight_mapping(config)
    return (
        config.sample_weight_column is not None
        or any(abs(weight - 1.0) > 1e-12 for weight in configured_weights.values())
    )


def apply_sequence_sample_weights(
    train_df: pd.DataFrame,
    config: SequenceTrainingConfig,
) -> pd.DataFrame:
    if not sample_weights_enabled(config):
        return train_df

    weighted_df = train_df.copy()
    sample_weights = pd.Series(1.0, index=weighted_df.index, dtype=float)

    if "window_type" in weighted_df.columns:
        mapping = window_type_weight_mapping(config)
        sample_weights *= (
            weighted_df["window_type"]
            .map(mapping)
            .fillna(1.0)
            .astype(float)
        )

    if config.sample_weight_column is not None:
        if config.sample_weight_column not in weighted_df.columns:
            raise ValueError(
                f"Sample weight column {config.sample_weight_column!r} "
                "was not found in the training manifest."
            )
        sample_weights *= weighted_df[config.sample_weight_column].astype(float)

    weighted_df["sample_weight"] = sample_weights
    return weighted_df


def reduce_weighted_loss(
    loss_values: torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    use_sample_weights: bool,
) -> torch.Tensor:
    if not use_sample_weights:
        return loss_values

    sample_weights = batch["sample_weight"].to(device=device, dtype=torch.float32)
    return (loss_values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def is_explicit_window_manifest(manifest_df: pd.DataFrame) -> bool:
    return "frame_paths" in manifest_df.columns


def build_sequence_dataset(
    manifest_df: pd.DataFrame,
    sequence_length: int,
    transform,
    sequence_stride: int = 1,
):
    if is_explicit_window_manifest(manifest_df):
        return ExplicitFrameWindowSequenceDataset(manifest_df, transform=transform)

    return FrameSequenceCollisionDataset(
        manifest_df,
        sequence_length=sequence_length,
        transform=transform,
        sequence_stride=sequence_stride,
    )


def sequence_targets(dataset) -> list[int]:
    if hasattr(dataset, "sequence_targets"):
        return [int(target) for target in dataset.sequence_targets]

    targets = []
    for sequence_positions in dataset.sequence_index:
        end_position = sequence_positions[-1]
        target = int(dataset.manifest.iloc[end_position]["target"])
        targets.append(target)
    return targets


def build_weighted_sequence_sampler(
    dataset,
    num_classes: int,
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    targets = sequence_targets(dataset)
    class_counts = np.bincount(targets, minlength=num_classes)

    if np.any(class_counts == 0):
        raise ValueError(
            "WeightedRandomSampler requires all classes in the training split. "
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
    amp_enabled: bool = False,
    scaler=None,
    epoch: int | None = None,
    log_every_n_batches: int = 0,
    use_sample_weights: bool = False,
) -> float:
    model.train()
    losses = []

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss_values = criterion(logits, targets)
            loss = reduce_weighted_loss(
                loss_values,
                batch=batch,
                device=device,
                use_sample_weights=use_sample_weights,
            )

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if log_every_n_batches > 0 and batch_idx % log_every_n_batches == 0:
            epoch_prefix = f"epoch={epoch} " if epoch is not None else ""
            print(
                f"{epoch_prefix}batch={batch_idx}/{len(loader)} "
                f"train_loss_running={float(np.mean(losses)):.4f}",
                flush=True,
            )

    return float(np.mean(losses))


@torch.no_grad()
def predict_sequences(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    alert_class_indices: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    model.eval()
    records = []

    for batch in loader:
        images = batch["images"].to(device)
        with autocast_context(device, amp_enabled):
            logits = model(images)
        probabilities = torch.softmax(logits, dim=1).cpu()
        scores = probabilities[:, list(alert_class_indices)].sum(dim=1).numpy()
        preds = probabilities.argmax(dim=1).numpy()

        for idx, score in enumerate(scores):
            record = {
                "id": batch["video_id"][idx],
                "timestamp": float(batch["timestamp"][idx]),
                "target": int(batch["target"][idx]),
                "video_target": int(batch["video_target"][idx]),
                "duration": float(batch["duration"][idx]),
                "score": float(score),
                "prediction": int(preds[idx]),
            }
            for class_index, probability in enumerate(probabilities[idx].tolist()):
                record[f"prob_class_{class_index}"] = float(probability)
            records.append(record)

    return pd.DataFrame(records)


def build_alert_sample_df(val_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    target_column = "video_target" if "video_target" in val_df.columns else "target"

    for video_id, video_df in val_df.groupby("id", sort=False):
        first_row = video_df.iloc[0]
        target = int(first_row[target_column])
        time_of_alert = (
            float(first_row["time_of_alert"])
            if "time_of_alert" in first_row and pd.notna(first_row["time_of_alert"])
            else None
        )
        time_of_event = (
            float(first_row["time_of_event"])
            if "time_of_event" in first_row and pd.notna(first_row["time_of_event"])
            else None
        )
        records.append(
            {
                "id": str(video_id),
                "target": target,
                "time_of_alert": time_of_alert,
                "time_of_event": time_of_event,
            }
        )

    return pd.DataFrame(records)


def first_alert_time_from_scores(
    video_scores: pd.DataFrame,
    threshold: float,
    min_consecutive_frames: int,
) -> float | None:
    sorted_scores = video_scores.sort_values("timestamp")
    above_threshold = sorted_scores["risk_score"] >= threshold

    if min_consecutive_frames <= 1:
        alerts = sorted_scores[above_threshold]
        if alerts.empty:
            return None
        return float(alerts.iloc[0]["timestamp"])

    consecutive_count = 0
    for timestamp, is_above_threshold in zip(
        sorted_scores["timestamp"],
        above_threshold,
    ):
        consecutive_count = consecutive_count + 1 if is_above_threshold else 0
        if consecutive_count >= min_consecutive_frames:
            return float(timestamp)

    return None


def build_alert_predictions_from_scores(
    risk_scores_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    threshold: float,
    min_consecutive_frames: int,
) -> pd.DataFrame:
    records = []
    for _, row in sample_df.iterrows():
        video_id = row["id"]
        target = int(row["target"])
        video_scores = risk_scores_df[risk_scores_df["id"] == video_id]

        if video_scores.empty:
            predicted_alert_time = None
            max_risk_score = None
            mean_risk_score = None
        else:
            predicted_alert_time = first_alert_time_from_scores(
                video_scores,
                threshold,
                min_consecutive_frames,
            )
            max_risk_score = float(video_scores["risk_score"].max())
            mean_risk_score = float(video_scores["risk_score"].mean())

        time_of_alert = (
            float(row["time_of_alert"]) if pd.notna(row["time_of_alert"]) else None
        )
        time_of_event = (
            float(row["time_of_event"]) if pd.notna(row["time_of_event"]) else None
        )
        predicted_lead_time = (
            time_of_event - predicted_alert_time
            if target == 1 and predicted_alert_time is not None and time_of_event is not None
            else None
        )
        alert_time_error = (
            predicted_alert_time - time_of_alert
            if target == 1 and predicted_alert_time is not None and time_of_alert is not None
            else None
        )

        records.append(
            {
                "id": video_id,
                "target": target,
                "time_of_alert": time_of_alert,
                "time_of_event": time_of_event,
                "predicted_alert_time": predicted_alert_time,
                "predicted_lead_time": predicted_lead_time,
                "alert_time_error": alert_time_error,
                "max_risk_score": max_risk_score,
                "mean_risk_score": mean_risk_score,
            }
        )

    return pd.DataFrame(records)


def alert_selection_score(
    metrics: dict[str, float | int | None],
    config: SequenceTrainingConfig,
) -> float:
    precision = float(metrics["alert_precision"] or 0.0)
    recall = float(metrics["alert_recall"] or 0.0)
    false_alarm_rate = float(metrics["false_alarm_rate"] or 1.0)

    recall_shortfall = max(0.0, config.alert_min_recall - recall)
    false_alarm_excess = max(0.0, false_alarm_rate - config.alert_max_false_alarm_rate)
    precision_shortfall = max(0.0, config.alert_min_precision - precision)
    gate_passed = (
        recall >= config.alert_min_recall
        and false_alarm_rate <= config.alert_max_false_alarm_rate
        and precision >= config.alert_min_precision
    )

    return (
        (10.0 if gate_passed else 0.0)
        + (3.0 * recall)
        + precision
        - (2.0 * false_alarm_rate)
        - (5.0 * recall_shortfall)
        - (3.0 * false_alarm_excess)
        - (2.0 * precision_shortfall)
    )


def select_alert_operating_point(
    predictions_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    config: SequenceTrainingConfig,
) -> dict[str, float | int | None]:
    risk_scores_df = predictions_df.rename(columns={"score": "risk_score"})[
        ["id", "timestamp", "risk_score"]
    ].copy()

    thresholds = np.round(
        np.arange(
            config.alert_threshold_start,
            config.alert_threshold_stop + config.alert_threshold_step / 2,
            config.alert_threshold_step,
        ),
        3,
    )
    records = []
    for threshold in thresholds:
        alert_df = build_alert_predictions_from_scores(
            risk_scores_df=risk_scores_df,
            sample_df=sample_df,
            threshold=float(threshold),
            min_consecutive_frames=config.alert_min_consecutive_frames,
        )
        metrics = compute_alert_metrics(alert_df)
        records.append(
            {
                "threshold": float(threshold),
                **metrics,
                "alert_selection_score": alert_selection_score(metrics, config),
            }
        )

    sweep_df = pd.DataFrame(records)
    sweep_df["gate_passed"] = (
        (sweep_df["alert_recall"] >= config.alert_min_recall)
        & (sweep_df["false_alarm_rate"] <= config.alert_max_false_alarm_rate)
        & (sweep_df["alert_precision"] >= config.alert_min_precision)
    )
    selected = sweep_df.sort_values(
        [
            "gate_passed",
            "alert_selection_score",
            "false_alarm_rate",
            "alert_recall",
            "alert_precision",
        ],
        ascending=[False, False, True, False, False],
    ).iloc[0]

    selected_dict = selected.to_dict()
    selected_metrics: dict[str, float | int | bool | None] = {}
    for key, value in selected_dict.items():
        output_key = key
        if key == "threshold":
            output_key = "alert_threshold"
        elif key == "gate_passed":
            output_key = "alert_gate_passed"

        if key == "gate_passed":
            selected_metrics[output_key] = bool(value)
        else:
            selected_metrics[output_key] = float(value) if pd.notna(value) else None

    return selected_metrics


def train_sequence_model(config: SequenceTrainingConfig | None = None) -> dict[str, object]:
    config = config or SequenceTrainingConfig()
    validate_config(config)
    seed_everything(config.random_state)
    device = resolve_device(config.device)
    best_checkpoint_path = config.best_checkpoint_path or default_best_checkpoint_path(
        config.checkpoint_path
    )

    manifest_df = load_manifest_with_optional_split(config)  # type: ignore[arg-type]
    manifest_df = prepare_sequence_target_column(manifest_df, config)
    train_df, val_df = split_by_video(
        manifest_df=manifest_df,
        val_size=config.val_size,
        random_state=config.random_state,
        split_column=config.split_column,
        train_split=config.train_split,
        val_split=config.val_split,
    )
    train_df = apply_sequence_sample_weights(train_df, config)
    train_transform, eval_transform = build_transforms()
    alert_sample_df = build_alert_sample_df(val_df) if config.alert_metric_selection else None
    use_sample_weights = sample_weights_enabled(config)
    alert_class_indices = resolved_alert_class_indices(config)

    train_dataset = build_sequence_dataset(
        train_df,
        sequence_length=config.sequence_length,
        transform=train_transform,
        sequence_stride=config.train_sequence_stride,
    )
    val_dataset = build_sequence_dataset(
        val_df,
        sequence_length=config.sequence_length,
        transform=eval_transform,
        sequence_stride=config.val_sequence_stride,
    )
    manifest_type = "explicit_windows" if is_explicit_window_manifest(manifest_df) else "causal_frames"

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
            train_dataset,
            config.num_classes,
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_temporal_model(**model_config_from_training_config(config)).to(device)
    criterion = build_sequence_loss(
        train_df,
        config,
        device,
        reduction="none" if use_sample_weights else "mean",
    )
    optimizer = build_optimizer(model, config)
    parameter_counts = count_trainable_parameters(model)
    amp_enabled = should_use_amp(device, config.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_epoch: int | None = None
    best_monitor_value: float | None = None
    best_frame_metrics: dict[str, float | None] | None = None
    best_alert_metrics: dict[str, float | int | bool | None] | None = None
    epochs_without_improvement = 0
    early_stopped = False
    stopped_epoch: int | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            epoch=epoch,
            log_every_n_batches=config.log_every_n_batches,
            use_sample_weights=use_sample_weights,
        )
        predictions_df = predict_sequences(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            alert_class_indices=alert_class_indices,
        )
        frame_metrics = compute_classification_metrics(
            predictions_df["target"],
            predictions_df["prediction"],
            predictions_df["score"],
            num_classes=config.num_classes,
        )
        alert_metrics = (
            select_alert_operating_point(predictions_df, alert_sample_df, config)
            if alert_sample_df is not None
            else {}
        )
        monitor_metrics = {**frame_metrics, **alert_metrics}
        monitor_value = monitor_value_from_metrics(
            config.monitor_metric,
            train_loss,
            monitor_metrics,
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
            best_alert_metrics = dict(alert_metrics) if alert_metrics else None
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
                **alert_metrics,
                "monitor_metric": config.monitor_metric,
                "monitor_value": monitor_value,
                "is_best": improved,
            }
        )
        best_marker = " best" if improved else ""
        alert_log = ""
        if alert_metrics:
            alert_log = (
                f" val_alert_score={alert_metrics['alert_selection_score']:.4f}"
                f" val_alert_threshold={alert_metrics['alert_threshold']:.2f}"
                f" val_alert_recall={alert_metrics['alert_recall']:.4f}"
                f" val_false_alarm_rate={alert_metrics['false_alarm_rate']:.4f}"
            )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_f1={frame_metrics['f1']:.4f} "
            f"val_{config.monitor_metric}={monitor_value:.4f}"
            f"{alert_log}{best_marker}",
            flush=True,
        )

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            early_stopped = True
            stopped_epoch = epoch
            print(
                "early_stopping="
                f"epoch={epoch} monitor={config.monitor_metric} "
                f"best_epoch={best_epoch} best_value={best_monitor_value:.4f}",
                flush=True,
            )
            break

    predictions_df = predict_sequences(
        model,
        val_loader,
        device,
        amp_enabled=amp_enabled,
        alert_class_indices=alert_class_indices,
    )
    frame_metrics = compute_classification_metrics(
        predictions_df["target"],
        predictions_df["prediction"],
        predictions_df["score"],
        num_classes=config.num_classes,
    )
    final_alert_metrics = (
        select_alert_operating_point(predictions_df, alert_sample_df, config)
        if alert_sample_df is not None
        else None
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
        "manifest_type": manifest_type,
        "amp_enabled": amp_enabled,
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
        "sample_weights_enabled": use_sample_weights,
        "window_type_weights": window_type_weight_mapping(config),
        "target_column": config.target_column,
        "num_classes": config.num_classes,
        "alert_class_indices": list(alert_class_indices),
        "history": history,
        "best_epoch": best_epoch,
        "best_monitor_metric": config.monitor_metric,
        "best_monitor_mode": config.monitor_mode,
        "best_monitor_value": best_monitor_value,
        "best_frame_metrics": best_frame_metrics,
        "best_alert_metrics": best_alert_metrics,
        "best_checkpoint_path": str(best_checkpoint_path),
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "frame_metrics": frame_metrics,
        "alert_metrics": final_alert_metrics,
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
        if report["best_alert_metrics"] is not None:
            log_metrics(
                mlflow,
                report["best_alert_metrics"],  # type: ignore[arg-type]
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
        if report["alert_metrics"] is not None:
            log_metrics(
                mlflow,
                report["alert_metrics"],  # type: ignore[arg-type]
                prefix="final_val_",
            )
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
