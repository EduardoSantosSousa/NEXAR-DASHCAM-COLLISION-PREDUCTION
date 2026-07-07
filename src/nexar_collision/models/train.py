"""Training loop utilities for the baseline frame classifier."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from nexar_collision.data.dataset import FrameCollisionDataset
from nexar_collision.evaluation.metrics import compute_metrics
from nexar_collision.models.baseline_cnn import build_baseline_cnn
from nexar_collision.tracking.mlflow_utils import (
    DEFAULT_EXPERIMENT_NAME,
    log_artifacts,
    log_metrics,
    log_params,
    mlflow_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainingConfig:
    manifest_path: Path = PROJECT_ROOT / "data" / "interim" / "sample_frames_manifest.csv"
    predictions_path: Path = PROJECT_ROOT / "outputs" / "predictions" / "baseline_frame_predictions.csv"
    metrics_path: Path = PROJECT_ROOT / "models" / "reports" / "baseline_metrics.json"
    checkpoint_path: Path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_resnet18.pt"
    best_checkpoint_path: Path | None = None
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    split_manifest_path: Path | None = None
    split_column: str = "split"
    train_split: str = "train"
    val_split: str = "val"
    figure_prefix: str = "baseline"
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    val_size: float = 0.2
    random_state: int = 42
    num_workers: int = 0
    pretrained: bool = False
    device: str = "auto"
    monitor_metric: str = "roc_auc"
    monitor_mode: str = "max"
    patience: int = 3
    min_delta: float = 0.0
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = DEFAULT_EXPERIMENT_NAME
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_best_checkpoint_path(checkpoint_path: Path) -> Path:
    """Return a sibling checkpoint path for the best monitored epoch."""
    if checkpoint_path.name.endswith("_resnet18.pt"):
        name = checkpoint_path.name.replace("_resnet18.pt", "_best_resnet18.pt")
    else:
        name = f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
    return checkpoint_path.with_name(name)


def monitor_value_from_metrics(
    monitor_metric: str,
    train_loss: float,
    frame_metrics: dict[str, float | None],
) -> float:
    if monitor_metric == "train_loss":
        return float(train_loss)
    if monitor_metric not in frame_metrics:
        available_metrics = sorted([*frame_metrics.keys(), "train_loss"])
        raise ValueError(
            f"Unknown monitor metric: {monitor_metric}. "
            f"Available metrics: {available_metrics}"
        )
    value = frame_metrics[monitor_metric]
    if value is None:
        raise ValueError(
            f"Monitor metric {monitor_metric!r} is None for this validation split."
        )
    return float(value)


def is_improvement(
    current_value: float,
    best_value: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if mode not in {"max", "min"}:
        raise ValueError("monitor_mode must be either 'max' or 'min'.")
    if best_value is None:
        return True
    if mode == "max":
        return current_value > best_value + min_delta
    return current_value < best_value - min_delta


def load_manifest_with_optional_split(config: TrainingConfig) -> pd.DataFrame:
    manifest_df = pd.read_csv(config.manifest_path, dtype={"id": str})

    if config.split_manifest_path is None:
        return manifest_df

    split_df = pd.read_csv(config.split_manifest_path, dtype={"id": str})
    required_columns = {"id", config.split_column}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in split manifest: {sorted(missing_columns)}"
        )

    split_df = split_df[["id", config.split_column]].drop_duplicates("id")
    merged_df = manifest_df.merge(split_df, on="id", how="left", validate="many_to_one")

    missing_split = merged_df[config.split_column].isna()
    if missing_split.any():
        missing_ids = merged_df.loc[missing_split, "id"].drop_duplicates().head(10).tolist()
        raise ValueError(
            "Some manifest video ids were not found in the split manifest. "
            f"Examples: {missing_ids}"
        )

    return merged_df


def split_by_video(
    manifest_df: pd.DataFrame,
    val_size: float,
    random_state: int,
    split_column: str = "split",
    train_split: str = "train",
    val_split: str = "val",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_column in manifest_df.columns:
        train_df = manifest_df[manifest_df[split_column] == train_split].copy()
        val_df = manifest_df[manifest_df[split_column] == val_split].copy()

        if train_df.empty:
            raise ValueError(f"No rows found for train split: {train_split}")
        if val_df.empty:
            raise ValueError(f"No rows found for validation split: {val_split}")

        overlap = set(train_df["id"]).intersection(set(val_df["id"]))
        if overlap:
            raise ValueError(f"Found video leakage between train and val: {sorted(overlap)[:10]}")

        return train_df, val_df

    stratify_col = "video_target" if "video_target" in manifest_df.columns else "target"
    video_df = manifest_df[["id", stratify_col]].drop_duplicates("id")
    val_video_ids = (
        video_df.groupby(stratify_col, group_keys=False)
        .sample(frac=val_size, random_state=random_state)["id"]
        .tolist()
    )

    val_df = manifest_df[manifest_df["id"].isin(val_video_ids)].copy()
    train_df = manifest_df[~manifest_df["id"].isin(val_video_ids)].copy()
    return train_df, val_df


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


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
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    records = []

    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (scores >= 0.5).astype(int)

        for idx, score in enumerate(scores):
            records.append(
                {
                    "id": batch["video_id"][idx],
                    "frame_label": batch["frame_label"][idx],
                    "timestamp": float(batch["timestamp"][idx]),
                    "target": int(batch["target"][idx]),
                    "score": float(score),
                    "prediction": int(preds[idx]),
                }
            )

    return pd.DataFrame(records)


def aggregate_video_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    video_df = (
        predictions_df.groupby("id")
        .agg(
            target=("target", "first"),
            score_mean=("score", "mean"),
            score_max=("score", "max"),
        )
        .reset_index()
    )
    video_df["prediction_mean"] = (video_df["score_mean"] >= 0.5).astype(int)
    video_df["prediction_max"] = (video_df["score_max"] >= 0.5).astype(int)
    return video_df


def save_figures(
    predictions_df: pd.DataFrame,
    figures_dir: Path,
    figure_prefix: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    y_true = predictions_df["target"]
    y_pred = predictions_df["prediction"]
    y_score = predictions_df["score"]

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"{figure_prefix} Frame Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{figure_prefix}_frame_confusion_matrix.png", dpi=150)
    plt.close()

    if len(np.unique(y_true)) == 2:
        RocCurveDisplay.from_predictions(y_true, y_score)
        plt.title(f"{figure_prefix} Frame ROC Curve")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{figure_prefix}_frame_roc_curve.png", dpi=150)
        plt.close()


def build_class_weighted_loss(
    train_df: pd.DataFrame,
    device: torch.device,
) -> nn.Module:
    class_counts = train_df["target"].value_counts().sort_index()
    if len(class_counts) != 2:
        return nn.CrossEntropyLoss()

    total = float(class_counts.sum())
    class_weights = torch.tensor(
        [total / (2.0 * class_counts.get(index, 1)) for index in [0, 1]],
        dtype=torch.float32,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=class_weights)


def train_model(config: TrainingConfig | None = None) -> dict[str, object]:
    config = config or TrainingConfig()
    seed_everything(config.random_state)
    device = resolve_device(config.device)
    best_checkpoint_path = config.best_checkpoint_path or default_best_checkpoint_path(
        config.checkpoint_path
    )

    manifest_df = load_manifest_with_optional_split(config)
    train_df, val_df = split_by_video(
        manifest_df=manifest_df,
        val_size=config.val_size,
        random_state=config.random_state,
        split_column=config.split_column,
        train_split=config.train_split,
        val_split=config.val_split,
    )
    train_transform, eval_transform = build_transforms()

    train_loader = DataLoader(
        FrameCollisionDataset(train_df, transform=train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        FrameCollisionDataset(val_df, transform=eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_baseline_cnn(pretrained=config.pretrained).to(device)
    criterion = build_class_weighted_loss(train_df, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

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
        predictions_df = predict(model, val_loader, device)
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
            torch.save(model.state_dict(), best_checkpoint_path)
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

    predictions_df = predict(model, val_loader, device)
    video_predictions_df = aggregate_video_predictions(predictions_df)

    frame_metrics = compute_metrics(
        predictions_df["target"],
        predictions_df["prediction"],
        predictions_df["score"],
    )
    video_metrics_mean = compute_metrics(
        video_predictions_df["target"],
        video_predictions_df["prediction_mean"],
        video_predictions_df["score_mean"],
    )
    video_metrics_max = compute_metrics(
        video_predictions_df["target"],
        video_predictions_df["prediction_max"],
        video_predictions_df["score_max"],
    )

    predictions_df.to_csv(config.predictions_path, index=False)
    video_predictions_path = config.predictions_path.with_name(
        config.predictions_path.name.replace("_frame_predictions.csv", "_video_predictions.csv")
    )
    video_predictions_df.to_csv(video_predictions_path, index=False)
    save_figures(predictions_df, config.figures_dir, config.figure_prefix)
    torch.save(model.state_dict(), config.checkpoint_path)

    report = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "train_frames": len(train_df),
        "val_frames": len(val_df),
        "train_videos": train_df["id"].nunique(),
        "val_videos": val_df["id"].nunique(),
        "train_video_ids": sorted(train_df["id"].drop_duplicates().tolist()),
        "val_video_ids": sorted(val_df["id"].drop_duplicates().tolist()),
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
        "video_metrics_mean": video_metrics_mean,
        "video_metrics_max": video_metrics_max,
    }

    config.metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_training_run_to_mlflow(
        config=config,
        report=report,
        video_predictions_path=video_predictions_path,
    )
    return report


def log_training_run_to_mlflow(
    config: TrainingConfig,
    report: dict[str, object],
    video_predictions_path: Path,
) -> None:
    run_name = config.mlflow_run_name or f"{config.figure_prefix}_train"
    tags = {
        "stage": "train",
        "model_family": "resnet18",
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
        log_params(
            mlflow,
            {
                "stage": "train",
                "manifest_path": config.manifest_path,
                "split_manifest_path": config.split_manifest_path,
                "split_column": config.split_column,
                "train_split": config.train_split,
                "val_split": config.val_split,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "val_size": config.val_size,
                "random_state": config.random_state,
                "num_workers": config.num_workers,
                "pretrained": config.pretrained,
                "device": config.device,
                "monitor_metric": config.monitor_metric,
                "monitor_mode": config.monitor_mode,
                "patience": config.patience,
                "min_delta": config.min_delta,
                "best_checkpoint_path": report["best_checkpoint_path"],
            },
        )
        log_metrics(
            mlflow,
            {
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
            log_metrics(
                mlflow,
                {"train_loss": epoch_metrics["train_loss"]},
                step=step,
            )
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
        log_metrics(
            mlflow,
            report["video_metrics_mean"],  # type: ignore[arg-type]
            prefix="final_video_mean_",
        )
        log_metrics(
            mlflow,
            report["video_metrics_max"],  # type: ignore[arg-type]
            prefix="final_video_max_",
        )
        log_artifacts(
            mlflow,
            [
                config.metrics_path,
                config.predictions_path,
                video_predictions_path,
            ],
            artifact_path="reports",
        )
        log_artifacts(
            mlflow,
            [config.checkpoint_path, Path(str(report["best_checkpoint_path"]))],
            artifact_path="checkpoints",
        )
        log_artifacts(mlflow, figure_paths, artifact_path="figures")
