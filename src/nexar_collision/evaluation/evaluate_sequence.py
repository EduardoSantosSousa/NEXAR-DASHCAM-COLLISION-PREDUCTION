"""Temporal alert evaluation for CNN + GRU/LSTM sequence models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from nexar_collision.data.dataset import FrameSequenceCollisionDataset
from nexar_collision.evaluation.evaluate import compute_alert_metrics, plot_risk_curves
from nexar_collision.models.temporal_model import build_temporal_model
from nexar_collision.models.train import build_transforms, load_manifest_with_optional_split, resolve_device
from nexar_collision.tracking.mlflow_utils import (
    DEFAULT_EXPERIMENT_NAME,
    log_artifacts,
    log_metrics,
    log_params,
    mlflow_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SequenceAlertEvaluationConfig:
    manifest_path: Path = PROJECT_ROOT / "data" / "interim" / "temporal_frames_224_manifest.csv"
    sample_csv: Path = PROJECT_ROOT / "data" / "interim" / "sample_100_videos_splits.csv"
    split_manifest_path: Path | None = PROJECT_ROOT / "data" / "interim" / "sample_100_videos_splits.csv"
    checkpoint_path: Path = PROJECT_ROOT / "models" / "checkpoints" / "sequence_best_model.pt"
    risk_scores_path: Path = PROJECT_ROOT / "outputs" / "predictions" / "sequence_temporal_risk_scores.csv"
    alert_predictions_path: Path = PROJECT_ROOT / "outputs" / "predictions" / "sequence_alert_predictions.csv"
    metrics_path: Path = PROJECT_ROOT / "models" / "reports" / "sequence_alert_metrics.json"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    split: str | None = "val"
    split_column: str = "split"
    sequence_length: int | None = None
    batch_size: int = 8
    threshold: float = 0.5
    device: str = "auto"
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = DEFAULT_EXPERIMENT_NAME
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None


def load_sequence_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Sequence checkpoints must contain model_state_dict and model_config."
        )

    model_config = checkpoint.get("model_config", {})
    sequence_length = int(checkpoint.get("sequence_length", 4))
    model = build_temporal_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, sequence_length, model_config


def load_eval_manifest(config: SequenceAlertEvaluationConfig) -> pd.DataFrame:
    manifest_df = load_manifest_with_optional_split(config)  # type: ignore[arg-type]
    if config.split is not None:
        if config.split_column not in manifest_df.columns:
            raise ValueError(
                f"Requested split={config.split}, but column "
                f"{config.split_column!r} was not found."
            )
        manifest_df = manifest_df[manifest_df[config.split_column] == config.split].copy()
    if manifest_df.empty:
        raise ValueError("No frames available for sequence evaluation.")
    return manifest_df


@torch.no_grad()
def score_sequences(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    records = []
    model.eval()
    for batch in loader:
        images = batch["images"].to(device)
        logits = model(images)
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        for index, score in enumerate(scores):
            records.append(
                {
                    "id": batch["video_id"][index],
                    "target": int(batch["video_target"][index]),
                    "frame_target": int(batch["target"][index]),
                    "timestamp": float(batch["timestamp"][index]),
                    "duration": float(batch["duration"][index]),
                    "risk_score": float(score),
                }
            )
    return pd.DataFrame(records)


def build_alert_predictions(
    risk_scores_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    threshold: float,
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
            alerts = video_scores[video_scores["risk_score"] >= threshold]
            predicted_alert_time = (
                float(alerts.sort_values("timestamp").iloc[0]["timestamp"])
                if not alerts.empty
                else None
            )
            max_risk_score = float(video_scores["risk_score"].max())
            mean_risk_score = float(video_scores["risk_score"].mean())

        time_of_alert = float(row["time_of_alert"]) if pd.notna(row["time_of_alert"]) else None
        time_of_event = float(row["time_of_event"]) if pd.notna(row["time_of_event"]) else None
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


def evaluate_sequence_alerts(
    config: SequenceAlertEvaluationConfig | None = None,
) -> dict[str, object]:
    config = config or SequenceAlertEvaluationConfig()
    device = resolve_device(config.device)
    model, checkpoint_sequence_length, model_config = load_sequence_checkpoint(
        config.checkpoint_path,
        device,
    )
    sequence_length = config.sequence_length or checkpoint_sequence_length
    _, eval_transform = build_transforms()

    manifest_df = load_eval_manifest(config)
    dataset = FrameSequenceCollisionDataset(
        manifest_df,
        sequence_length=sequence_length,
        transform=eval_transform,
        sequence_stride=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    risk_scores_df = score_sequences(model, loader, device)
    sample_df = pd.read_csv(config.sample_csv, dtype={"id": str})
    if config.split is not None:
        sample_df = sample_df[sample_df[config.split_column] == config.split].copy()

    alert_df = build_alert_predictions(risk_scores_df, sample_df, config.threshold)
    metrics = compute_alert_metrics(alert_df)

    config.risk_scores_path.parent.mkdir(parents=True, exist_ok=True)
    config.alert_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    risk_scores_df.to_csv(config.risk_scores_path, index=False)
    alert_df.to_csv(config.alert_predictions_path, index=False)

    figure_prefix = config.metrics_path.stem.replace("_alert_metrics", "")
    plot_risk_curves(
        risk_scores_df=risk_scores_df,
        sample_df=sample_df,
        figures_dir=config.figures_dir,
        threshold=config.threshold,
        figure_prefix=figure_prefix,
    )

    report = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "model_config": model_config,
        "sequence_length": sequence_length,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "metrics": metrics,
    }
    config.metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_sequence_alert_evaluation_to_mlflow(config, report, figure_prefix)
    return report


def log_sequence_alert_evaluation_to_mlflow(
    config: SequenceAlertEvaluationConfig,
    report: dict[str, object],
    figure_prefix: str,
) -> None:
    run_name = config.mlflow_run_name or f"{figure_prefix}_alert_eval"
    tags = {
        "stage": "alert_evaluation",
        "model_family": "cnn_rnn",
        "experiment_name": figure_prefix,
    }
    risk_curve_paths = sorted(
        config.figures_dir.glob(f"{figure_prefix}_temporal_risk_curve_*.png")
    )

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
                **asdict(config),
                "sequence_length": report["sequence_length"],
                **report["model_config"],  # type: ignore[arg-type]
            },
        )
        log_metrics(mlflow, report["metrics"], prefix="alert_")  # type: ignore[arg-type]
        log_artifacts(
            mlflow,
            [
                config.metrics_path,
                config.risk_scores_path,
                config.alert_predictions_path,
            ],
            artifact_path="reports",
        )
        log_artifacts(mlflow, risk_curve_paths, artifact_path="figures")
