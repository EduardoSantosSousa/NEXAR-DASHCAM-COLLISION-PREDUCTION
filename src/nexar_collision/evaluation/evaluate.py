"""Temporal alert evaluation for collision prediction models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from nexar_collision.models.baseline_cnn import build_baseline_cnn
from nexar_collision.models.train import build_transforms, resolve_device
from nexar_collision.tracking.mlflow_utils import (
    DEFAULT_EXPERIMENT_NAME,
    log_artifacts,
    log_metrics,
    log_params,
    mlflow_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AlertEvaluationConfig:
    sample_csv: Path = PROJECT_ROOT / "data" / "interim" / "sample_100_videos.csv"
    checkpoint_path: Path = PROJECT_ROOT / "models" / "checkpoints" / "baseline_resnet18.pt"
    risk_scores_path: Path = PROJECT_ROOT / "outputs" / "predictions" / "temporal_risk_scores.csv"
    alert_predictions_path: Path = PROJECT_ROOT / "outputs" / "predictions" / "alert_predictions.csv"
    metrics_path: Path = PROJECT_ROOT / "models" / "reports" / "alert_metrics.json"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    split: str | None = None
    split_column: str = "split"
    fps: float = 1.0
    threshold: float = 0.5
    batch_size: int = 32
    backbone: str | None = None
    device: str = "auto"
    max_videos: int | None = None
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = DEFAULT_EXPERIMENT_NAME
    mlflow_run_name: str | None = None
    mlflow_tracking_uri: str | None = None


def load_sample_df(config: AlertEvaluationConfig) -> pd.DataFrame:
    sample_df = pd.read_csv(config.sample_csv, dtype={"id": str})

    if config.split is not None:
        if config.split_column not in sample_df.columns:
            raise ValueError(
                f"Requested split={config.split}, but column "
                f"{config.split_column!r} was not found in {config.sample_csv}"
            )
        sample_df = sample_df[sample_df[config.split_column] == config.split].copy()

    if config.max_videos is not None:
        sample_df = sample_df.head(config.max_videos).copy()

    if sample_df.empty:
        raise ValueError("No videos available for alert evaluation.")

    return sample_df


def video_duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return float(frame_count / fps) if fps else 0.0


def read_frame_from_capture(
    capture: cv2.VideoCapture,
    video_path: Path,
    timestamp: float,
) -> Image.Image:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()

    if not ok or frame is None:
        raise ValueError(f"Could not read frame at {timestamp:.3f}s from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def timestamps_for_video(duration: float, fps: float) -> list[float]:
    if duration <= 0:
        return [0.0]

    step = 1.0 / fps
    timestamps = np.arange(0.0, max(duration, step), step)
    return [float(min(timestamp, max(duration - 0.05, 0.0))) for timestamp in timestamps]


@torch.no_grad()
def score_video(
    model: torch.nn.Module,
    video_path: Path,
    transform,
    device: torch.device,
    fps: float,
    batch_size: int,
) -> list[dict[str, float]]:
    duration = video_duration(video_path)
    timestamps = timestamps_for_video(duration, fps)
    records: list[dict[str, float]] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        for start in range(0, len(timestamps), batch_size):
            batch_timestamps = timestamps[start : start + batch_size]
            images = [
                transform(read_frame_from_capture(capture, video_path, timestamp))
                for timestamp in batch_timestamps
            ]
            batch = torch.stack(images).to(device)
            logits = model(batch)
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for timestamp, score in zip(batch_timestamps, scores):
                records.append(
                    {
                        "timestamp": round(float(timestamp), 3),
                        "risk_score": float(score),
                        "duration": round(duration, 3),
                    }
                )
    finally:
        capture.release()

    return records


def first_alert_time(video_scores: pd.DataFrame, threshold: float) -> float | None:
    alerts = video_scores[video_scores["risk_score"] >= threshold]
    if alerts.empty:
        return None
    return float(alerts.sort_values("timestamp").iloc[0]["timestamp"])


def load_baseline_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    backbone_override: str | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = dict(checkpoint.get("model_config", {}))
        state_dict = checkpoint["model_state_dict"]
    else:
        # Backward compatibility with older ResNet18 checkpoints saved as raw
        # state_dict objects.
        model_config = {}
        state_dict = checkpoint

    if backbone_override is not None:
        model_config["backbone"] = backbone_override
    model_config.setdefault("backbone", "resnet18")
    model_config.setdefault("num_classes", 2)

    model = build_baseline_cnn(
        backbone=str(model_config["backbone"]),
        num_classes=int(model_config["num_classes"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_config


def compute_alert_metrics(alert_df: pd.DataFrame) -> dict[str, float | int | None]:
    positives = alert_df[alert_df["target"] == 1].copy()
    negatives = alert_df[alert_df["target"] == 0].copy()

    positive_count = len(positives)
    negative_count = len(negatives)
    predicted_alerts = alert_df["predicted_alert_time"].notna().sum()
    negative_alerts = negatives["predicted_alert_time"].notna().sum()

    true_positive_alerts = positives[
        positives["predicted_alert_time"].notna()
        & (positives["predicted_alert_time"] <= positives["time_of_event"])
    ].copy()
    missed_events = positive_count - len(true_positive_alerts)

    alert_precision_denominator = len(true_positive_alerts) + negative_alerts
    alert_precision = (
        len(true_positive_alerts) / alert_precision_denominator
        if alert_precision_denominator
        else None
    )
    alert_recall = len(true_positive_alerts) / positive_count if positive_count else None
    false_alarm_rate = negative_alerts / negative_count if negative_count else None
    missed_event_rate = missed_events / positive_count if positive_count else None

    lead_times = true_positive_alerts["predicted_lead_time"].dropna()
    alert_errors = true_positive_alerts["alert_time_error"].dropna()

    return {
        "videos": int(len(alert_df)),
        "positive_videos": int(positive_count),
        "negative_videos": int(negative_count),
        "predicted_alerts": int(predicted_alerts),
        "true_positive_alerts": int(len(true_positive_alerts)),
        "false_positive_alerts": int(negative_alerts),
        "missed_events": int(missed_events),
        "alert_precision": float(alert_precision) if alert_precision is not None else None,
        "alert_recall": float(alert_recall) if alert_recall is not None else None,
        "false_alarm_rate": float(false_alarm_rate) if false_alarm_rate is not None else None,
        "missed_event_rate": float(missed_event_rate) if missed_event_rate is not None else None,
        "mean_predicted_lead_time": float(lead_times.mean()) if not lead_times.empty else None,
        "median_predicted_lead_time": float(lead_times.median()) if not lead_times.empty else None,
        "mean_alert_time_error": float(alert_errors.mean()) if not alert_errors.empty else None,
        "median_alert_time_error": float(alert_errors.median()) if not alert_errors.empty else None,
    }


def plot_risk_curves(
    risk_scores_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    figures_dir: Path,
    threshold: float,
    figure_prefix: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for target in [1, 0]:
        candidate = sample_df[sample_df["target"] == target].head(1)
        if not candidate.empty:
            examples.append(candidate.iloc[0])

    for row in examples:
        video_id = row["id"]
        target = int(row["target"])
        video_scores = risk_scores_df[risk_scores_df["id"] == video_id]

        if video_scores.empty:
            continue

        plt.figure(figsize=(9, 4))
        plt.plot(video_scores["timestamp"], video_scores["risk_score"], marker="o", linewidth=1)
        plt.axhline(threshold, color="red", linestyle="--", label="alert threshold")
        if target == 1:
            plt.axvline(float(row["time_of_alert"]), color="orange", linestyle="--", label="true alert")
            plt.axvline(float(row["time_of_event"]), color="black", linestyle="--", label="event")
        plt.ylim(0, 1)
        plt.xlabel("Timestamp (s)")
        plt.ylabel("Risk score")
        plt.title(f"Temporal Risk Curve - video {video_id} target={target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{figure_prefix}_temporal_risk_curve_{video_id}.png", dpi=150)
        plt.close()


def evaluate_alerts(config: AlertEvaluationConfig | None = None) -> dict[str, object]:
    config = config or AlertEvaluationConfig()
    device = resolve_device(config.device)
    _, eval_transform = build_transforms()

    sample_df = load_sample_df(config)

    model, model_config = load_baseline_checkpoint(
        checkpoint_path=config.checkpoint_path,
        device=device,
        backbone_override=config.backbone,
    )

    risk_records: list[dict[str, object]] = []
    alert_records: list[dict[str, object]] = []

    total_videos = len(sample_df)
    for video_index, (_, row) in enumerate(sample_df.iterrows(), start=1):
        video_id = row["id"]
        target = int(row["target"])
        video_path = Path(row["video_path"])
        print(f"scoring_video={video_index}/{total_videos} id={video_id} target={target}")

        scores = score_video(
            model=model,
            video_path=video_path,
            transform=eval_transform,
            device=device,
            fps=config.fps,
            batch_size=config.batch_size,
        )
        video_scores = pd.DataFrame(scores)
        predicted_alert_time = first_alert_time(video_scores, config.threshold)

        for record in scores:
            risk_records.append(
                {
                    "id": video_id,
                    "target": target,
                    **record,
                }
            )

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

        alert_records.append(
            {
                "id": video_id,
                "target": target,
                "time_of_alert": time_of_alert,
                "time_of_event": time_of_event,
                "predicted_alert_time": predicted_alert_time,
                "predicted_lead_time": predicted_lead_time,
                "alert_time_error": alert_time_error,
                "max_risk_score": float(video_scores["risk_score"].max()),
                "mean_risk_score": float(video_scores["risk_score"].mean()),
            }
        )

    risk_scores_df = pd.DataFrame(risk_records)
    alert_df = pd.DataFrame(alert_records)
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
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "metrics": metrics,
    }
    config.metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_alert_evaluation_run_to_mlflow(
        config=config,
        report=report,
        figure_prefix=figure_prefix,
    )
    return report


def evaluate_model() -> dict[str, object]:
    return evaluate_alerts()


def log_alert_evaluation_run_to_mlflow(
    config: AlertEvaluationConfig,
    report: dict[str, object],
    figure_prefix: str,
) -> None:
    run_name = config.mlflow_run_name or f"{figure_prefix}_alert_eval"
    tags = {
        "stage": "alert_evaluation",
        "model_family": str(report.get("model_config", {}).get("backbone", "resnet18")),
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
                "stage": "alert_evaluation",
                "sample_csv": config.sample_csv,
                "checkpoint_path": config.checkpoint_path,
                "split": config.split,
                "split_column": config.split_column,
                "fps": config.fps,
                "threshold": config.threshold,
                "batch_size": config.batch_size,
                "backbone": report.get("model_config", {}).get("backbone", "resnet18"),
                "device": config.device,
                "max_videos": config.max_videos,
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
