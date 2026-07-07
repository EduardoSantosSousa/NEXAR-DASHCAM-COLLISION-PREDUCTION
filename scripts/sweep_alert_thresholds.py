from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nexar_collision.evaluation.evaluate import compute_alert_metrics
from nexar_collision.tracking.mlflow_utils import (
    DEFAULT_EXPERIMENT_NAME,
    log_artifacts,
    log_metrics,
    log_params,
    mlflow_run,
)


def load_sample_df(
    sample_csv: Path,
    split: str | None,
    split_column: str,
) -> pd.DataFrame:
    sample_df = pd.read_csv(sample_csv, dtype={"id": str})

    if split is not None:
        if split_column not in sample_df.columns:
            raise ValueError(
                f"Requested split={split}, but column {split_column!r} "
                f"was not found in {sample_csv}"
            )
        sample_df = sample_df[sample_df[split_column] == split].copy()

    if sample_df.empty:
        raise ValueError("No videos available for threshold sweep.")

    return sample_df


def build_alert_predictions(
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
            predicted_alert_time = first_alert_time(
                video_scores,
                threshold,
                min_consecutive_frames,
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


def first_alert_time(
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


def rolling_window_size(video_scores: pd.DataFrame, window_seconds: float) -> int:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than 0.")

    timestamps = video_scores["timestamp"].sort_values().to_numpy(dtype=float)
    if len(timestamps) < 2:
        return 1

    timestamp_diffs = np.diff(timestamps)
    positive_diffs = timestamp_diffs[timestamp_diffs > 0]
    if len(positive_diffs) == 0:
        return 1

    median_step = float(np.median(positive_diffs))
    return max(1, int(np.ceil(window_seconds / median_step)))


def apply_temporal_aggregation(
    risk_scores_df: pd.DataFrame,
    aggregation: str,
    window_seconds: float,
) -> pd.DataFrame:
    if aggregation not in {"raw", "moving_average", "rolling_max"}:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    aggregated_groups = []
    for _, video_scores in risk_scores_df.groupby("id", sort=False):
        video_scores = video_scores.sort_values("timestamp").copy()
        video_scores["raw_risk_score"] = video_scores["risk_score"].astype(float)
        video_scores["aggregation"] = aggregation
        video_scores["aggregation_window_seconds"] = window_seconds

        if aggregation == "raw":
            aggregated_groups.append(video_scores)
            continue

        window_size = rolling_window_size(video_scores, window_seconds)
        video_scores["aggregation_window_frames"] = window_size
        rolling_scores = video_scores["raw_risk_score"].rolling(
            window=window_size,
            min_periods=1,
        )
        if aggregation == "moving_average":
            video_scores["risk_score"] = rolling_scores.mean()
        elif aggregation == "rolling_max":
            video_scores["risk_score"] = rolling_scores.max()

        aggregated_groups.append(video_scores)

    return pd.concat(aggregated_groups, ignore_index=True)


def aggregation_suffix(
    aggregation: str,
    window_seconds: float,
    min_consecutive_frames: int,
) -> str:
    tokens = []
    if aggregation == "moving_average":
        tokens.append(f"ma{window_seconds:g}s")
    elif aggregation == "rolling_max":
        tokens.append(f"rmax{window_seconds:g}s")
    if min_consecutive_frames > 1:
        tokens.append(f"consecutive{min_consecutive_frames}")
    return f"_{'_'.join(tokens)}" if tokens else ""


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(sweep_df["threshold"], sweep_df["alert_recall"], marker="o", label="alert recall")
    plt.plot(sweep_df["threshold"], sweep_df["alert_precision"], marker="o", label="alert precision")
    plt.plot(sweep_df["threshold"], sweep_df["false_alarm_rate"], marker="o", label="false alarm rate")
    plt.xlabel("Alert threshold")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    title = "Temporal Alert Threshold Sweep"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def choose_operating_point(
    sweep_df: pd.DataFrame,
    min_recall: float,
) -> pd.Series | None:
    candidates = sweep_df[sweep_df["alert_recall"] >= min_recall].copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(
        ["false_alarm_rate", "missed_event_rate", "alert_precision", "mean_alert_time_error"],
        ascending=[True, True, False, False],
    )
    return candidates.iloc[0]


def log_threshold_sweep_to_mlflow(
    args: argparse.Namespace,
    risk_scores_path: Path,
    output_path: Path,
    figure_path: Path,
    sweep_df: pd.DataFrame,
    candidate: pd.DataFrame,
    run_suffix: str,
) -> None:
    run_name = args.mlflow_run_name or f"{args.experiment_name}{run_suffix}_threshold_sweep"
    tags = {
        "stage": "threshold_sweep",
        "experiment_name": args.experiment_name,
        "aggregation": args.aggregation,
        "min_consecutive_frames": args.min_consecutive_frames,
    }

    with mlflow_run(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment_name,
        run_name=run_name,
        tracking_uri=args.mlflow_tracking_uri,
        tags=tags,
    ) as mlflow:
        log_params(
            mlflow,
            {
                "stage": "threshold_sweep",
                "experiment_name": args.experiment_name,
                "risk_scores_path": risk_scores_path,
                "sample_csv": args.sample_csv,
                "split": args.split,
                "split_column": args.split_column,
                "start": args.start,
                "stop": args.stop,
                "step": args.step,
                "aggregation": args.aggregation,
                "aggregation_window_seconds": args.aggregation_window_seconds,
                "min_consecutive_frames": args.min_consecutive_frames,
            },
        )

        if not candidate.empty:
            candidate_metrics = candidate.iloc[0].to_dict()
            log_metrics(mlflow, candidate_metrics, prefix="candidate_")

        for min_recall in [0.7, 0.6, 0.5, 0.4, 0.3]:
            point = choose_operating_point(sweep_df, min_recall)
            if point is None:
                continue
            token = str(min_recall).replace(".", "_")
            log_metrics(
                mlflow,
                {
                    "threshold": point["threshold"],
                    "alert_precision": point["alert_precision"],
                    "alert_recall": point["alert_recall"],
                    "false_alarm_rate": point["false_alarm_rate"],
                    "missed_event_rate": point["missed_event_rate"],
                    "mean_predicted_lead_time": point["mean_predicted_lead_time"],
                    "mean_alert_time_error": point["mean_alert_time_error"],
                },
                prefix=f"min_recall_{token}_",
            )

        log_artifacts(mlflow, [output_path, risk_scores_path], artifact_path="reports")
        log_artifacts(mlflow, [figure_path], artifact_path="figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep temporal alert thresholds.")
    parser.add_argument(
        "--risk-scores",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "sample_100_videos.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--split", default=None)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--start", type=float, default=0.1)
    parser.add_argument("--stop", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--aggregation",
        choices=["raw", "moving_average", "rolling_max"],
        default="raw",
        help="Optional causal temporal score aggregation before thresholding.",
    )
    parser.add_argument(
        "--aggregation-window-seconds",
        type=float,
        default=3.0,
        help="Trailing time window used by moving_average and rolling_max.",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=1,
        help="Require N consecutive frames above threshold before alerting.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_suffix = aggregation_suffix(
        args.aggregation,
        args.aggregation_window_seconds,
        args.min_consecutive_frames,
    )
    risk_scores_path = args.risk_scores or (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / f"{args.experiment_name}_temporal_risk_scores.csv"
    )
    output_path = args.output or (
        PROJECT_ROOT
        / "models"
        / "reports"
        / f"{args.experiment_name}{run_suffix}_alert_threshold_sweep.csv"
    )

    if not risk_scores_path.exists():
        raise FileNotFoundError(
            f"Risk scores file not found: {risk_scores_path}\n"
            "Run scripts/evaluate_model.py for this experiment before sweeping "
            "thresholds. Example:\n"
            f".\\venv\\Scripts\\python.exe scripts\\evaluate_model.py "
            f"--checkpoint models\\checkpoints\\{args.experiment_name}_resnet18.pt "
            f"--experiment-name {args.experiment_name} "
            f"--sample-csv {args.sample_csv} "
            f"{'--split ' + args.split if args.split else ''}"
        )

    risk_scores_df = pd.read_csv(risk_scores_path, dtype={"id": str})
    risk_scores_df = apply_temporal_aggregation(
        risk_scores_df,
        args.aggregation,
        args.aggregation_window_seconds,
    )
    logged_risk_scores_path = risk_scores_path
    if run_suffix:
        logged_risk_scores_path = (
            PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{args.experiment_name}{run_suffix}_temporal_risk_scores.csv"
        )
        logged_risk_scores_path.parent.mkdir(parents=True, exist_ok=True)
        risk_scores_df.to_csv(logged_risk_scores_path, index=False)

    sample_df = load_sample_df(args.sample_csv, args.split, args.split_column)

    records = []
    thresholds = np.round(np.arange(args.start, args.stop + args.step / 2, args.step), 3)
    for threshold in thresholds:
        alert_df = build_alert_predictions(
            risk_scores_df,
            sample_df,
            float(threshold),
            args.min_consecutive_frames,
        )
        metrics = compute_alert_metrics(alert_df)
        records.append({"threshold": float(threshold), **metrics})

    sweep_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_path, index=False)
    figure_path = (
        PROJECT_ROOT
        / "outputs"
        / "figures"
        / f"{args.experiment_name}{run_suffix}_alert_threshold_sweep.png"
    )
    plot_threshold_sweep(
        sweep_df,
        figure_path,
        run_suffix.strip("_").replace("_", " "),
    )

    candidate = sweep_df.sort_values(
        ["false_alarm_rate", "missed_event_rate", "mean_alert_time_error"],
        ascending=[True, True, False],
    ).head(1)
    log_threshold_sweep_to_mlflow(
        args=args,
        risk_scores_path=logged_risk_scores_path,
        output_path=output_path,
        figure_path=figure_path,
        sweep_df=sweep_df,
        candidate=candidate,
        run_suffix=run_suffix,
    )
    print(f"Saved threshold sweep to: {output_path}")
    if logged_risk_scores_path != risk_scores_path:
        print(f"Saved aggregated risk scores to: {logged_risk_scores_path}")
    print("Candidate operating point:")
    print(candidate.to_string(index=False))
    print("\nOperating points by minimum recall:")
    operating_points = []
    for min_recall in [0.7, 0.6, 0.5, 0.4, 0.3]:
        point = choose_operating_point(sweep_df, min_recall)
        if point is not None:
            operating_points.append({"min_recall": min_recall, **point.to_dict()})
    if operating_points:
        print(pd.DataFrame(operating_points).to_string(index=False))


if __name__ == "__main__":
    main()
