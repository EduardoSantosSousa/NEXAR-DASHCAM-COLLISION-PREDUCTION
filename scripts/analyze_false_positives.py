from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ERROR_TYPE_OPTIONS = [
    "camera_motion",
    "close_vehicle_no_collision",
    "traffic_density",
    "lane_change_or_turn",
    "visual_occlusion",
    "night_or_low_quality",
    "label_or_timing_suspicion",
    "background_bias",
    "unknown",
]


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return thresholds


def timestamp_token(timestamp: float) -> str:
    return f"{timestamp:07.3f}".replace(".", "_")


def ensure_probability_columns(risk_scores_df: pd.DataFrame) -> pd.DataFrame:
    risk_scores_df = risk_scores_df.copy()
    for column in ["prob_class_2", "prob_class_3"]:
        if column not in risk_scores_df.columns:
            risk_scores_df[column] = np.nan
    return risk_scores_df


def load_inputs(
    risk_scores_path: Path,
    sample_csv: Path,
    frame_manifest: Path,
    split: str | None,
    split_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk_scores_df = pd.read_csv(risk_scores_path, dtype={"id": str})
    risk_scores_df = ensure_probability_columns(risk_scores_df)

    sample_df = pd.read_csv(sample_csv, dtype={"id": str})
    if split is not None:
        if split_column not in sample_df.columns:
            raise ValueError(
                f"Requested split={split}, but {split_column!r} was not found."
            )
        sample_df = sample_df[sample_df[split_column] == split].copy()

    if sample_df.empty:
        raise ValueError("No videos available after applying split filter.")

    frame_df = pd.read_csv(frame_manifest, dtype={"id": str})
    return risk_scores_df, sample_df, frame_df


def count_above_segments(
    sorted_scores: pd.DataFrame,
    threshold: float,
) -> tuple[int, float]:
    above = sorted_scores["risk_score"].astype(float).to_numpy() >= threshold
    timestamps = sorted_scores["timestamp"].astype(float).to_numpy()
    if len(above) == 0:
        return 0, 0.0

    segment_count = 0
    longest_segment_seconds = 0.0
    start_index: int | None = None

    for index, is_above in enumerate(above):
        if is_above and start_index is None:
            start_index = index
        if (not is_above or index == len(above) - 1) and start_index is not None:
            end_index = index if is_above and index == len(above) - 1 else index - 1
            segment_count += 1
            longest_segment_seconds = max(
                longest_segment_seconds,
                float(timestamps[end_index] - timestamps[start_index]),
            )
            start_index = None

    return segment_count, longest_segment_seconds


def summarize_video_scores(
    video_scores: pd.DataFrame,
    threshold: float,
    thresholds: list[float],
) -> dict[str, object]:
    sorted_scores = video_scores.sort_values("timestamp").copy()
    top_row = sorted_scores.sort_values("risk_score", ascending=False).iloc[0]
    above_threshold = sorted_scores["risk_score"].astype(float) >= threshold
    first_alert_rows = sorted_scores[above_threshold]
    segment_count, longest_segment_seconds = count_above_segments(
        sorted_scores,
        threshold,
    )

    summary: dict[str, object] = {
        "max_risk_score": float(sorted_scores["risk_score"].max()),
        "mean_risk_score": float(sorted_scores["risk_score"].mean()),
        "p90_risk_score": float(sorted_scores["risk_score"].quantile(0.90)),
        "frames_scored": int(len(sorted_scores)),
        "frames_above_threshold": int(above_threshold.sum()),
        "fraction_above_threshold": float(above_threshold.mean()),
        "first_false_alert_time": (
            float(first_alert_rows.iloc[0]["timestamp"])
            if not first_alert_rows.empty
            else None
        ),
        "top_risk_timestamp": float(top_row["timestamp"]),
        "top_risk_score": float(top_row["risk_score"]),
        "top_prob_class_2": (
            float(top_row["prob_class_2"]) if pd.notna(top_row["prob_class_2"]) else None
        ),
        "top_prob_class_3": (
            float(top_row["prob_class_3"]) if pd.notna(top_row["prob_class_3"]) else None
        ),
        "risk_segments_above_threshold": int(segment_count),
        "longest_segment_seconds": float(longest_segment_seconds),
    }

    for item in thresholds:
        key = str(item).replace(".", "_")
        above_item = sorted_scores["risk_score"].astype(float) >= item
        summary[f"frames_above_{key}"] = int(above_item.sum())
        summary[f"fraction_above_{key}"] = float(above_item.mean())

    return summary


def nearest_frame_rows(
    frame_video_df: pd.DataFrame,
    timestamps: list[float],
) -> pd.DataFrame:
    if frame_video_df.empty:
        return pd.DataFrame()

    frames = frame_video_df.copy()
    frames["timestamp"] = frames["timestamp"].astype(float)
    selected_rows = []
    for timestamp in timestamps:
        nearest_index = (frames["timestamp"] - float(timestamp)).abs().idxmin()
        selected_rows.append(frames.loc[nearest_index])
    return pd.DataFrame(selected_rows).drop_duplicates(subset=["frame_path"])


def copy_top_frames(
    video_scores: pd.DataFrame,
    frame_video_df: pd.DataFrame,
    output_dir: Path,
    top_frames: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    top_score_rows = (
        video_scores.sort_values("risk_score", ascending=False)
        .head(top_frames)
        .copy()
    )
    frame_rows = nearest_frame_rows(
        frame_video_df,
        top_score_rows["timestamp"].astype(float).tolist(),
    )

    copied_paths = []
    for _, frame_row in frame_rows.iterrows():
        source_path = Path(str(frame_row["frame_path"]))
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.exists():
            continue

        timestamp = float(frame_row["timestamp"])
        score_row = top_score_rows.iloc[
            (top_score_rows["timestamp"].astype(float) - timestamp).abs().argmin()
        ]
        destination = (
            output_dir
            / f"t_{timestamp_token(timestamp)}_risk_{float(score_row['risk_score']):.3f}.jpg"
        )
        shutil.copy2(source_path, destination)
        copied_paths.append(str(destination))

    return copied_paths


def plot_video_risk_curve(
    video_scores: pd.DataFrame,
    output_path: Path,
    threshold: float,
    title: str,
    time_of_alert: float | None = None,
    time_of_event: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_scores = video_scores.sort_values("timestamp")

    plt.figure(figsize=(10, 4))
    plt.plot(
        sorted_scores["timestamp"],
        sorted_scores["risk_score"],
        label="alert risk",
        linewidth=2,
    )
    if "prob_class_2" in sorted_scores.columns:
        plt.plot(
            sorted_scores["timestamp"],
            sorted_scores["prob_class_2"],
            label="P(prealert_late)",
            alpha=0.65,
        )
    if "prob_class_3" in sorted_scores.columns:
        plt.plot(
            sorted_scores["timestamp"],
            sorted_scores["prob_class_3"],
            label="P(event_near)",
            alpha=0.65,
        )

    plt.axhline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:g}")
    if time_of_alert is not None:
        plt.axvline(time_of_alert, color="tab:orange", linestyle=":", label="time_of_alert")
    if time_of_event is not None:
        plt.axvline(time_of_event, color="tab:purple", linestyle=":", label="time_of_event")

    plt.title(title)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_false_positive_ranking(
    risk_scores_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    threshold: float,
    thresholds: list[float],
) -> pd.DataFrame:
    negative_ids = set(sample_df[sample_df["target"].astype(int) == 0]["id"])
    negative_scores = risk_scores_df[risk_scores_df["id"].isin(negative_ids)].copy()

    records = []
    for video_id, video_scores in negative_scores.groupby("id", sort=False):
        sample_row = sample_df[sample_df["id"] == video_id].iloc[0]
        summary = summarize_video_scores(video_scores, threshold, thresholds)
        records.append(
            {
                "id": str(video_id),
                "target": 0,
                "split": sample_row.get("split"),
                "video_path": sample_row.get("video_path"),
                **summary,
                "suggested_error_type": "unknown",
                "manual_error_type": "",
                "notes": "",
            }
        )

    ranking_df = pd.DataFrame(records)
    if ranking_df.empty:
        return ranking_df

    return ranking_df.sort_values(
        [
            "frames_above_threshold",
            "max_risk_score",
            "risk_segments_above_threshold",
            "mean_risk_score",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_positive_reference_ranking(
    risk_scores_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    threshold: float,
    thresholds: list[float],
) -> pd.DataFrame:
    positive_ids = set(sample_df[sample_df["target"].astype(int) == 1]["id"])
    positive_scores = risk_scores_df[risk_scores_df["id"].isin(positive_ids)].copy()

    records = []
    for video_id, video_scores in positive_scores.groupby("id", sort=False):
        sample_row = sample_df[sample_df["id"] == video_id].iloc[0]
        summary = summarize_video_scores(video_scores, threshold, thresholds)
        records.append(
            {
                "id": str(video_id),
                "target": 1,
                "split": sample_row.get("split"),
                "time_of_alert": sample_row.get("time_of_alert"),
                "time_of_event": sample_row.get("time_of_event"),
                "video_path": sample_row.get("video_path"),
                **summary,
            }
        )

    ranking_df = pd.DataFrame(records)
    if ranking_df.empty:
        return ranking_df

    return ranking_df.sort_values(
        ["max_risk_score", "frames_above_threshold", "mean_risk_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_markdown_report(
    output_path: Path,
    experiment_name: str,
    threshold: float,
    ranking_df: pd.DataFrame,
    positive_reference_df: pd.DataFrame,
    top_videos: int,
    figures_dir: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_false = ranking_df.head(top_videos)
    top_positive = positive_reference_df.head(min(10, len(positive_reference_df)))

    false_alert_count = int((ranking_df["frames_above_threshold"] > 0).sum())
    total_negative = int(len(ranking_df))
    false_alert_rate = false_alert_count / total_negative if total_negative else 0.0

    lines = [
        f"# False positive error analysis - {experiment_name}",
        "",
        "## Objective",
        "",
        "Identify negative validation videos that repeatedly trigger high alert",
        "scores, then prepare visual artifacts for manual error categorization.",
        "",
        "## Configuration",
        "",
        f"- Threshold used for ranking: `{threshold:.3f}`",
        f"- Negative validation videos analyzed: `{total_negative}`",
        f"- Negative videos with at least one false alert: `{false_alert_count}`",
        f"- False-alert video rate at this threshold: `{false_alert_rate:.3f}`",
        f"- Figures directory: `{figures_dir}`",
        "",
        "## Error Type Taxonomy",
        "",
    ]
    for option in ERROR_TYPE_OPTIONS:
        lines.append(f"- `{option}`")

    lines.extend(
        [
            "",
            "## Top False Positive Videos",
            "",
            "| Rank | Video ID | Max risk | Mean risk | Frames above threshold | First false alert | Segments | Suggested type |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for rank, (_, row) in enumerate(top_false.iterrows(), start=1):
        first_alert = row["first_false_alert_time"]
        first_alert_text = "" if pd.isna(first_alert) else f"{float(first_alert):.3f}"
        lines.append(
            "| "
            f"{rank} | {row['id']} | {float(row['max_risk_score']):.3f} | "
            f"{float(row['mean_risk_score']):.3f} | "
            f"{int(row['frames_above_threshold'])} | {first_alert_text} | "
            f"{int(row['risk_segments_above_threshold'])} | "
            f"{row['suggested_error_type']} |"
        )

    lines.extend(
        [
            "",
            "## Positive Reference Videos",
            "",
            "Use these examples to compare high-risk true positives against false",
            "positive videos.",
            "",
            "| Rank | Video ID | Max risk | Mean risk | Frames above threshold | Top risk timestamp |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for rank, (_, row) in enumerate(top_positive.iterrows(), start=1):
        lines.append(
            "| "
            f"{rank} | {row['id']} | {float(row['max_risk_score']):.3f} | "
            f"{float(row['mean_risk_score']):.3f} | "
            f"{int(row['frames_above_threshold'])} | "
            f"{float(row['top_risk_timestamp']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Manual Review Instructions",
            "",
            "1. Open each top false-positive folder under the figures directory.",
            "2. Compare the risk curve with the copied top-risk frames.",
            "3. Fill `manual_error_type` and `notes` in the review CSV.",
            "4. Use the dominant error type to choose the next modeling change.",
            "",
            "Recommended decision rules:",
            "",
            "- many `close_vehicle_no_collision`: mine harder safe negatives;",
            "- many `camera_motion` or `lane_change_or_turn`: add stronger temporal context;",
            "- many `label_or_timing_suspicion`: revise phase windows or labels;",
            "- many visually ambiguous cases: consider risk-level product UX before binary alerting.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_visual_artifacts(
    ranking_df: pd.DataFrame,
    positive_reference_df: pd.DataFrame,
    risk_scores_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    figures_dir: Path,
    threshold: float,
    top_videos: int,
    top_frames: int,
    positive_reference_count: int,
) -> None:
    frame_groups = {video_id: df for video_id, df in frame_df.groupby("id", sort=False)}
    sample_lookup = sample_df.set_index("id", drop=False)

    for _, row in ranking_df.head(top_videos).iterrows():
        video_id = str(row["id"])
        video_dir = figures_dir / "top_false_positives" / video_id
        video_scores = risk_scores_df[risk_scores_df["id"] == video_id]
        frame_video_df = frame_groups.get(video_id, pd.DataFrame())
        plot_video_risk_curve(
            video_scores=video_scores,
            output_path=video_dir / f"{video_id}_risk_curve.png",
            threshold=threshold,
            title=f"False positive candidate {video_id}",
        )
        copy_top_frames(
            video_scores=video_scores,
            frame_video_df=frame_video_df,
            output_dir=video_dir / "top_frames",
            top_frames=top_frames,
        )

    for _, row in positive_reference_df.head(positive_reference_count).iterrows():
        video_id = str(row["id"])
        video_dir = figures_dir / "positive_references" / video_id
        video_scores = risk_scores_df[risk_scores_df["id"] == video_id]
        frame_video_df = frame_groups.get(video_id, pd.DataFrame())
        sample_row = sample_lookup.loc[video_id] if video_id in sample_lookup.index else None
        time_of_alert = (
            float(sample_row["time_of_alert"])
            if sample_row is not None and pd.notna(sample_row["time_of_alert"])
            else None
        )
        time_of_event = (
            float(sample_row["time_of_event"])
            if sample_row is not None and pd.notna(sample_row["time_of_event"])
            else None
        )
        plot_video_risk_curve(
            video_scores=video_scores,
            output_path=video_dir / f"{video_id}_risk_curve.png",
            threshold=threshold,
            title=f"Positive reference {video_id}",
            time_of_alert=time_of_alert,
            time_of_event=time_of_event,
        )
        copy_top_frames(
            video_scores=video_scores,
            frame_video_df=frame_video_df,
            output_dir=video_dir / "top_frames",
            top_frames=top_frames,
        )


def analyze_false_positives(args: argparse.Namespace) -> dict[str, object]:
    thresholds = parse_thresholds(args.thresholds)
    if args.threshold not in thresholds:
        thresholds = sorted(set([args.threshold, *thresholds]))

    risk_scores_df, sample_df, frame_df = load_inputs(
        risk_scores_path=args.risk_scores,
        sample_csv=args.sample_csv,
        frame_manifest=args.frame_manifest,
        split=args.split,
        split_column=args.split_column,
    )

    ranking_df = build_false_positive_ranking(
        risk_scores_df=risk_scores_df,
        sample_df=sample_df,
        threshold=args.threshold,
        thresholds=thresholds,
    )
    positive_reference_df = build_positive_reference_ranking(
        risk_scores_df=risk_scores_df,
        sample_df=sample_df,
        threshold=args.threshold,
        thresholds=thresholds,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    args.positive_reference_csv.parent.mkdir(parents=True, exist_ok=True)

    ranking_df.to_csv(args.output_csv, index=False)
    ranking_df.head(args.top_videos).to_csv(args.review_csv, index=False)
    positive_reference_df.to_csv(args.positive_reference_csv, index=False)

    generate_visual_artifacts(
        ranking_df=ranking_df,
        positive_reference_df=positive_reference_df,
        risk_scores_df=risk_scores_df,
        frame_df=frame_df,
        sample_df=sample_df,
        figures_dir=args.figures_dir,
        threshold=args.threshold,
        top_videos=args.top_videos,
        top_frames=args.top_frames,
        positive_reference_count=args.positive_reference_count,
    )

    write_markdown_report(
        output_path=args.report,
        experiment_name=args.experiment_name,
        threshold=args.threshold,
        ranking_df=ranking_df,
        positive_reference_df=positive_reference_df,
        top_videos=args.top_videos,
        figures_dir=args.figures_dir,
    )

    false_alert_count = int((ranking_df["frames_above_threshold"] > 0).sum())
    summary = {
        "experiment_name": args.experiment_name,
        "threshold": args.threshold,
        "negative_videos": int(len(ranking_df)),
        "negative_videos_with_false_alert": false_alert_count,
        "false_alert_video_rate": (
            false_alert_count / len(ranking_df) if len(ranking_df) else 0.0
        ),
        "output_csv": str(args.output_csv),
        "review_csv": str(args.review_csv),
        "positive_reference_csv": str(args.positive_reference_csv),
        "report": str(args.report),
        "figures_dir": str(args.figures_dir),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank and inspect false positives from temporal risk scores."
    )
    parser.add_argument(
        "--risk-scores",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "product_event_window_phase_classifier_seq8_temporal_risk_scores.csv",
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "full_train_product_splits.csv",
    )
    parser.add_argument(
        "--frame-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "interim"
        / "product_temporal_frames_224_manifest.csv",
    )
    parser.add_argument("--experiment-name", default="product_event_window_phase_classifier_seq8")
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--threshold", type=float, default=0.64)
    parser.add_argument("--thresholds", default="0.50,0.64")
    parser.add_argument("--top-videos", type=int, default=20)
    parser.add_argument("--top-frames", type=int, default=6)
    parser.add_argument("--positive-reference-count", type=int, default=10)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_false_positive_error_analysis.csv",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_false_positive_manual_review.csv",
    )
    parser.add_argument(
        "--positive-reference-csv",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_positive_reference_analysis.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_false_positive_error_analysis_summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "product_false_positive_error_analysis.md",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "figures"
        / "product_false_positive_error_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_false_positives(args)
    print("False-positive analysis complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
