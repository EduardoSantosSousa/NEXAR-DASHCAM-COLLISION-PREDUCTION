from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

from nexar_collision.evaluation.evaluate_sequence import (
    load_sequence_checkpoint,
    score_sequences,
)
from nexar_collision.models.train import build_transforms, resolve_device
from nexar_collision.models.train_sequence import build_sequence_dataset


def parse_int_offsets(value: str) -> list[int]:
    offsets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not offsets:
        raise ValueError("At least one neighbor frame offset must be provided.")
    return offsets


def optional_float(value) -> float | None:
    return float(value) if pd.notna(value) else None


def timestamp_token(timestamp: float) -> str:
    return f"{timestamp:07.3f}".replace(".", "_")


def load_frame_manifest_with_split(
    frame_manifest: Path,
    split_manifest: Path,
    split_column: str,
) -> pd.DataFrame:
    frame_df = pd.read_csv(frame_manifest, dtype={"id": str})
    if split_column in frame_df.columns:
        frame_df = frame_df.drop(columns=[split_column])

    split_df = pd.read_csv(split_manifest, dtype={"id": str})
    required_columns = {"id", split_column, "target"}
    missing_columns = required_columns - set(split_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required split columns: {sorted(missing_columns)}"
        )

    merge_columns = ["id", split_column, "target"]
    if "video_path" in split_df.columns and "video_path" not in frame_df.columns:
        merge_columns.append("video_path")

    frame_df = frame_df.merge(
        split_df[merge_columns],
        on="id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_split"),
    )
    if split_column != "split":
        frame_df = frame_df.rename(columns={split_column: "split"})
    if "video_target" not in frame_df.columns:
        frame_df["video_target"] = frame_df["target_split"].astype(int)
    if "target_split" in frame_df.columns:
        frame_df = frame_df.drop(columns=["target_split"])

    return frame_df


def score_train_negative_frames(
    frame_df: pd.DataFrame,
    checkpoint_path: Path,
    sequence_length: int,
    batch_size: int,
    num_workers: int,
    device_name: str,
    output_path: Path,
) -> pd.DataFrame:
    negative_frame_df = frame_df[
        (frame_df["split"] == "train") & (frame_df["video_target"].astype(int) == 0)
    ].copy()
    if negative_frame_df.empty:
        raise ValueError("No train negative frames available for mining.")

    device = resolve_device(device_name)
    model, checkpoint_sequence_length, _, alert_class_indices = load_sequence_checkpoint(
        checkpoint_path,
        device,
    )
    sequence_length = sequence_length or checkpoint_sequence_length
    _, eval_transform = build_transforms()

    dataset = build_sequence_dataset(
        negative_frame_df,
        sequence_length=sequence_length,
        transform=eval_transform,
        sequence_stride=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(
        "scoring_train_negative_frames="
        f"videos={negative_frame_df['id'].nunique()} frames={len(negative_frame_df)}",
        flush=True,
    )
    with torch.no_grad():
        scores_df = score_sequences(
            model=model,
            loader=loader,
            device=device,
            alert_class_indices=alert_class_indices,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path, index=False)
    return scores_df


def select_peak_timestamps(
    video_scores: pd.DataFrame,
    threshold: float,
    max_peaks: int,
    min_gap_seconds: float,
) -> pd.DataFrame:
    candidates = video_scores[video_scores["risk_score"] >= threshold].copy()
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values("risk_score", ascending=False)
    selected_rows = []
    selected_timestamps: list[float] = []

    for _, row in candidates.iterrows():
        timestamp = float(row["timestamp"])
        if any(abs(timestamp - selected) < min_gap_seconds for selected in selected_timestamps):
            continue
        selected_rows.append(row)
        selected_timestamps.append(timestamp)
        if len(selected_rows) >= max_peaks:
            break

    if not selected_rows:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(selected_rows)


def window_payload(
    video_df: pd.DataFrame,
    end_offset: int,
    sequence_length: int,
) -> tuple[list[str], list[float]]:
    start_offset = max(0, end_offset - sequence_length + 1)
    row_offsets = list(range(start_offset, end_offset + 1))
    if len(row_offsets) < sequence_length:
        row_offsets = [row_offsets[0]] * (sequence_length - len(row_offsets)) + row_offsets

    rows = video_df.iloc[row_offsets]
    return (
        [str(path) for path in rows["frame_path"].tolist()],
        [float(timestamp) for timestamp in rows["timestamp"].tolist()],
    )


def add_context_window_record(
    records: list[dict[str, object]],
    video_df: pd.DataFrame,
    end_offset: int,
    sequence_length: int,
    sample_weight: float,
    source_peak: pd.Series,
    counter: int,
) -> None:
    end_row = video_df.iloc[end_offset]
    frame_paths, timestamps = window_payload(video_df, end_offset, sequence_length)
    video_id = str(end_row["id"])
    window_end = float(end_row["timestamp"])
    window_start = float(timestamps[0])
    token = timestamp_token(window_end)

    records.append(
        {
            "window_id": f"{video_id}_context_hard_negative_{counter:03d}_{token}",
            "id": video_id,
            "video_target": 0,
            "target": 0,
            "temporal_target": 0,
            "window_type": "contextual_hard_negative",
            "phase_name": "contextual_hard_negative",
            "phase_index": 0,
            "sample_weight": float(sample_weight),
            "timestamp": window_end,
            "window_start_timestamp": window_start,
            "window_end_timestamp": window_end,
            "center_timestamp": float((window_start + window_end) / 2.0),
            "sequence_length": int(sequence_length),
            "frame_count": len(frame_paths),
            "frame_paths": json.dumps(frame_paths),
            "timestamps": json.dumps(timestamps),
            "time_of_alert": None,
            "time_of_event": None,
            "seconds_to_alert": None,
            "seconds_to_event": None,
            "duration": float(end_row["duration"]) if pd.notna(end_row["duration"]) else 0.0,
            "video_path": str(end_row["video_path"]),
            "split": "train",
            "source_peak_timestamp": float(source_peak["timestamp"]),
            "source_peak_risk_score": float(source_peak["risk_score"]),
            "source_peak_prob_class_2": (
                float(source_peak["prob_class_2"])
                if "prob_class_2" in source_peak and pd.notna(source_peak["prob_class_2"])
                else None
            ),
            "source_peak_prob_class_3": (
                float(source_peak["prob_class_3"])
                if "prob_class_3" in source_peak and pd.notna(source_peak["prob_class_3"])
                else None
            ),
        }
    )


def build_contextual_hard_negative_windows(
    scores_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    sequence_length: int,
    threshold: float,
    max_peaks_per_video: int,
    min_gap_seconds: float,
    neighbor_frame_offsets: list[int],
    max_windows_per_video: int,
    sample_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative_scores = scores_df[scores_df["target"].astype(int) == 0].copy()
    if negative_scores.empty:
        raise ValueError("No negative scores were found.")

    train_negative_ids = set(
        frame_df[
            (frame_df["split"] == "train") & (frame_df["video_target"].astype(int) == 0)
        ]["id"].astype(str)
    )
    negative_scores = negative_scores[
        negative_scores["id"].astype(str).isin(train_negative_ids)
    ].copy()

    frame_groups = {
        str(video_id): video_df.sort_values("timestamp").reset_index(drop=True)
        for video_id, video_df in frame_df.groupby("id", sort=False)
    }

    records: list[dict[str, object]] = []
    selected_peak_records = []

    for video_id, video_scores in negative_scores.groupby("id", sort=False):
        video_id = str(video_id)
        video_df = frame_groups.get(video_id)
        if video_df is None or video_df.empty:
            continue

        peaks_df = select_peak_timestamps(
            video_scores=video_scores,
            threshold=threshold,
            max_peaks=max_peaks_per_video,
            min_gap_seconds=min_gap_seconds,
        )
        if peaks_df.empty:
            continue

        used_offsets: set[int] = set()
        counter = 1
        timestamps = video_df["timestamp"].astype(float).to_numpy()

        for _, peak in peaks_df.iterrows():
            peak_timestamp = float(peak["timestamp"])
            nearest_offset = int(np.abs(timestamps - peak_timestamp).argmin())
            selected_peak_records.append(
                {
                    "id": video_id,
                    "peak_timestamp": peak_timestamp,
                    "peak_risk_score": float(peak["risk_score"]),
                    "peak_prob_class_2": (
                        float(peak["prob_class_2"])
                        if "prob_class_2" in peak and pd.notna(peak["prob_class_2"])
                        else None
                    ),
                    "peak_prob_class_3": (
                        float(peak["prob_class_3"])
                        if "prob_class_3" in peak and pd.notna(peak["prob_class_3"])
                        else None
                    ),
                }
            )

            for neighbor_offset in neighbor_frame_offsets:
                end_offset = nearest_offset + int(neighbor_offset)
                if end_offset < 0 or end_offset >= len(video_df):
                    continue
                if end_offset in used_offsets:
                    continue
                if len(used_offsets) >= max_windows_per_video:
                    break

                add_context_window_record(
                    records=records,
                    video_df=video_df,
                    end_offset=end_offset,
                    sequence_length=sequence_length,
                    sample_weight=sample_weight,
                    source_peak=peak,
                    counter=counter,
                )
                used_offsets.add(end_offset)
                counter += 1

            if len(used_offsets) >= max_windows_per_video:
                break

    mined_df = pd.DataFrame(records)
    selected_peaks_df = pd.DataFrame(selected_peak_records)
    if mined_df.empty:
        raise ValueError(
            "No contextual hard negatives were mined. Lower --threshold or "
            "increase --max-peaks-per-video."
        )

    return mined_df, selected_peaks_df


def augment_manifest(
    base_manifest: Path,
    mined_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    base_df = pd.read_csv(base_manifest, dtype={"id": str, "window_id": str})
    if "split" in base_df.columns and (base_df["split"] == "holdout").any():
        raise ValueError(
            "Base manifest contains holdout rows. Refusing to create augmented manifest."
        )

    overlap = set(base_df["window_id"].astype(str)) & set(mined_df["window_id"].astype(str))
    if overlap:
        raise ValueError(f"Mined window IDs overlap base manifest: {sorted(overlap)[:5]}")

    augmented_df = pd.concat([base_df, mined_df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    return augmented_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine contextual hard-negative windows from train negative videos."
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "interim"
        / "product_event_windows_seq8_prealert_phases_manifest.csv",
    )
    parser.add_argument(
        "--frame-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "interim"
        / "product_temporal_frames_224_manifest.csv",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "full_train_product_splits.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "checkpoints"
        / "product_event_window_phase_classifier_seq8_best_sequence.pt",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_context_hard_negative_train_scores.csv",
    )
    parser.add_argument(
        "--selected-peaks-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_context_hard_negative_selected_peaks.csv",
    )
    parser.add_argument(
        "--mined-windows-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_context_hard_negative_windows.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "interim"
        / "product_event_windows_seq8_context_hard_negatives_manifest.csv",
    )
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.64)
    parser.add_argument("--max-peaks-per-video", type=int, default=4)
    parser.add_argument("--min-gap-seconds", type=float, default=2.0)
    parser.add_argument("--neighbor-frame-offsets", default="-2,-1,0,1,2")
    parser.add_argument("--max-windows-per-video", type=int, default=12)
    parser.add_argument("--sample-weight", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Recompute train negative scores even if --scores-output exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    neighbor_frame_offsets = parse_int_offsets(args.neighbor_frame_offsets)
    frame_df = load_frame_manifest_with_split(
        frame_manifest=args.frame_manifest,
        split_manifest=args.split_manifest,
        split_column=args.split_column,
    )

    if args.scores_output.exists() and not args.force_rescore:
        scores_df = pd.read_csv(args.scores_output, dtype={"id": str})
        print(f"Loaded existing train negative scores from: {args.scores_output}")
    else:
        scores_df = score_train_negative_frames(
            frame_df=frame_df,
            checkpoint_path=args.checkpoint,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            output_path=args.scores_output,
        )

    mined_df, selected_peaks_df = build_contextual_hard_negative_windows(
        scores_df=scores_df,
        frame_df=frame_df,
        sequence_length=args.sequence_length,
        threshold=args.threshold,
        max_peaks_per_video=args.max_peaks_per_video,
        min_gap_seconds=args.min_gap_seconds,
        neighbor_frame_offsets=neighbor_frame_offsets,
        max_windows_per_video=args.max_windows_per_video,
        sample_weight=args.sample_weight,
    )
    args.mined_windows_output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_peaks_output.parent.mkdir(parents=True, exist_ok=True)
    mined_df.to_csv(args.mined_windows_output, index=False)
    selected_peaks_df.to_csv(args.selected_peaks_output, index=False)

    augmented_df = augment_manifest(
        base_manifest=args.base_manifest,
        mined_df=mined_df,
        output_path=args.output,
    )

    print(f"Saved train negative scores to: {args.scores_output}")
    print(f"Saved selected peaks to: {args.selected_peaks_output}")
    print(f"Saved mined windows to: {args.mined_windows_output}")
    print(f"Saved augmented manifest to: {args.output}")
    print(f"Selected peak videos: {selected_peaks_df['id'].nunique()}")
    print(f"Selected peaks: {len(selected_peaks_df)}")
    print(f"Mined contextual hard-negative windows: {len(mined_df)}")
    print("Augmented split distribution:")
    print(augmented_df["split"].value_counts().sort_index().to_string())
    print("Augmented phase distribution:")
    print(augmented_df["phase_index"].value_counts().sort_index().to_string())
    print("Augmented window type distribution:")
    print(augmented_df["window_type"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
