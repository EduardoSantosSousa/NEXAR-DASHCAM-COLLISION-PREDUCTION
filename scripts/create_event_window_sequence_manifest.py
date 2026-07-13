from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def optional_float(value) -> float | None:
    return float(value) if pd.notna(value) else None


def parse_allowed_splits(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def timestamp_token(timestamp: float) -> str:
    return f"{timestamp:07.3f}".replace(".", "_")


def limited_evenly_spaced(indices: list[int], max_count: int) -> list[int]:
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=max_count, dtype=int)
    return [indices[int(position)] for position in positions]


def sampled_indices(indices: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return indices
    selected = rng.choice(indices, size=count, replace=False)
    return sorted(int(index) for index in selected)


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


def add_window_record(
    records: list[dict[str, object]],
    video_df: pd.DataFrame,
    end_offset: int,
    sequence_length: int,
    target: int,
    window_type: str,
    counter: int,
) -> None:
    end_row = video_df.iloc[end_offset]
    frame_paths, timestamps = window_payload(video_df, end_offset, sequence_length)
    time_of_alert = optional_float(end_row.get("time_of_alert"))
    time_of_event = optional_float(end_row.get("time_of_event"))
    window_end = float(end_row["timestamp"])
    window_start = float(timestamps[0])
    center_timestamp = float((window_start + window_end) / 2.0)
    video_id = str(end_row["id"])
    token = timestamp_token(window_end)

    records.append(
        {
            "window_id": f"{video_id}_{window_type}_{counter:03d}_{token}",
            "id": video_id,
            "video_target": int(end_row["video_target"]),
            "target": int(target),
            "temporal_target": int(target),
            "window_type": window_type,
            "timestamp": window_end,
            "window_start_timestamp": window_start,
            "window_end_timestamp": window_end,
            "center_timestamp": center_timestamp,
            "sequence_length": int(sequence_length),
            "frame_count": len(frame_paths),
            "frame_paths": json.dumps(frame_paths),
            "timestamps": json.dumps(timestamps),
            "time_of_alert": time_of_alert,
            "time_of_event": time_of_event,
            "seconds_to_alert": (
                time_of_alert - window_end if time_of_alert is not None else None
            ),
            "seconds_to_event": (
                time_of_event - window_end if time_of_event is not None else None
            ),
            "duration": float(end_row["duration"]) if pd.notna(end_row["duration"]) else 0.0,
            "video_path": str(end_row["video_path"]),
            "split": end_row["split"] if "split" in end_row and pd.notna(end_row["split"]) else None,
        }
    )


def positive_end_offsets(
    video_df: pd.DataFrame,
    pre_alert_margin: float,
) -> list[int]:
    first_row = video_df.iloc[0]
    time_of_alert = optional_float(first_row.get("time_of_alert"))
    time_of_event = optional_float(first_row.get("time_of_event"))
    if time_of_alert is None or time_of_event is None:
        return []

    alert_start = max(0.0, time_of_alert - pre_alert_margin)
    timestamps = video_df["timestamp"].astype(float).to_numpy()
    return [
        int(index)
        for index, timestamp in enumerate(timestamps)
        if alert_start <= float(timestamp) <= time_of_event
    ]


def positive_safe_end_offsets(
    video_df: pd.DataFrame,
    pre_alert_margin: float,
    safe_gap_seconds: float,
) -> list[int]:
    first_row = video_df.iloc[0]
    time_of_alert = optional_float(first_row.get("time_of_alert"))
    if time_of_alert is None:
        return []

    alert_start = max(0.0, time_of_alert - pre_alert_margin)
    safe_end = max(0.0, alert_start - safe_gap_seconds)
    timestamps = video_df["timestamp"].astype(float).to_numpy()
    safe_offsets = [
        int(index)
        for index, timestamp in enumerate(timestamps)
        if float(timestamp) < safe_end
    ]
    if safe_offsets:
        return safe_offsets

    return [
        int(index)
        for index, timestamp in enumerate(timestamps)
        if float(timestamp) < alert_start
    ]


def create_event_window_manifest(
    frame_manifest: Path,
    output: Path,
    split_manifest: Path | None,
    split_column: str,
    allowed_splits: set[str],
    sequence_length: int,
    pre_alert_margin: float,
    safe_gap_seconds: float,
    positive_window_stride: int,
    max_positive_windows_per_video: int,
    negative_windows_per_video: int,
    positive_safe_windows_per_video: int,
    random_state: int,
    max_videos: int | None,
) -> pd.DataFrame:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")
    if positive_window_stride <= 0:
        raise ValueError("positive_window_stride must be greater than 0.")

    manifest_df = pd.read_csv(frame_manifest, dtype={"id": str})
    if split_manifest is not None:
        if split_column in manifest_df.columns:
            manifest_df = manifest_df.drop(columns=[split_column])
        split_df = pd.read_csv(split_manifest, dtype={"id": str})
        required_split_columns = {"id", split_column}
        missing_columns = required_split_columns - set(split_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required split columns: {sorted(missing_columns)}"
            )
        manifest_df = manifest_df.merge(
            split_df[["id", split_column]],
            on="id",
            how="left",
            validate="many_to_one",
        )
        if split_column != "split":
            manifest_df = manifest_df.rename(columns={split_column: "split"})

    if allowed_splits and "split" not in manifest_df.columns:
        raise ValueError(
            "allowed_splits was provided, but the manifest has no split column. "
            "Pass --split-manifest or use --allowed-splits \"\" intentionally."
        )

    if allowed_splits:
        manifest_df = manifest_df[manifest_df["split"].isin(allowed_splits)].copy()

    if max_videos is not None:
        selected_ids = manifest_df["id"].drop_duplicates().head(max_videos)
        manifest_df = manifest_df[manifest_df["id"].isin(selected_ids)].copy()

    if manifest_df.empty:
        raise ValueError("No frames available after filtering.")

    rng = np.random.default_rng(random_state)
    records: list[dict[str, object]] = []

    grouped = manifest_df.sort_values(["id", "timestamp"]).groupby("id", sort=False)
    for video_index, (video_id, video_df) in enumerate(grouped, start=1):
        video_df = video_df.reset_index(drop=True)
        video_target = int(video_df.iloc[0]["video_target"])
        print(
            f"building_windows={video_index}/{grouped.ngroups} "
            f"id={video_id} target={video_target}"
        )

        if video_target == 1:
            positive_offsets = positive_end_offsets(video_df, pre_alert_margin)
            positive_offsets = positive_offsets[::positive_window_stride]
            positive_offsets = limited_evenly_spaced(
                positive_offsets,
                max_positive_windows_per_video,
            )
            for counter, end_offset in enumerate(positive_offsets, start=1):
                add_window_record(
                    records,
                    video_df,
                    end_offset,
                    sequence_length,
                    target=1,
                    window_type="positive_event",
                    counter=counter,
                )

            safe_offsets = positive_safe_end_offsets(
                video_df,
                pre_alert_margin=pre_alert_margin,
                safe_gap_seconds=safe_gap_seconds,
            )
            safe_offsets = sampled_indices(
                safe_offsets,
                positive_safe_windows_per_video,
                rng,
            )
            for counter, end_offset in enumerate(safe_offsets, start=1):
                add_window_record(
                    records,
                    video_df,
                    end_offset,
                    sequence_length,
                    target=0,
                    window_type="positive_safe",
                    counter=counter,
                )
        else:
            offsets = list(range(len(video_df)))
            negative_offsets = sampled_indices(offsets, negative_windows_per_video, rng)
            for counter, end_offset in enumerate(negative_offsets, start=1):
                add_window_record(
                    records,
                    video_df,
                    end_offset,
                    sequence_length,
                    target=0,
                    window_type="negative_video",
                    counter=counter,
                )

    window_df = pd.DataFrame(records)
    if window_df.empty:
        raise ValueError("No event-window records were generated.")

    output.parent.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(output, index=False)
    return window_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an event-centered explicit sequence-window manifest."
    )
    parser.add_argument(
        "--frame-manifest",
        type=Path,
        default=INTERIM_DATA_DIR / "product_temporal_frames_224_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERIM_DATA_DIR / "product_event_windows_seq8_manifest.csv",
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split-column", default="split")
    parser.add_argument(
        "--allowed-splits",
        default="train,val",
        help="Comma-separated split names to include. Use an empty string for all.",
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--pre-alert-margin", type=float, default=3.0)
    parser.add_argument("--safe-gap-seconds", type=float, default=2.0)
    parser.add_argument("--positive-window-stride", type=int, default=1)
    parser.add_argument("--max-positive-windows-per-video", type=int, default=8)
    parser.add_argument("--negative-windows-per-video", type=int, default=6)
    parser.add_argument("--positive-safe-windows-per-video", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_df = create_event_window_manifest(
        frame_manifest=args.frame_manifest,
        output=args.output,
        split_manifest=args.split_manifest,
        split_column=args.split_column,
        allowed_splits=parse_allowed_splits(args.allowed_splits),
        sequence_length=args.sequence_length,
        pre_alert_margin=args.pre_alert_margin,
        safe_gap_seconds=args.safe_gap_seconds,
        positive_window_stride=args.positive_window_stride,
        max_positive_windows_per_video=args.max_positive_windows_per_video,
        negative_windows_per_video=args.negative_windows_per_video,
        positive_safe_windows_per_video=args.positive_safe_windows_per_video,
        random_state=args.random_state,
        max_videos=args.max_videos,
    )

    print(f"Saved event-window manifest to: {args.output}")
    print(f"Windows: {len(window_df)}")
    print("Target distribution:")
    print(window_df["target"].value_counts().sort_index().to_string())
    print("Window type distribution:")
    print(window_df["window_type"].value_counts().sort_index().to_string())
    if "split" in window_df.columns:
        print("Split distribution:")
        print(window_df["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
