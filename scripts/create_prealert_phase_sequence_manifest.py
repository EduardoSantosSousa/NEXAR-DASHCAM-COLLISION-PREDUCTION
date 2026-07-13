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


def sampled_indices(indices: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return indices
    selected = rng.choice(indices, size=count, replace=False)
    return sorted(int(index) for index in selected)


def limited_evenly_spaced(indices: list[int], max_count: int) -> list[int]:
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=max_count, dtype=int)
    return [indices[int(position)] for position in positions]


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


def phase_offsets(
    video_df: pd.DataFrame,
    start_time: float,
    end_time: float,
    stride: int,
    max_count: int,
) -> list[int]:
    timestamps = video_df["timestamp"].astype(float).to_numpy()
    offsets = [
        int(index)
        for index, timestamp in enumerate(timestamps)
        if start_time <= float(timestamp) <= end_time
    ]
    offsets = offsets[::stride]
    return limited_evenly_spaced(offsets, max_count)


def add_window_record(
    records: list[dict[str, object]],
    video_df: pd.DataFrame,
    end_offset: int,
    sequence_length: int,
    target: int,
    window_type: str,
    phase_name: str,
    phase_index: int,
    sample_weight: float,
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
            "phase_name": phase_name,
            "phase_index": int(phase_index),
            "sample_weight": float(sample_weight),
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


def positive_safe_offsets(
    video_df: pd.DataFrame,
    safe_end_time: float,
) -> list[int]:
    timestamps = video_df["timestamp"].astype(float).to_numpy()
    return [
        int(index)
        for index, timestamp in enumerate(timestamps)
        if float(timestamp) < safe_end_time
    ]


def load_frame_manifest(
    frame_manifest: Path,
    split_manifest: Path | None,
    split_column: str,
    allowed_splits: set[str],
) -> pd.DataFrame:
    manifest_df = pd.read_csv(frame_manifest, dtype={"id": str})
    if split_manifest is not None:
        if split_column in manifest_df.columns:
            manifest_df = manifest_df.drop(columns=[split_column])
        split_df = pd.read_csv(split_manifest, dtype={"id": str})
        required_columns = {"id", split_column}
        missing_columns = required_columns - set(split_df.columns)
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

    if manifest_df.empty:
        raise ValueError("No frames available after filtering.")
    return manifest_df


def create_prealert_phase_manifest(
    frame_manifest: Path,
    output: Path,
    split_manifest: Path | None,
    split_column: str,
    allowed_splits: set[str],
    sequence_length: int,
    pre_alert_margin: float,
    safe_gap_seconds: float,
    event_near_seconds: float,
    positive_window_stride: int,
    max_early_prealert_windows_per_video: int,
    max_late_prealert_windows_per_video: int,
    max_event_near_windows_per_video: int,
    negative_windows_per_video: int,
    positive_safe_windows_per_video: int,
    negative_weight: float,
    positive_safe_weight: float,
    early_prealert_weight: float,
    late_prealert_weight: float,
    event_near_weight: float,
    random_state: int,
    max_videos: int | None,
) -> pd.DataFrame:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")
    if positive_window_stride <= 0:
        raise ValueError("positive_window_stride must be greater than 0.")
    if pre_alert_margin <= 0:
        raise ValueError("pre_alert_margin must be greater than 0.")
    if event_near_seconds <= 0:
        raise ValueError("event_near_seconds must be greater than 0.")

    manifest_df = load_frame_manifest(
        frame_manifest=frame_manifest,
        split_manifest=split_manifest,
        split_column=split_column,
        allowed_splits=allowed_splits,
    )
    if max_videos is not None:
        selected_ids = manifest_df["id"].drop_duplicates().head(max_videos)
        manifest_df = manifest_df[manifest_df["id"].isin(selected_ids)].copy()

    rng = np.random.default_rng(random_state)
    records: list[dict[str, object]] = []

    grouped = manifest_df.sort_values(["id", "timestamp"]).groupby("id", sort=False)
    for video_index, (video_id, video_df) in enumerate(grouped, start=1):
        video_df = video_df.reset_index(drop=True)
        video_target = int(video_df.iloc[0]["video_target"])
        print(
            f"building_phase_windows={video_index}/{grouped.ngroups} "
            f"id={video_id} target={video_target}",
            flush=True,
        )

        if video_target == 0:
            offsets = list(range(len(video_df)))
            negative_offsets = sampled_indices(offsets, negative_windows_per_video, rng)
            for counter, end_offset in enumerate(negative_offsets, start=1):
                add_window_record(
                    records=records,
                    video_df=video_df,
                    end_offset=end_offset,
                    sequence_length=sequence_length,
                    target=0,
                    window_type="negative_video",
                    phase_name="negative_video",
                    phase_index=0,
                    sample_weight=negative_weight,
                    counter=counter,
                )
            continue

        first_row = video_df.iloc[0]
        time_of_alert = optional_float(first_row.get("time_of_alert"))
        time_of_event = optional_float(first_row.get("time_of_event"))
        if time_of_alert is None or time_of_event is None:
            continue

        early_start = max(0.0, time_of_alert - pre_alert_margin)
        safe_end = max(0.0, early_start - safe_gap_seconds)
        event_near_start = max(time_of_alert, time_of_event - event_near_seconds)

        safe_offsets = sampled_indices(
            positive_safe_offsets(video_df, safe_end),
            positive_safe_windows_per_video,
            rng,
        )
        for counter, end_offset in enumerate(safe_offsets, start=1):
            add_window_record(
                records=records,
                video_df=video_df,
                end_offset=end_offset,
                sequence_length=sequence_length,
                target=0,
                window_type="positive_safe",
                phase_name="positive_safe",
                phase_index=0,
                sample_weight=positive_safe_weight,
                counter=counter,
            )

        early_offsets = phase_offsets(
            video_df=video_df,
            start_time=early_start,
            end_time=max(early_start, time_of_alert - 1e-6),
            stride=positive_window_stride,
            max_count=max_early_prealert_windows_per_video,
        )
        for counter, end_offset in enumerate(early_offsets, start=1):
            add_window_record(
                records=records,
                video_df=video_df,
                end_offset=end_offset,
                sequence_length=sequence_length,
                target=0,
                window_type="prealert_early",
                phase_name="prealert_early",
                phase_index=1,
                sample_weight=early_prealert_weight,
                counter=counter,
            )

        late_offsets = phase_offsets(
            video_df=video_df,
            start_time=time_of_alert,
            end_time=max(time_of_alert, event_near_start - 1e-6),
            stride=positive_window_stride,
            max_count=max_late_prealert_windows_per_video,
        )
        for counter, end_offset in enumerate(late_offsets, start=1):
            add_window_record(
                records=records,
                video_df=video_df,
                end_offset=end_offset,
                sequence_length=sequence_length,
                target=1,
                window_type="prealert_late",
                phase_name="prealert_late",
                phase_index=2,
                sample_weight=late_prealert_weight,
                counter=counter,
            )

        event_offsets = phase_offsets(
            video_df=video_df,
            start_time=event_near_start,
            end_time=time_of_event,
            stride=positive_window_stride,
            max_count=max_event_near_windows_per_video,
        )
        for counter, end_offset in enumerate(event_offsets, start=1):
            add_window_record(
                records=records,
                video_df=video_df,
                end_offset=end_offset,
                sequence_length=sequence_length,
                target=1,
                window_type="event_near",
                phase_name="event_near",
                phase_index=3,
                sample_weight=event_near_weight,
                counter=counter,
            )

    phase_df = pd.DataFrame(records)
    if phase_df.empty:
        raise ValueError("No pre-alert phase records were generated.")

    output.parent.mkdir(parents=True, exist_ok=True)
    phase_df.to_csv(output, index=False)
    return phase_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a phase-aware explicit sequence-window manifest."
    )
    parser.add_argument(
        "--frame-manifest",
        type=Path,
        default=INTERIM_DATA_DIR / "product_temporal_frames_224_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERIM_DATA_DIR / "product_event_windows_seq8_prealert_phases_manifest.csv",
    )
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--allowed-splits", default="train,val")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--pre-alert-margin", type=float, default=3.0)
    parser.add_argument("--safe-gap-seconds", type=float, default=2.0)
    parser.add_argument("--event-near-seconds", type=float, default=1.5)
    parser.add_argument("--positive-window-stride", type=int, default=1)
    parser.add_argument("--max-early-prealert-windows-per-video", type=int, default=4)
    parser.add_argument("--max-late-prealert-windows-per-video", type=int, default=4)
    parser.add_argument("--max-event-near-windows-per-video", type=int, default=4)
    parser.add_argument("--negative-windows-per-video", type=int, default=12)
    parser.add_argument("--positive-safe-windows-per-video", type=int, default=6)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--positive-safe-weight", type=float, default=1.5)
    parser.add_argument("--early-prealert-weight", type=float, default=0.6)
    parser.add_argument("--late-prealert-weight", type=float, default=1.0)
    parser.add_argument("--event-near-weight", type=float, default=1.4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_df = create_prealert_phase_manifest(
        frame_manifest=args.frame_manifest,
        output=args.output,
        split_manifest=args.split_manifest,
        split_column=args.split_column,
        allowed_splits=parse_allowed_splits(args.allowed_splits),
        sequence_length=args.sequence_length,
        pre_alert_margin=args.pre_alert_margin,
        safe_gap_seconds=args.safe_gap_seconds,
        event_near_seconds=args.event_near_seconds,
        positive_window_stride=args.positive_window_stride,
        max_early_prealert_windows_per_video=args.max_early_prealert_windows_per_video,
        max_late_prealert_windows_per_video=args.max_late_prealert_windows_per_video,
        max_event_near_windows_per_video=args.max_event_near_windows_per_video,
        negative_windows_per_video=args.negative_windows_per_video,
        positive_safe_windows_per_video=args.positive_safe_windows_per_video,
        negative_weight=args.negative_weight,
        positive_safe_weight=args.positive_safe_weight,
        early_prealert_weight=args.early_prealert_weight,
        late_prealert_weight=args.late_prealert_weight,
        event_near_weight=args.event_near_weight,
        random_state=args.random_state,
        max_videos=args.max_videos,
    )

    print(f"Saved pre-alert phase manifest to: {args.output}")
    print(f"Windows: {len(phase_df)}")
    print("Target distribution:")
    print(phase_df["target"].value_counts().sort_index().to_string())
    print("Window type distribution:")
    print(phase_df["window_type"].value_counts().sort_index().to_string())
    print("Sample weight by window type:")
    print(
        phase_df.groupby("window_type")["sample_weight"]
        .first()
        .sort_index()
        .to_string()
    )
    if "split" in phase_df.columns:
        print("Split distribution:")
        print(phase_df["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
