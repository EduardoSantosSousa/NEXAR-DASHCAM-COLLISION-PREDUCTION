from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def video_duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return float(frame_count / fps) if fps else 0.0


def timestamps_for_video(duration: float, fps: float) -> list[float]:
    if duration <= 0:
        return [0.0]

    step = 1.0 / fps
    timestamps = []
    timestamp = 0.0
    while timestamp < duration:
        timestamps.append(round(timestamp, 3))
        timestamp += step
    return timestamps


def temporal_target(
    target: int,
    timestamp: float,
    time_of_alert: float | None,
    time_of_event: float | None,
    pre_alert_margin: float,
) -> int:
    if target == 0 or time_of_alert is None or time_of_event is None:
        return 0

    alert_start = max(0.0, time_of_alert - pre_alert_margin)
    return int(alert_start <= timestamp <= time_of_event)


def extract_frame(capture: cv2.VideoCapture, timestamp: float) -> tuple[bool, object]:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    return capture.read()


def create_temporal_frame_dataset(
    sample_csv: Path,
    output_dir: Path,
    manifest_path: Path,
    fps: float,
    pre_alert_margin: float,
    image_size: int,
    max_videos: int | None,
) -> pd.DataFrame:
    sample_df = pd.read_csv(sample_csv, dtype={"id": str})
    if max_videos is not None:
        sample_df = sample_df.head(max_videos).copy()

    records: list[dict[str, object]] = []

    for video_index, (_, row) in enumerate(sample_df.iterrows(), start=1):
        video_id = row["id"]
        target = int(row["target"])
        video_path = Path(row["video_path"])
        duration = video_duration(video_path)
        timestamps = timestamps_for_video(duration, fps)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        time_of_alert = float(row["time_of_alert"]) if pd.notna(row["time_of_alert"]) else None
        time_of_event = float(row["time_of_event"]) if pd.notna(row["time_of_event"]) else None
        video_output_dir = output_dir / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"extracting_video={video_index}/{len(sample_df)} id={video_id} target={target}")
        try:
            for timestamp in timestamps:
                ok, frame = extract_frame(capture, timestamp)
                if not ok or frame is None:
                    continue
                if image_size > 0:
                    frame = cv2.resize(
                        frame,
                        (image_size, image_size),
                        interpolation=cv2.INTER_AREA,
                    )

                frame_target = temporal_target(
                    target=target,
                    timestamp=timestamp,
                    time_of_alert=time_of_alert,
                    time_of_event=time_of_event,
                    pre_alert_margin=pre_alert_margin,
                )
                timestamp_token = f"{timestamp:07.3f}".replace(".", "_")
                frame_name = f"t_{timestamp_token}.jpg"
                frame_path = video_output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)

                records.append(
                    {
                        "id": video_id,
                        "video_target": target,
                        "target": frame_target,
                        "temporal_target": frame_target,
                        "timestamp": timestamp,
                        "time_of_alert": time_of_alert,
                        "time_of_event": time_of_event,
                        "seconds_to_alert": (
                            time_of_alert - timestamp if time_of_alert is not None else None
                        ),
                        "seconds_to_event": (
                            time_of_event - timestamp if time_of_event is not None else None
                        ),
                        "frame_path": str(frame_path),
                        "video_path": str(video_path),
                        "duration": round(duration, 3),
                    }
                )
        finally:
            capture.release()

    manifest_df = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a temporally labeled frame dataset for alert prediction."
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=INTERIM_DATA_DIR / "sample_100_videos.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INTERIM_DATA_DIR / "temporal_frames_sample",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=INTERIM_DATA_DIR / "temporal_frames_manifest.csv",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--pre-alert-margin", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = create_temporal_frame_dataset(
        sample_csv=args.sample_csv,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        fps=args.fps,
        pre_alert_margin=args.pre_alert_margin,
        image_size=args.image_size,
        max_videos=args.max_videos,
    )

    print(f"Saved temporal manifest to: {args.manifest}")
    print(f"Saved temporal frames to: {args.output_dir}")
    print(f"Frames: {len(manifest_df)}")
    print(manifest_df["target"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
