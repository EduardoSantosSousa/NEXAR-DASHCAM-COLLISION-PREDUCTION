from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


@dataclass(frozen=True)
class FrameRequest:
    label: str
    timestamp: float


def video_metadata(video_path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0
    capture.release()
    return fps, duration, frame_count


def clamp_timestamp(timestamp: float, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return max(0.0, min(timestamp, max(duration - 0.05, 0.0)))


def frame_requests(row: pd.Series, duration: float) -> list[FrameRequest]:
    if int(row["target"]) == 1:
        event_time = float(row["time_of_event"])
        alert_time = float(row["time_of_alert"])
        requests = [
            FrameRequest("event_minus_5s", event_time - 5.0),
            FrameRequest("event_minus_3s", event_time - 3.0),
            FrameRequest("event_minus_1s", event_time - 1.0),
            FrameRequest("alert", alert_time),
            FrameRequest("event", event_time),
        ]
    else:
        requests = [
            FrameRequest("video_20pct", duration * 0.20),
            FrameRequest("video_40pct", duration * 0.40),
            FrameRequest("video_60pct", duration * 0.60),
            FrameRequest("video_80pct", duration * 0.80),
            FrameRequest("video_95pct", duration * 0.95),
        ]

    return [
        FrameRequest(request.label, clamp_timestamp(request.timestamp, duration))
        for request in requests
    ]


def extract_frame(video_path: Path, timestamp: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise ValueError(f"Could not read frame at {timestamp:.3f}s from {video_path}")

    return frame


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def make_timeline_figure(frame_paths: list[Path], output_path: Path) -> None:
    images = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        images.append(add_label(image, path.stem))

    if not images:
        return

    grid = np.hstack(images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


def make_comparison_figure(
    positive_frame: Path,
    negative_frame: Path,
    output_path: Path,
) -> None:
    images = []
    for label, path in [("positive_event", positive_frame), ("negative_mid_video", negative_frame)]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        images.append(add_label(image, label))

    if len(images) != 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.hstack(images))


def extract_sample_frames(
    sample_csv: Path,
    frames_dir: Path,
    figures_dir: Path,
    max_videos: int | None,
) -> pd.DataFrame:
    sample_df = pd.read_csv(sample_csv, dtype={"id": str})
    if max_videos is not None:
        sample_df = sample_df.head(max_videos).copy()

    records: list[dict[str, object]] = []
    example_positive_paths: list[Path] = []
    example_negative_paths: list[Path] = []

    for _, row in sample_df.iterrows():
        video_id = row["id"]
        target = int(row["target"])
        video_path = Path(row["video_path"])
        fps, duration, frame_count = video_metadata(video_path)
        video_frame_dir = frames_dir / video_id
        video_frame_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        for request in frame_requests(row, duration):
            frame = extract_frame(video_path, request.timestamp)
            frame_path = video_frame_dir / f"{request.label}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            records.append(
                {
                    "id": video_id,
                    "target": target,
                    "frame_label": request.label,
                    "timestamp": round(request.timestamp, 3),
                    "frame_path": str(frame_path),
                    "fps": round(fps, 3),
                    "duration": round(duration, 3),
                    "frame_count": frame_count,
                }
            )

        if target == 1 and not example_positive_paths:
            example_positive_paths = frame_paths
            make_timeline_figure(
                frame_paths,
                figures_dir / f"positive_timeline_{video_id}.png",
            )
        if target == 0 and not example_negative_paths:
            example_negative_paths = frame_paths
            make_timeline_figure(
                frame_paths,
                figures_dir / f"negative_timeline_{video_id}.png",
            )

    if example_positive_paths and example_negative_paths:
        make_comparison_figure(
            example_positive_paths[-1],
            example_negative_paths[2],
            figures_dir / "positive_vs_negative_example.png",
        )

    manifest_df = pd.DataFrame(records)
    manifest_path = frames_dir.parent / "sample_frames_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract representative frames from a sampled Nexar dataset."
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=INTERIM_DATA_DIR / "sample_100_videos.csv",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=INTERIM_DATA_DIR / "frames_sample",
    )
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = extract_sample_frames(
        sample_csv=args.sample_csv,
        frames_dir=args.frames_dir,
        figures_dir=args.figures_dir,
        max_videos=args.max_videos,
    )

    print(f"Extracted frames: {len(manifest_df)}")
    print(f"Saved manifest to: {args.frames_dir.parent / 'sample_frames_manifest.csv'}")
    print(f"Saved figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()
