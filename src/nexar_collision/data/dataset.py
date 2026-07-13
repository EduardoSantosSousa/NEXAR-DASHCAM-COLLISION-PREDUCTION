"""Dataset definitions for video and frame-level training data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FrameCollisionDataset(Dataset):
    """Frame-level dataset backed by the extracted frame manifest."""

    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image = Image.open(Path(row["frame_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": int(row["target"]),
            "video_id": row["id"],
            "frame_label": row["frame_label"] if "frame_label" in row else f"t_{row['timestamp']}",
            "timestamp": float(row["timestamp"]),
        }


class FrameSequenceCollisionDataset(Dataset):
    """Causal frame-sequence dataset backed by a temporal frame manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        sequence_length: int,
        transform=None,
        sequence_stride: int = 1,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0.")
        if sequence_stride <= 0:
            raise ValueError("sequence_stride must be greater than 0.")

        self.manifest = (
            manifest.sort_values(["id", "timestamp"])
            .reset_index(drop=True)
            .copy()
        )
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequence_index = self._build_sequence_index(sequence_stride)

    def _build_sequence_index(self, sequence_stride: int) -> list[list[int]]:
        sequence_index: list[list[int]] = []
        for _, video_df in self.manifest.groupby("id", sort=False):
            positions = video_df.index.to_list()
            for end_offset in range(0, len(positions), sequence_stride):
                end_position = positions[end_offset]
                start_offset = max(0, end_offset - self.sequence_length + 1)
                window_positions = positions[start_offset : end_offset + 1]

                if len(window_positions) < self.sequence_length:
                    pad_count = self.sequence_length - len(window_positions)
                    window_positions = [window_positions[0]] * pad_count + window_positions

                sequence_index.append(window_positions)

        return sequence_index

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, index: int):
        sequence_positions = self.sequence_index[index]
        rows = self.manifest.iloc[sequence_positions]
        end_row = rows.iloc[-1]

        images = []
        for _, row in rows.iterrows():
            image = Image.open(Path(row["frame_path"])).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        video_target = int(end_row["video_target"]) if "video_target" in end_row else int(end_row["target"])
        duration = float(end_row["duration"]) if "duration" in end_row else 0.0
        sample_weight = (
            float(end_row["sample_weight"])
            if "sample_weight" in end_row and pd.notna(end_row["sample_weight"])
            else 1.0
        )

        return {
            "images": torch.stack(images),
            "target": int(end_row["target"]),
            "video_target": video_target,
            "video_id": end_row["id"],
            "timestamp": float(end_row["timestamp"]),
            "duration": duration,
            "sample_weight": sample_weight,
        }


class ExplicitFrameWindowSequenceDataset(Dataset):
    """Sequence dataset backed by one manifest row per explicit frame window."""

    def __init__(self, manifest: pd.DataFrame, transform=None):
        required_columns = {"id", "target", "frame_paths", "timestamp"}
        missing_columns = required_columns - set(manifest.columns)
        if missing_columns:
            raise ValueError(
                "Missing required columns for explicit sequence windows: "
                f"{sorted(missing_columns)}"
            )

        self.manifest = manifest.reset_index(drop=True).copy()
        self.transform = transform
        self.sequence_targets = [int(value) for value in self.manifest["target"]]

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _loads_list(value) -> list:
        if isinstance(value, list):
            return value
        return json.loads(str(value))

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        frame_paths = self._loads_list(row["frame_paths"])

        images = []
        for frame_path in frame_paths:
            image = Image.open(Path(frame_path)).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        video_target = (
            int(row["video_target"])
            if "video_target" in row and pd.notna(row["video_target"])
            else int(row["target"])
        )
        duration = (
            float(row["duration"])
            if "duration" in row and pd.notna(row["duration"])
            else 0.0
        )
        sample_weight = (
            float(row["sample_weight"])
            if "sample_weight" in row and pd.notna(row["sample_weight"])
            else 1.0
        )

        return {
            "images": torch.stack(images),
            "target": int(row["target"]),
            "video_target": video_target,
            "video_id": row["id"],
            "window_id": row["window_id"] if "window_id" in row else f"{row['id']}_{index}",
            "window_type": row["window_type"] if "window_type" in row else "explicit_window",
            "timestamp": float(row["timestamp"]),
            "duration": duration,
            "sample_weight": sample_weight,
        }
