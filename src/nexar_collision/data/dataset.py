"""Dataset definitions for video and frame-level training data."""

from __future__ import annotations

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

        return {
            "images": torch.stack(images),
            "target": int(end_row["target"]),
            "video_target": video_target,
            "video_id": end_row["id"],
            "timestamp": float(end_row["timestamp"]),
            "duration": duration,
        }
