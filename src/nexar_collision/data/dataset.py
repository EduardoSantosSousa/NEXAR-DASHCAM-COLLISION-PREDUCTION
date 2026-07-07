"""Dataset definitions for video and frame-level training data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
