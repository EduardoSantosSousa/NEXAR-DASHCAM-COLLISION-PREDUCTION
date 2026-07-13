from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

from nexar_collision.data.dataset import ExplicitFrameWindowSequenceDataset
from nexar_collision.evaluation.evaluate_sequence import load_sequence_checkpoint
from nexar_collision.models.train import build_transforms, resolve_device


def score_negative_windows(
    manifest_df: pd.DataFrame,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str,
    num_workers: int,
) -> pd.DataFrame:
    negative_df = manifest_df[
        (manifest_df["split"] == "train") & (manifest_df["target"] == 0)
    ].copy()
    if negative_df.empty:
        raise ValueError("No train negative windows available for hard-negative mining.")

    device = resolve_device(device_name)
    model, _, _, alert_class_indices = load_sequence_checkpoint(checkpoint_path, device)
    _, eval_transform = build_transforms()
    dataset = ExplicitFrameWindowSequenceDataset(negative_df, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    records: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["images"].to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu()
            scores = probabilities[:, list(alert_class_indices)].sum(dim=1).numpy()
            for index, score in enumerate(scores):
                records.append(
                    {
                        "window_id": batch["window_id"][index],
                        "id": batch["video_id"][index],
                        "window_type": batch["window_type"][index],
                        "timestamp": float(batch["timestamp"][index]),
                        "score": float(score),
                    }
                )
            if batch_index % 50 == 0:
                print(f"scored_batches={batch_index}/{len(loader)}", flush=True)

    return pd.DataFrame(records)


def select_hard_negatives(
    scores_df: pd.DataFrame,
    min_score: float,
    max_per_video: int,
) -> pd.DataFrame:
    candidates = scores_df[scores_df["score"] >= min_score].copy()
    if candidates.empty:
        raise ValueError(
            "No hard negatives found. Lower --min-score or inspect score distribution."
        )

    candidates = candidates.sort_values(["id", "score"], ascending=[True, False])
    return candidates.groupby("id", as_index=False).head(max_per_video).copy()


def augment_manifest(
    manifest_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    selected_ids = set(selected_df["window_id"].astype(str))
    mined_df = manifest_df[
        manifest_df["window_id"].astype(str).isin(selected_ids)
    ].copy()
    if mined_df.empty:
        raise ValueError("Selected hard negatives were not found in the base manifest.")

    score_lookup = selected_df.set_index("window_id")["score"].to_dict()
    mined_df["source_window_id"] = mined_df["window_id"]
    mined_df["hard_negative_score"] = mined_df["window_id"].map(score_lookup)
    mined_df["window_id"] = mined_df["window_id"].astype(str) + "_hard_negative"
    mined_df["window_type"] = "hard_negative_mined"
    mined_df["target"] = 0
    mined_df["temporal_target"] = 0
    mined_df["split"] = "train"

    augmented_df = pd.concat([manifest_df, mined_df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    return augmented_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine high-scoring train negative windows and augment a sequence manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "product_event_windows_seq8_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "checkpoints"
        / "product_event_window_gru_seq8_resnet18_frozen_best_sequence.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "interim"
        / "product_event_windows_seq8_hard_negative_manifest.csv",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_event_window_gru_seq8_train_negative_window_scores.csv",
    )
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-hard-negatives-per-video", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_df = pd.read_csv(args.manifest, dtype={"id": str, "window_id": str})
    scores_df = score_negative_windows(
        manifest_df=manifest_df,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
        num_workers=args.num_workers,
    )
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(args.scores_output, index=False)

    selected_df = select_hard_negatives(
        scores_df=scores_df,
        min_score=args.min_score,
        max_per_video=args.max_hard_negatives_per_video,
    )
    augmented_df = augment_manifest(
        manifest_df=manifest_df,
        selected_df=selected_df,
        output_path=args.output,
    )

    print(f"Saved train negative scores to: {args.scores_output}")
    print(f"Saved augmented manifest to: {args.output}")
    print(f"Scored negative train windows: {len(scores_df)}")
    print(f"Selected hard negatives: {len(selected_df)}")
    print("Augmented split distribution:")
    print(augmented_df["split"].value_counts().sort_index().to_string())
    print("Augmented target distribution:")
    print(augmented_df["target"].value_counts().sort_index().to_string())
    print("Augmented window type distribution:")
    print(augmented_df["window_type"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
