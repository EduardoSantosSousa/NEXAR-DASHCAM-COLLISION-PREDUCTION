from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def create_stratified_sample(
    train_csv: Path,
    output_csv: Path,
    positives: int,
    negatives: int,
    random_state: int,
) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv, dtype={"id": str})

    positive_df = train_df[train_df["target"] == 1].sample(
        n=positives,
        random_state=random_state,
    )
    negative_df = train_df[train_df["target"] == 0].sample(
        n=negatives,
        random_state=random_state,
    )

    sample_df = (
        pd.concat([positive_df, negative_df], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    sample_df["video_path"] = sample_df["id"].apply(
        lambda video_id: str(RAW_DATA_DIR / "train" / f"{video_id}.mp4")
    )
    sample_df["split"] = "train_sample"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_csv, index=False)

    return sample_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small stratified sample for visual analysis."
    )
    parser.add_argument("--positives", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERIM_DATA_DIR / "sample_100_videos.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_df = create_stratified_sample(
        train_csv=RAW_DATA_DIR / "train.csv",
        output_csv=args.output,
        positives=args.positives,
        negatives=args.negatives,
        random_state=args.random_state,
    )

    print(f"Saved sample to: {args.output}")
    print(sample_df["target"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
