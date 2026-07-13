from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


def validate_fraction(name: str, value: float) -> None:
    if value <= 0 or value >= 1:
        raise ValueError(f"{name} must be greater than 0 and less than 1.")


def assign_splits_for_class(
    class_df: pd.DataFrame,
    val_size: float,
    holdout_size: float,
    random_state: int,
) -> pd.DataFrame:
    shuffled_df = class_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    total = len(shuffled_df)
    holdout_count = round(total * holdout_size)
    val_count = round(total * val_size)

    if holdout_count <= 0 or val_count <= 0:
        raise ValueError(
            "Each target class must contribute at least one validation and "
            "holdout video. Lower the split fractions or use more data."
        )
    if holdout_count + val_count >= total:
        raise ValueError(
            "Validation and holdout splits are too large for the available data."
        )

    shuffled_df["split"] = "train"
    shuffled_df.loc[: holdout_count - 1, "split"] = "holdout"
    shuffled_df.loc[holdout_count : holdout_count + val_count - 1, "split"] = "val"
    return shuffled_df


def create_product_split(
    input_csv: Path,
    output_csv: Path,
    val_size: float,
    holdout_size: float,
    random_state: int,
) -> pd.DataFrame:
    validate_fraction("val_size", val_size)
    validate_fraction("holdout_size", holdout_size)
    if val_size + holdout_size >= 1:
        raise ValueError("val_size + holdout_size must be less than 1.")

    train_df = pd.read_csv(input_csv, dtype={"id": str})
    required_columns = {"id", "target", "time_of_event", "time_of_alert"}
    missing_columns = required_columns.difference(train_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {input_csv}: {sorted(missing_columns)}"
        )

    duplicated_ids = train_df["id"].duplicated().sum()
    if duplicated_ids:
        raise ValueError(f"Found duplicated video ids: {duplicated_ids}")

    split_parts = []
    for target, class_df in train_df.groupby("target", sort=True):
        split_parts.append(
            assign_splits_for_class(
                class_df=class_df.copy(),
                val_size=val_size,
                holdout_size=holdout_size,
                random_state=random_state + int(target),
            )
        )

    split_df = (
        pd.concat(split_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    split_df["video_path"] = split_df["id"].apply(
        lambda video_id: str(RAW_DATA_DIR / "train" / f"{video_id}.mp4")
    )

    missing_video_paths = [
        path for path in split_df["video_path"].map(Path) if not path.exists()
    ]
    if missing_video_paths:
        examples = [str(path) for path in missing_video_paths[:10]]
        raise ValueError(f"Missing video files. Examples: {examples}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)
    return split_df


def print_summary(split_df: pd.DataFrame) -> None:
    summary = (
        split_df.groupby(["split", "target"])
        .size()
        .rename("videos")
        .reset_index()
        .sort_values(["split", "target"])
    )
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic train/val/holdout split for product modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR / "train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INTERIM_DATA_DIR / "full_train_product_splits.csv",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--holdout-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_df = create_product_split(
        input_csv=args.input,
        output_csv=args.output,
        val_size=args.val_size,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
    )
    print(f"Saved product split to: {args.output}")
    print_summary(split_df)


if __name__ == "__main__":
    main()
