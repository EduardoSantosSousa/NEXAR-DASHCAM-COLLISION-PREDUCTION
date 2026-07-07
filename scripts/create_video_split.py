from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_video_split(
        input_csv: Path, 
        output_csv: Path, 
        val_size: float, 
        random_state: int, 
        split_column: str,
        ) -> pd.DataFrame:
             sample_df = pd.read_csv(input_csv, dtype={"id": str})
             if split_column in sample_df.columns:
                     sample_df = sample_df.drop(columns=[split_column])

             required_columns={"id", "target"}
             missing_columns = required_columns.difference(sample_df.columns)
             if missing_columns:
                     raise ValueError(
                             f"Missing required columns in {input_csv}: {sorted(missing_columns)}"
                     )

             video_df = sample_df[["id", "target"]].drop_duplicates("id").copy()

             duplicated_ids = video_df["id"].duplicated().sum()
             if duplicated_ids:
                     raise ValueError(f"Found duplicated video ids: {duplicated_ids}")

             val_video_ids = (
                     video_df.groupby("target", group_keys=False).sample(frac=val_size, random_state=random_state)["id"].tolist()
             )
             val_video_ids = set(val_video_ids)

             split_df = video_df.copy()
             split_df[split_column] = split_df["id"].apply(lambda video_id: "val" if video_id in val_video_ids else "train")

             output_df = sample_df.merge(split_df[["id", split_column]],
                                         on="id",
                                         how="left",
                                         validate="many_to_one")
             missing_split = output_df[split_column].isna().sum()
             if missing_split:
                     raise ValueError(f"Some rows did not receive a split: {missing_split}")
             
             output_csv.parent.mkdir(parents=True, exist_ok=True)
             output_df.to_csv(output_csv, index=False)
             return output_df


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description = "Create a deterministic train/val split by video id."
                )
        
        parser.add_argument(
                "--input",
                type=Path,
                default=PROJECT_ROOT / "data" / "interim" / "sample_100_videos.csv",
        )
        
        parser.add_argument(
                "--output",
                type=Path,
                default= PROJECT_ROOT / "data" / "interim" / "sample_100_videos_splits.csv",
        )

        parser.add_argument("--val-size", type=float, default=0.2)
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--split-column", default="split")
        return parser.parse_args()


def main () -> None:
        args = parse_args()
        split_df = create_video_split(
                input_csv=args.input,
                output_csv=args.output,
                val_size=args.val_size,
                random_state=args.random_state,
                split_column=args.split_column,
        )

        summary = (
                split_df[["id", "target", args.split_column]]
                .drop_duplicates("id")
                .groupby([args.split_column, "target"])
                .size()
                .rename("videos")
                .reset_index()
        )

        print(f"Saved split file to: {args.output}") 
        print(summary.to_string(index=False))

if __name__ == "__main__":
        main()               
