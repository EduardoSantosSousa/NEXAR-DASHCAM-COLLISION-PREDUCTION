from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationVariant:
    name: str
    max_windows: int | None
    sample_weight: float
    max_windows_per_video: int | None
    output_name: str


DEFAULT_VARIANTS = [
    AblationVariant(
        name="all_w15",
        max_windows=None,
        sample_weight=1.5,
        max_windows_per_video=None,
        output_name="product_event_windows_seq8_context_hn_all_w15_manifest.csv",
    ),
    AblationVariant(
        name="top1000_w20",
        max_windows=1000,
        sample_weight=2.0,
        max_windows_per_video=6,
        output_name="product_event_windows_seq8_context_hn_top1000_w20_manifest.csv",
    ),
    AblationVariant(
        name="top1000_w15",
        max_windows=1000,
        sample_weight=1.5,
        max_windows_per_video=6,
        output_name="product_event_windows_seq8_context_hn_top1000_w15_manifest.csv",
    ),
]


def load_base_manifest(path: Path) -> pd.DataFrame:
    base_df = pd.read_csv(path, dtype={"id": str, "window_id": str})
    if "split" in base_df.columns and (base_df["split"] == "holdout").any():
        raise ValueError("Base manifest contains holdout rows. Refusing to continue.")
    return base_df


def load_mined_windows(path: Path) -> pd.DataFrame:
    mined_df = pd.read_csv(path, dtype={"id": str, "window_id": str})
    if mined_df.empty:
        raise ValueError("Mined hard-negative window file is empty.")
    if "split" in mined_df.columns and (mined_df["split"] != "train").any():
        raise ValueError("Mined hard-negative windows must all belong to train split.")
    required_columns = {"source_peak_risk_score", "id", "window_id"}
    missing_columns = required_columns - set(mined_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required mined-window columns: {sorted(missing_columns)}")
    return mined_df


def select_variant_windows(
    mined_df: pd.DataFrame,
    variant: AblationVariant,
) -> pd.DataFrame:
    selected_df = mined_df.copy()
    selected_df["source_peak_risk_score"] = selected_df["source_peak_risk_score"].astype(float)
    if "source_peak_prob_class_3" in selected_df.columns:
        selected_df["source_peak_prob_class_3"] = selected_df[
            "source_peak_prob_class_3"
        ].astype(float)
    else:
        selected_df["source_peak_prob_class_3"] = 0.0

    selected_df = selected_df.sort_values(
        ["source_peak_risk_score", "source_peak_prob_class_3"],
        ascending=[False, False],
    )

    if variant.max_windows_per_video is not None:
        selected_df = (
            selected_df.groupby("id", as_index=False, group_keys=False)
            .head(variant.max_windows_per_video)
            .copy()
        )
        selected_df = selected_df.sort_values(
            ["source_peak_risk_score", "source_peak_prob_class_3"],
            ascending=[False, False],
        )

    if variant.max_windows is not None:
        selected_df = selected_df.head(variant.max_windows).copy()

    selected_df = selected_df.reset_index(drop=True)
    selected_df["ablation_variant"] = variant.name
    selected_df["source_window_id"] = selected_df["window_id"].astype(str)
    selected_df["window_id"] = (
        selected_df["window_id"].astype(str) + f"_{variant.name}"
    )
    selected_df["sample_weight"] = float(variant.sample_weight)
    selected_df["selection_rank"] = selected_df.index + 1
    selected_df["window_type"] = "contextual_hard_negative"
    selected_df["phase_name"] = "contextual_hard_negative"
    selected_df["phase_index"] = 0
    selected_df["target"] = 0
    selected_df["temporal_target"] = 0
    selected_df["split"] = "train"
    return selected_df


def build_variant_manifest(
    base_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    overlap = set(base_df["window_id"].astype(str)) & set(selected_df["window_id"].astype(str))
    if overlap:
        raise ValueError(f"Window IDs overlap base manifest: {sorted(overlap)[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.concat([base_df, selected_df], ignore_index=True)
    manifest_df.to_csv(output_path, index=False)
    return manifest_df


def summarize_manifest(
    variant: AblationVariant,
    selected_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, object]:
    split_counts = {
        str(key): int(value)
        for key, value in manifest_df["split"].value_counts().sort_index().items()
    }
    phase_counts = {
        str(key): int(value)
        for key, value in manifest_df["phase_index"].value_counts().sort_index().items()
    }
    return {
        "variant": variant.name,
        "manifest_path": str(output_path),
        "hard_negative_windows": int(len(selected_df)),
        "hard_negative_videos": int(selected_df["id"].nunique()),
        "hard_negative_sample_weight": float(variant.sample_weight),
        "max_windows_per_video": variant.max_windows_per_video,
        "max_windows_requested": variant.max_windows,
        "mean_source_peak_risk_score": float(selected_df["source_peak_risk_score"].mean()),
        "min_source_peak_risk_score": float(selected_df["source_peak_risk_score"].min()),
        "max_source_peak_risk_score": float(selected_df["source_peak_risk_score"].max()),
        "train_windows": split_counts.get("train", 0),
        "val_windows": split_counts.get("val", 0),
        "phase_0_windows": phase_counts.get("0", 0),
        "phase_1_windows": phase_counts.get("1", 0),
        "phase_2_windows": phase_counts.get("2", 0),
        "phase_3_windows": phase_counts.get("3", 0),
    }


def create_ablation_manifests(args: argparse.Namespace) -> pd.DataFrame:
    base_df = load_base_manifest(args.base_manifest)
    mined_df = load_mined_windows(args.mined_windows)

    summaries: list[dict[str, object]] = []
    for variant in DEFAULT_VARIANTS:
        selected_df = select_variant_windows(mined_df, variant)
        selected_output = args.report_dir / f"{args.experiment_prefix}_{variant.name}_windows.csv"
        selected_output.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(selected_output, index=False)

        output_path = args.output_dir / variant.output_name
        manifest_df = build_variant_manifest(base_df, selected_df, output_path)
        summary = summarize_manifest(
            variant=variant,
            selected_df=selected_df,
            manifest_df=manifest_df,
            output_path=output_path,
        )
        summary["selected_windows_path"] = str(selected_output)
        summaries.append(summary)

        print(f"Saved {variant.name} manifest to: {output_path}")
        print(f"  hard_negative_windows={len(selected_df)}")
        print(f"  hard_negative_videos={selected_df['id'].nunique()}")
        print(f"  sample_weight={variant.sample_weight}")

    summary_df = pd.DataFrame(summaries)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create hard-negative pressure ablation manifests."
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "interim"
        / "product_event_windows_seq8_prealert_phases_manifest.csv",
    )
    parser.add_argument(
        "--mined-windows",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_context_hard_negative_windows.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "reports",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "models"
        / "reports"
        / "product_context_hard_negative_ablation_summary.csv",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="product_context_hard_negative_ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = create_ablation_manifests(args)
    print("Ablation manifests complete")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
