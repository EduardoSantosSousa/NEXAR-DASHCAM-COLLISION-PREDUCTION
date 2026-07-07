from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

from nexar_collision.models.train import TrainingConfig, train_model
from nexar_collision.tracking.mlflow_utils import DEFAULT_EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ResNet18 frame baseline.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "sample_frames_manifest.csv",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional CSV with columns id and split.",
    )
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--monitor-metric", default="roc_auc")
    parser.add_argument("--monitor-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=None,
        help="Optional path for the best monitored checkpoint.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name
    report = train_model(
        TrainingConfig(
            manifest_path=args.manifest,
            split_manifest_path=args.split_manifest,
            split_column=args.split_column,
            train_split=args.train_split,
            val_split=args.val_split,
            predictions_path=PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{experiment_name}_frame_predictions.csv",
            metrics_path=PROJECT_ROOT
            / "models"
            / "reports"
            / f"{experiment_name}_metrics.json",
            checkpoint_path=PROJECT_ROOT
            / "models"
            / "checkpoints"
            / f"{experiment_name}_resnet18.pt",
            best_checkpoint_path=args.best_checkpoint
            or PROJECT_ROOT
            / "models"
            / "checkpoints"
            / f"{experiment_name}_best_resnet18.pt",
            figure_prefix=experiment_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            pretrained=args.pretrained,
            num_workers=args.num_workers,
            monitor_metric=args.monitor_metric,
            monitor_mode=args.monitor_mode,
            patience=args.patience,
            min_delta=args.min_delta,
            mlflow_enabled=not args.no_mlflow,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )
    )
    print("Training complete")
    print(f"Device: {report['device']}")
    print(f"Train videos: {report['train_videos']}")
    print(f"Validation videos: {report['val_videos']}")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Best {report['best_monitor_metric']}: {report['best_monitor_value']}")
    print(f"Best checkpoint: {report['best_checkpoint_path']}")
    print(f"Frame metrics: {report['frame_metrics']}")
    print(f"Video metrics mean: {report['video_metrics_mean']}")
    print(f"Video metrics max: {report['video_metrics_max']}")


if __name__ == "__main__":
    main()
