from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

from nexar_collision.models.train_sequence import SequenceTrainingConfig, train_sequence_model
from nexar_collision.tracking.mlflow_utils import DEFAULT_EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN + GRU/LSTM temporal model.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--fine-tune-cnn", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--train-sequence-stride", type=int, default=1)
    parser.add_argument("--val-sequence-stride", type=int, default=1)
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "temporal_frames_224_manifest.csv",
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
    parser.add_argument("--experiment-name", default="cnn_gru_sequence")
    parser.add_argument("--monitor-metric", default="roc_auc")
    parser.add_argument("--monitor-mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name
    report = train_sequence_model(
        SequenceTrainingConfig(
            manifest_path=args.manifest,
            split_manifest_path=args.split_manifest,
            split_column=args.split_column,
            train_split=args.train_split,
            val_split=args.val_split,
            predictions_path=PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{experiment_name}_sequence_predictions.csv",
            metrics_path=PROJECT_ROOT
            / "models"
            / "reports"
            / f"{experiment_name}_sequence_metrics.json",
            checkpoint_path=PROJECT_ROOT
            / "models"
            / "checkpoints"
            / f"{experiment_name}_sequence.pt",
            best_checkpoint_path=PROJECT_ROOT
            / "models"
            / "checkpoints"
            / f"{experiment_name}_best_sequence.pt",
            figure_prefix=experiment_name,
            sequence_length=args.sequence_length,
            train_sequence_stride=args.train_sequence_stride,
            val_sequence_stride=args.val_sequence_stride,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            pretrained=args.pretrained,
            freeze_cnn=not args.fine_tune_cnn,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
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
    print("Sequence training complete")
    print(f"Device: {report['device']}")
    print(f"Train sequences: {report['train_sequences']}")
    print(f"Validation sequences: {report['val_sequences']}")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Best {report['best_monitor_metric']}: {report['best_monitor_value']}")
    print(f"Best checkpoint: {report['best_checkpoint_path']}")
    print(f"Frame metrics: {report['frame_metrics']}")


if __name__ == "__main__":
    main()
