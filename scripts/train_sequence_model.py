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


def parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return None
    return tuple(int(token) for token in tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN + GRU/LSTM temporal model.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--cnn-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--cnn-train-policy",
        choices=["frozen", "layer4", "full"],
        default="frozen",
        help="CNN fine-tuning policy. Use layer4 for partial fine-tuning.",
    )
    parser.add_argument(
        "--fine-tune-cnn",
        action="store_true",
        help="Backward-compatible alias for --cnn-train-policy full.",
    )
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
        "--imbalance-strategy",
        choices=["none", "class_weight", "weighted_sampler"],
        default="class_weight",
    )
    parser.add_argument(
        "--loss-name",
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--sample-weight-column",
        default=None,
        help="Optional manifest column multiplied into the per-sample training loss.",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--negative-video-weight", type=float, default=1.0)
    parser.add_argument("--positive-safe-weight", type=float, default=1.0)
    parser.add_argument("--positive-event-weight", type=float, default=1.0)
    parser.add_argument(
        "--alert-metric-selection",
        action="store_true",
        help="Select best checkpoints using validation alert operating-point metrics.",
    )
    parser.add_argument("--alert-min-recall", type=float, default=0.80)
    parser.add_argument("--alert-max-false-alarm-rate", type=float, default=0.30)
    parser.add_argument("--alert-min-precision", type=float, default=0.70)
    parser.add_argument("--alert-threshold-start", type=float, default=0.10)
    parser.add_argument("--alert-threshold-stop", type=float, default=0.99)
    parser.add_argument("--alert-threshold-step", type=float, default=0.01)
    parser.add_argument("--alert-min-consecutive-frames", type=int, default=1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA mixed precision when a CUDA device is available.",
    )
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=0,
        help="Print running training loss every N batches. Use 0 to disable.",
    )
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes. Use 4 for the pre-alert phase classifier.",
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help="Manifest column used as training target. Use phase_index for phase training.",
    )
    parser.add_argument(
        "--alert-class-indices",
        default=None,
        help="Comma-separated class indices whose probabilities define alert risk.",
    )
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
    cnn_train_policy = "full" if args.fine_tune_cnn else args.cnn_train_policy

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
            num_classes=args.num_classes,
            target_column=args.target_column,
            sequence_length=args.sequence_length,
            train_sequence_stride=args.train_sequence_stride,
            val_sequence_stride=args.val_sequence_stride,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            head_learning_rate=args.head_learning_rate,
            cnn_learning_rate=args.cnn_learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            pretrained=args.pretrained,
            cnn_train_policy=cnn_train_policy,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            imbalance_strategy=args.imbalance_strategy,
            loss_name=args.loss_name,
            focal_gamma=args.focal_gamma,
            sample_weight_column=args.sample_weight_column,
            hard_negative_weight=args.hard_negative_weight,
            negative_video_weight=args.negative_video_weight,
            positive_safe_weight=args.positive_safe_weight,
            positive_event_weight=args.positive_event_weight,
            alert_metric_selection=args.alert_metric_selection,
            alert_min_recall=args.alert_min_recall,
            alert_max_false_alarm_rate=args.alert_max_false_alarm_rate,
            alert_min_precision=args.alert_min_precision,
            alert_threshold_start=args.alert_threshold_start,
            alert_threshold_stop=args.alert_threshold_stop,
            alert_threshold_step=args.alert_threshold_step,
            alert_min_consecutive_frames=args.alert_min_consecutive_frames,
            alert_class_indices=parse_int_list(args.alert_class_indices),
            amp=args.amp,
            log_every_n_batches=args.log_every_n_batches,
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
    print(f"Manifest type: {report['manifest_type']}")
    print(f"AMP enabled: {report['amp_enabled']}")
    print(f"Train sequences: {report['train_sequences']}")
    print(f"Validation sequences: {report['val_sequences']}")
    print(f"Parameter counts: {report['parameter_counts']}")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Best {report['best_monitor_metric']}: {report['best_monitor_value']}")
    print(f"Best checkpoint: {report['best_checkpoint_path']}")
    print(f"Frame metrics: {report['frame_metrics']}")
    if report.get("best_alert_metrics") is not None:
        print(f"Best alert metrics: {report['best_alert_metrics']}")


if __name__ == "__main__":
    main()
