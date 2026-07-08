from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

from nexar_collision.evaluation.evaluate_sequence import (
    SequenceAlertEvaluationConfig,
    evaluate_sequence_alerts,
)
from nexar_collision.tracking.mlflow_utils import DEFAULT_EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal alerts from a CNN + GRU/LSTM sequence model."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "temporal_frames_224_manifest.csv",
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "sample_100_videos_splits.csv",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "sample_100_videos_splits.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "cnn_gru_sequence_best_sequence.pt",
    )
    parser.add_argument("--experiment-name", default="cnn_gru_sequence")
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_sequence_alerts(
        SequenceAlertEvaluationConfig(
            manifest_path=args.manifest,
            sample_csv=args.sample_csv,
            split_manifest_path=args.split_manifest,
            checkpoint_path=args.checkpoint,
            risk_scores_path=PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{args.experiment_name}_temporal_risk_scores.csv",
            alert_predictions_path=PROJECT_ROOT
            / "outputs"
            / "predictions"
            / f"{args.experiment_name}_alert_predictions.csv",
            metrics_path=PROJECT_ROOT
            / "models"
            / "reports"
            / f"{args.experiment_name}_alert_metrics.json",
            split=args.split,
            split_column=args.split_column,
            sequence_length=args.sequence_length,
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=args.device,
            mlflow_enabled=not args.no_mlflow,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        )
    )
    print("Sequence temporal alert evaluation complete")
    print(f"Device: {report['device']}")
    print(f"Sequence length: {report['sequence_length']}")
    print(f"Metrics: {report['metrics']}")


if __name__ == "__main__":
    main()
