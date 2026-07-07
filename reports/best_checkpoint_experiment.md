# Best Checkpoint Experiment

Run date: 2026-07-07

## Objective

The previous validation experiments saved only the final training epoch. This was
not ideal because the strongest validation ROC-AUC appeared before the final
epoch. This experiment adds best-checkpoint selection, logs the selected epoch to
MLflow, and evaluates the best checkpoints as temporal alert systems.

## Implementation

Training now supports:

- `--monitor-metric`, default `roc_auc`.
- `--monitor-mode`, default `max`.
- `--patience`, default `3`.
- `--min-delta`, default `0.0`.
- `--best-checkpoint`, optional explicit output path.

Each training run saves:

- final checkpoint: `models/checkpoints/<experiment_name>_resnet18.pt`;
- best checkpoint: `models/checkpoints/<experiment_name>_best_resnet18.pt`;
- JSON metrics with `best_epoch`, `best_monitor_value`, `best_frame_metrics`,
  `early_stopped`, and `stopped_epoch`;
- MLflow metrics and checkpoint artifacts.

## Commands

Scratch training:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_split --epochs 3 --batch-size 64 --learning-rate 0.0001 --num-workers 2 --monitor-metric roc_auc --monitor-mode max --patience 3
```

Pretrained training:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_pretrained --pretrained --epochs 8 --batch-size 64 --learning-rate 0.00005 --num-workers 2 --monitor-metric roc_auc --monitor-mode max --patience 3
```

Best-checkpoint temporal evaluation:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_split_best_resnet18.pt --experiment-name temporal_alert_224_split_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --fps 1 --threshold 0.5 --batch-size 32
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_split_best --sample-csv data\interim\sample_100_videos_splits.csv --split val

.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_pretrained_best_resnet18.pt --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --fps 1 --threshold 0.5 --batch-size 32
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val
```

## Training Results

| Experiment | Best epoch | Best ROC-AUC | Best F1 | Final ROC-AUC | Final F1 | Early stopped |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `temporal_alert_224_split` | 2 | 0.514 | 0.137 | 0.502 | 0.093 | No |
| `temporal_alert_224_pretrained` | 3 | 0.599 | 0.058 | 0.558 | 0.078 | Yes, epoch 6 |

The pretrained model improved the monitored validation ROC-AUC when selecting the
best checkpoint. The selected best epoch was 3, and training stopped at epoch 6
after three epochs without ROC-AUC improvement.

## Alert Results at Threshold 0.50

| Experiment | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_split_best` | 0.533 | 0.800 | 0.700 | 0.200 | -14.366 s |
| `temporal_alert_224_pretrained_best` | 0.500 | 0.300 | 0.300 | 0.700 | -11.794 s |

The threshold `0.50` is not the best operating point for the pretrained best
checkpoint. Threshold sweeps remain necessary.

## Sweep Operating Points

| Experiment | Minimum recall | Threshold | Precision | Recall | False alarm rate | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_split_best` | 0.70 | 0.65 | 0.636 | 0.700 | 0.400 | -14.495 s |
| `temporal_alert_224_split_best` | 0.60 | 0.66 | 0.667 | 0.600 | 0.300 | -15.638 s |
| `temporal_alert_224_pretrained_best` | 0.70 | 0.23 | 0.667 | 0.800 | 0.400 | -7.616 s |
| `temporal_alert_224_pretrained_best` | 0.50 | 0.21 | 0.667 | 0.800 | 0.400 | -9.741 s |

## Interpretation

Best-checkpoint selection materially improves the pretrained experiment under
threshold tuning. The best pretrained checkpoint can now reach recall `0.80` on
the validation split with false alarm rate `0.40`, while keeping alert timing
less extreme than the scratch model.

The result is still not final because the validation set has only 20 videos and
the threshold is selected on that same validation split. However, the experiment
is a stronger and more defensible baseline before testing imbalance handling and
temporal aggregation.

## Temporal Aggregation Follow-Up

Simple temporal aggregation was tested after this best-checkpoint experiment.
The strongest follow-up result uses the pretrained best checkpoint with a
2-consecutive-frame alert rule:

| Post-processing | Threshold | Precision | Recall | False alarm rate | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 consecutive frames | 0.13 | 0.727 | 0.800 | 0.300 | -7.991 s |

This improves over the raw pretrained best operating point, which had precision
`0.667`, recall `0.800`, and false alarm rate `0.400`.

See:

```text
reports/temporal_aggregation_experiment.md
```

## Review Notebook

Use:

```text
notebooks/08_best_checkpoint_mlflow_review.ipynb
```

The notebook summarizes local metrics, threshold sweeps, risk curves, and MLflow
runs from `mlflow.db`.
