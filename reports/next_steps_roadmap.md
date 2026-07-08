# Next Steps Roadmap

This document records the recommended next steps after adding MLflow tracking and
running the corrected validation-only experiments.

The project is now in the experiment-management stage: the main priority is to
make each result comparable, reproducible, and useful for later article writing.

## Current State

Implemented:

- Initial dataset inspection.
- Stratified 100-video sample.
- Frame extraction.
- Frame-level ResNet18 baseline.
- Temporal label progression dataset.
- Fixed train/validation split by video.
- Validation-only temporal alert evaluation.
- Threshold sweeps.
- Experiment comparison report.
- Jupyter analysis notebook.
- MLflow tracking for training, alert evaluation, and threshold sweeps.

Important files:

```text
data/interim/sample_100_videos_splits.csv
data/interim/temporal_frames_224_manifest.csv
reports/experiment_comparison_val.md
reports/best_checkpoint_experiment.md
reports/temporal_aggregation_experiment.md
reports/cnn_rnn_sequence_experiment.md
reports/mlflow_tracking.md
notebooks/07_validation_experiment_analysis.ipynb
notebooks/08_best_checkpoint_mlflow_review.ipynb
mlflow.db
```

## Progress Update - 2026-07-07

Completed after the original roadmap:

- Best-checkpoint selection for training.
- Optional early stopping controlled by `--patience`.
- MLflow logging of `best_epoch`, best validation metrics, early-stopping state,
  final checkpoint, and best checkpoint.
- New best-checkpoint evaluations:
  - `temporal_alert_224_split_best`
  - `temporal_alert_224_pretrained_best`
- New review notebook:
  `notebooks/08_best_checkpoint_mlflow_review.ipynb`.
- Simple temporal aggregation threshold sweeps:
  - moving average 3s;
  - moving average 5s;
  - rolling max 3s;
  - 2 consecutive frames above threshold;
  - moving average 3s plus 2 consecutive frames.

Key result:

The pretrained best checkpoint reached recall `0.80` at threshold `0.23` on the
validation split, with precision `0.667`, false alarm rate `0.400`, and mean
alert error `-7.616 s`.

After temporal aggregation, the strongest operating point is now the pretrained
best checkpoint with a 2-consecutive-frame alert rule: threshold `0.13`, recall
`0.80`, precision `0.727`, false alarm rate `0.300`, and mean alert error
`-7.991 s`.

The next modeling priority is now imbalance handling.

Additional sequence-model progress:

- Implemented CNN + GRU/LSTM sequence training and evaluation.
- Ran first frozen-encoder sequence experiments:
  - `temporal_alert_224_pretrained_gru_seq4`
  - `temporal_alert_224_pretrained_lstm_seq4`
- The sequence pipeline works end to end, but does not yet beat the current best
  baseline.

## Immediate Next Step

Rerun the two main validation experiments with MLflow enabled, because some
previous results were generated before the full MLflow integration.

### Scratch ResNet18

Train:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_split --epochs 3 --batch-size 64 --learning-rate 0.0001 --num-workers 2
```

Evaluate:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_split_resnet18.pt --experiment-name temporal_alert_224_split --sample-csv data\interim\sample_100_videos_splits.csv --split val --fps 1 --threshold 0.5 --batch-size 32
```

Sweep thresholds:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_split --sample-csv data\interim\sample_100_videos_splits.csv --split val
```

### ImageNet-Pretrained ResNet18

Train:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_pretrained --pretrained --epochs 8 --batch-size 64 --learning-rate 0.00005 --num-workers 2
```

Evaluate:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_pretrained_resnet18.pt --experiment-name temporal_alert_224_pretrained --sample-csv data\interim\sample_100_videos_splits.csv --split val --fps 1 --threshold 0.5 --batch-size 32
```

Sweep thresholds:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained --sample-csv data\interim\sample_100_videos_splits.csv --split val
```

Open MLflow:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## Priority 1 - Best Checkpoint and Early Stopping

Problem:

The current training script saves the final epoch. This is not ideal because the
pretrained experiment had stronger intermediate validation ROC-AUC than the final
epoch.

Status:

Completed on 2026-07-07. See:

```text
reports/best_checkpoint_experiment.md
```

Goal:

- Save the best checkpoint according to validation ROC-AUC or validation F1.
- Log the best epoch and best metric to MLflow.
- Optionally stop training early when the metric does not improve.

Recommended implementation:

- Add `--monitor-metric`, default `roc_auc`.
- Add `--monitor-mode`, default `max`.
- Add `--patience`, default `3`.
- Save:

```text
models/checkpoints/<experiment_name>_best_resnet18.pt
```

Suggested MLflow metrics:

```text
best_epoch
best_val_roc_auc
best_val_f1
```

Decision criterion:

This was implemented before adding CNN + GRU/LSTM. A better checkpoint selection
policy improved the pretrained operating points without changing the architecture.

## Priority 2 - Better Imbalance Handling

Problem:

The temporal frame dataset is highly imbalanced:

```text
0: 7295 frames
1: 465 frames
```

Class-weighted cross entropy helps, but current recall is still weak.

Candidate experiments:

1. WeightedRandomSampler.
2. Focal loss.
3. Positive-frame oversampling.
4. Stronger data augmentation for positive frames.

Recommended experiment names:

```text
temporal_alert_224_pretrained_sampler
temporal_alert_224_pretrained_focal
temporal_alert_224_pretrained_oversample
```

Metrics to prioritize:

```text
alert_recall
false_alarm_rate
alert_precision
mean_alert_time_error
```

Decision criterion:

Keep any method that improves false alarm rate at matched recall, or improves
recall without causing false alarms to explode.

## Priority 3 - Short-Window Temporal Aggregation

Problem:

The current model classifies frames independently. This can produce noisy risk
curves and unstable alerts.

Status:

Completed on 2026-07-07. See:

```text
reports/temporal_aggregation_experiment.md
```

Before implementing GRU/LSTM, test simple temporal post-processing:

- moving average over 3 seconds;
- moving average over 5 seconds;
- rolling max over 3 seconds;
- alert only after N consecutive frames above threshold.

Suggested experiment names:

```text
temporal_alert_224_pretrained_ma3
temporal_alert_224_pretrained_ma5
temporal_alert_224_pretrained_consecutive2
```

Why this matters:

Simple temporal aggregation may reduce false positives and make alert timing more
stable without adding a new neural architecture.

Result:

The 2-consecutive-frame rule improved precision from `0.667` to `0.727` and
reduced false alarm rate from `0.400` to `0.300` while preserving recall `0.80`.

## Priority 4 - CNN + GRU/LSTM

Only start this after:

- best checkpoint selection is implemented;
- imbalance experiments are logged in MLflow;
- short-window temporal aggregation is tested.

Status:

Started on 2026-07-07. See:

```text
reports/cnn_rnn_sequence_experiment.md
```

The first frozen-encoder GRU/LSTM experiments are complete. They validate the
sequence pipeline but do not yet outperform the ResNet18 best-checkpoint model
with the 2-consecutive-frame rule.

Proposed architecture:

```text
frame sequence -> CNN encoder -> feature sequence -> GRU/LSTM -> alert score
```

Possible input design:

- 8 to 16 frames per sequence;
- 1 FPS or 2 FPS;
- positive windows centered around alert/event region;
- negative windows sampled from normal videos.

Important:

The split must remain video-level. Frames or windows from the same video must not
appear in both train and validation.

## Priority 5 - Notebook and Report Updates

Update:

```text
notebooks/07_validation_experiment_analysis.ipynb
reports/experiment_comparison_val.md
reports/methodology.md
reports/temporal_label_progression.md
```

Recommended notebook improvement:

- Load runs directly from MLflow.
- Compare all runs with a single table.
- Generate article-ready charts from MLflow metrics.

## Priority 6 - Streamlit Dashboard

Current dashboard is still a scaffold.

Recommended features:

- experiment selector;
- video selector;
- risk curve plot;
- threshold slider;
- true alert/event markers;
- predicted alert marker;
- alert metrics summary;
- model comparison table.

The dashboard should consume existing artifacts:

```text
outputs/predictions/*_temporal_risk_scores.csv
outputs/predictions/*_alert_predictions.csv
models/reports/*_alert_metrics.json
```

Later, it can read MLflow runs directly.

## Priority 7 - Paper Draft

Start turning the results into the article narrative.

Suggested structure:

1. Introduction.
2. Dataset and task definition.
3. Temporal alert framing.
4. Preprocessing and temporal labeling.
5. Baseline frame-level model.
6. Validation-only protocol.
7. Transfer learning comparison.
8. Threshold and alert-time analysis.
9. Limitations.
10. Future work with sequence models.

Core claim for now:

```text
Temporally meaningful labels and transfer learning improve calibration and false
alarm control, but frame-independent classification remains insufficient for
reliable early collision anticipation.
```

## Recommended Order of Work

1. Test imbalance handling for the sequence model.
2. Try GRU sequence length 8 with partial CNN fine-tuning.
3. Update notebook and reports from MLflow as new experiments land.
4. Build a useful Streamlit dashboard.
5. Start writing the paper draft.

## Resume Here Next Time

If continuing from a fresh day, start with:

```powershell
cd "C:\Users\z004hn4c\Documents\Estudo\LLMOps And AIOps Bootcamp With 8 End To End Projects\nexar-dashcam-collision-prediction"
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then rerun or inspect the current experiments in MLflow.
