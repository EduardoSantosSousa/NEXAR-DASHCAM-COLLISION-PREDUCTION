# Product event-window phase classifier seq8 next step

Run date: 2026-07-13

## Objective

Train a phase-aware temporal model instead of compressing the pre-alert phases
into a binary target.

Experiment name:

```text
product_event_window_phase_classifier_seq8
```

The model should predict four temporal phases:

| Class | Phase | Product meaning |
| ---: | --- | --- |
| 0 | `negative_video` / `positive_safe` | no alert |
| 1 | `prealert_early` | transition, still not alertable |
| 2 | `prealert_late` | alertable pre-collision region |
| 3 | `event_near` | near-collision region |

The product alert score should be:

```text
P(prealert_late) + P(event_near)
```

In class-index terms:

```text
prob_class_2 + prob_class_3
```

## Implementation Prepared

The sequence training pipeline now supports:

- `--num-classes`;
- `--target-column`;
- `--alert-class-indices`.

This keeps all previous binary experiments compatible while enabling the phase
classifier.

Updated files:

```text
scripts/train_sequence_model.py
scripts/evaluate_sequence_model.py
src/nexar_collision/models/train_sequence.py
src/nexar_collision/evaluation/evaluate_sequence.py
src/nexar_collision/evaluation/metrics.py
```

The full-video evaluator also supports phase models. It writes
`prob_class_0`, `prob_class_1`, `prob_class_2`, and `prob_class_3` to the risk
score CSV, while `risk_score` stores the product alert score.

## Input Manifest

Use the existing phase-aware manifest:

```text
data/interim/product_event_windows_seq8_prealert_phases_manifest.csv
```

Current phase distribution:

| Phase index | Windows |
| ---: | ---: |
| 0 | 11444 |
| 1 | 2552 |
| 2 | 486 |
| 3 | 1577 |

The holdout remains sealed. This manifest contains only `train` and `val`.

## Training Command

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\product_event_windows_seq8_prealert_phases_manifest.csv `
  --experiment-name product_event_window_phase_classifier_seq8 `
  --pretrained `
  --epochs 8 `
  --batch-size 16 `
  --learning-rate 0.0001 `
  --num-workers 2 `
  --sequence-length 8 `
  --rnn-type gru `
  --hidden-size 128 `
  --num-classes 4 `
  --target-column phase_index `
  --alert-class-indices 2,3 `
  --imbalance-strategy class_weight `
  --loss-name cross_entropy `
  --sample-weight-column sample_weight `
  --alert-metric-selection `
  --alert-min-recall 0.80 `
  --alert-max-false-alarm-rate 0.30 `
  --alert-min-precision 0.70 `
  --alert-threshold-start 0.25 `
  --alert-threshold-stop 0.85 `
  --alert-threshold-step 0.01 `
  --alert-min-consecutive-frames 2 `
  --monitor-metric alert_selection_score `
  --monitor-mode max `
  --patience 2 `
  --amp `
  --log-every-n-batches 100
```

Expected checkpoint:

```text
models/checkpoints/product_event_window_phase_classifier_seq8_best_sequence.pt
```

Expected training report:

```text
models/reports/product_event_window_phase_classifier_seq8_sequence_metrics.json
```

## Full-Video Validation Evaluation

After training, evaluate on full validation videos:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_sequence_model.py `
  --manifest data\interim\product_temporal_frames_224_manifest.csv `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split-manifest data\interim\full_train_product_splits.csv `
  --checkpoint models\checkpoints\product_event_window_phase_classifier_seq8_best_sequence.pt `
  --experiment-name product_event_window_phase_classifier_seq8 `
  --split val `
  --sequence-length 8 `
  --alert-class-indices 2,3 `
  --threshold 0.5 `
  --batch-size 16
```

Expected risk-score output:

```text
outputs/predictions/product_event_window_phase_classifier_seq8_temporal_risk_scores.csv
```

## Threshold Sweeps

Raw scores:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py `
  --experiment-name product_event_window_phase_classifier_seq8 `
  --risk-scores outputs\predictions\product_event_window_phase_classifier_seq8_temporal_risk_scores.csv `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split val `
  --output models\reports\product_event_window_phase_classifier_seq8_threshold_sweep_raw.csv
```

Two consecutive frames:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py `
  --experiment-name product_event_window_phase_classifier_seq8_consecutive2 `
  --risk-scores outputs\predictions\product_event_window_phase_classifier_seq8_temporal_risk_scores.csv `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split val `
  --min-consecutive-frames 2 `
  --output models\reports\product_event_window_phase_classifier_seq8_threshold_sweep_consecutive2.csv
```

Moving average 3s plus two consecutive frames:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py `
  --experiment-name product_event_window_phase_classifier_seq8_ma3s_consecutive2 `
  --risk-scores outputs\predictions\product_event_window_phase_classifier_seq8_temporal_risk_scores.csv `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split val `
  --aggregation moving_average `
  --aggregation-window-seconds 3 `
  --min-consecutive-frames 2 `
  --output models\reports\product_event_window_phase_classifier_seq8_threshold_sweep_ma3s_consecutive2.csv
```

## Product Gate

Only consider the model a holdout candidate if validation satisfies:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Quick PowerShell check:

```powershell
Import-Csv models\reports\product_event_window_phase_classifier_seq8_threshold_sweep_consecutive2.csv |
  Where-Object {
    [double]$_.alert_recall -ge 0.80 -and
    [double]$_.false_alarm_rate -le 0.30 -and
    [double]$_.alert_precision -ge 0.70
  }
```

If this returns no rows, do not run holdout.

## Article Use

This experiment is useful for the article even if it fails, because it tests a
clear hypothesis:

```text
Explicit temporal phase supervision should produce a better calibrated alert
curve than binary event-window supervision.
```

Record after running:

- best phase-classification metrics;
- validation alert precision, recall, and false alarm rate;
- best low-false-alarm operating point;
- whether the validation product gate passed;
- decision about holdout.
