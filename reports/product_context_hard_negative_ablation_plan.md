# Product context hard-negative ablation plan

Run date: 2026-07-13

## Objective

Reduce the hard-negative pressure introduced by
`product_context_hard_negatives_phase_classifier_seq8`.

The previous run added 2202 contextual hard negatives with weight `2.5`. It
improved conservative operating points but still missed too many positive
events. The ablation tests whether lighter hard-negative pressure can preserve
false-alarm reduction while recovering recall.

## Implementation

Created:

```text
scripts/create_hard_negative_ablation_manifests.py
```

Input mined windows:

```text
models/reports/product_context_hard_negative_windows.csv
```

Base manifest:

```text
data/interim/product_event_windows_seq8_prealert_phases_manifest.csv
```

Summary output:

```text
models/reports/product_context_hard_negative_ablation_summary.csv
```

## Generated Variants

| Variant | Manifest | Hard negatives | Videos | Weight | Purpose |
| --- | --- | ---: | ---: | ---: | --- |
| A | `data/interim/product_event_windows_seq8_context_hn_all_w15_manifest.csv` | 2202 | 240 | 1.5 | keep full coverage, reduce weight |
| B | `data/interim/product_event_windows_seq8_context_hn_top1000_w20_manifest.csv` | 1000 | 181 | 2.0 | reduce volume, keep stronger penalty |
| C | `data/interim/product_event_windows_seq8_context_hn_top1000_w15_manifest.csv` | 1000 | 181 | 1.5 | reduce both volume and weight |

All variants contain:

```text
holdout_rows = 0
```

Split distribution:

| Variant | Train windows | Validation windows |
| --- | ---: | ---: |
| A | 15424 | 2837 |
| B | 14222 | 2837 |
| C | 14222 | 2837 |

Phase distribution:

| Variant | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
| --- | ---: | ---: | ---: | ---: |
| A | 13646 | 2552 | 486 | 1577 |
| B | 12444 | 2552 | 486 | 1577 |
| C | 12444 | 2552 | 486 | 1577 |

## Training Commands

### Variant A - all hard negatives, weight 1.5

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\product_event_windows_seq8_context_hn_all_w15_manifest.csv `
  --experiment-name product_context_hn_all_w15_phase_classifier_seq8 `
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

### Variant B - top 1000 hard negatives, weight 2.0

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\product_event_windows_seq8_context_hn_top1000_w20_manifest.csv `
  --experiment-name product_context_hn_top1000_w20_phase_classifier_seq8 `
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

### Variant C - top 1000 hard negatives, weight 1.5

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\product_event_windows_seq8_context_hn_top1000_w15_manifest.csv `
  --experiment-name product_context_hn_top1000_w15_phase_classifier_seq8 `
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

## Evaluation Rule

After each run:

1. Evaluate full validation videos with `scripts/evaluate_sequence_model.py`.
2. Sweep raw, 2 consecutive frames, and moving average 3s + 2 consecutive frames.
3. Compare with:
   - `product_event_window_phase_classifier_seq8`;
   - `product_context_hard_negatives_phase_classifier_seq8`.
4. Do not run holdout unless validation satisfies:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Recommended run order:

```text
C -> B -> A
```

Start with Variant C because it applies the lightest hard-negative pressure and
has the best chance of recovering recall.
