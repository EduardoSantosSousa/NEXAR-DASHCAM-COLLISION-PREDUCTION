# Product context hard-negative mining

Run date: 2026-07-13

## Objective

Use the false-positive analysis from
`product_event_window_phase_classifier_seq8` to mine contextual hard negatives
from the training split.

The goal is to teach the phase classifier that visually risky but safe contexts
are still class `0`, instead of only adding generic negative windows.

Contextual hard negative pattern identified:

- dense traffic;
- close vehicles without collision;
- brake-light reflections;
- low-light/night scenes;
- degraded visual quality.

## Leakage Control

Validation false positives were used only to identify the failure pattern.

The mined training data was selected only from:

```text
split = train
video_target = 0
```

The holdout remained sealed.

## Mining Command

```powershell
.\venv\Scripts\python.exe scripts\mine_context_hard_negative_windows.py `
  --base-manifest data\interim\product_event_windows_seq8_prealert_phases_manifest.csv `
  --frame-manifest data\interim\product_temporal_frames_224_manifest.csv `
  --split-manifest data\interim\full_train_product_splits.csv `
  --checkpoint models\checkpoints\product_event_window_phase_classifier_seq8_best_sequence.pt `
  --threshold 0.64 `
  --max-peaks-per-video 4 `
  --min-gap-seconds 2.0 `
  --neighbor-frame-offsets=-2,-1,0,1,2 `
  --max-windows-per-video 12 `
  --sample-weight 2.5 `
  --batch-size 16 `
  --num-workers 2 `
  --output data\interim\product_event_windows_seq8_context_hard_negatives_manifest.csv
```

## Generated Artifacts

Train negative scores:

```text
models/reports/product_context_hard_negative_train_scores.csv
```

Selected risk peaks:

```text
models/reports/product_context_hard_negative_selected_peaks.csv
```

Mined windows:

```text
models/reports/product_context_hard_negative_windows.csv
```

Augmented manifest:

```text
data/interim/product_event_windows_seq8_context_hard_negatives_manifest.csv
```

## Mining Result

| Metric | Value |
| --- | ---: |
| Train negative videos scored | 526 |
| Train negative frame sequences scored | 39204 |
| Videos with selected peaks | 240 |
| Selected peaks | 513 |
| Mined contextual hard-negative windows | 2202 |
| Hard-negative sample weight | 2.5 |

Top selected training peaks:

| Video ID | Timestamp | Risk score | P(class 2) | P(class 3) |
| --- | ---: | ---: | ---: | ---: |
| 01785 | 26.5 | 0.866 | 0.256 | 0.611 |
| 01052 | 2.5 | 0.847 | 0.216 | 0.631 |
| 01822 | 27.5 | 0.833 | 0.143 | 0.690 |
| 01269 | 27.0 | 0.832 | 0.153 | 0.679 |
| 01653 | 0.5 | 0.827 | 0.102 | 0.725 |

## Augmented Manifest Distribution

Split distribution:

| Split | Windows |
| --- | ---: |
| train | 15424 |
| val | 2837 |

Phase distribution:

| Phase index | Windows |
| ---: | ---: |
| 0 | 13646 |
| 1 | 2552 |
| 2 | 486 |
| 3 | 1577 |

Window type distribution:

| Window type | Windows |
| --- | ---: |
| contextual_hard_negative | 2202 |
| event_near | 1577 |
| negative_video | 7656 |
| positive_safe | 3788 |
| prealert_early | 2552 |
| prealert_late | 486 |

## Next Training Command

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\product_event_windows_seq8_context_hard_negatives_manifest.csv `
  --experiment-name product_context_hard_negatives_phase_classifier_seq8 `
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

## Evaluation Plan

After training:

1. Evaluate full validation videos.
2. Sweep thresholds with raw scores, 2 consecutive frames, and moving average
   3s plus 2 consecutive frames.
3. Compare against `product_event_window_phase_classifier_seq8`.
4. Do not run holdout unless validation satisfies:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```
