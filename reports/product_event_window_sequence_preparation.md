# Product event-window sequence preparation

Run date: 2026-07-13

## Objective

Prepare the next modeling step after the frame-level backbones failed to reach a
product-grade validation tradeoff.

The new formulation trains on explicit temporal windows instead of treating
every frame as an independent classification example.

## What Changed

Added an event-centered sequence-window manifest generator:

```text
scripts/create_event_window_sequence_manifest.py
```

Added dataset/training support for explicit sequence windows:

```text
src/nexar_collision/data/dataset.py
src/nexar_collision/models/train_sequence.py
src/nexar_collision/evaluation/evaluate_sequence.py
scripts/train_sequence_model.py
```

The sequence trainer now supports:

```text
--amp
--log-every-n-batches
```

## Window Formulation

Each manifest row is one causal sequence window with:

- `frame_paths`: JSON list of frame paths;
- `timestamps`: JSON list of frame timestamps;
- `target`: window label;
- `window_type`: source strategy;
- `timestamp`: window end timestamp.

Positive windows are sampled from the pre-alert/event interval:

```text
max(0, time_of_alert - pre_alert_margin) <= window_end <= time_of_event
```

Negative windows come from:

- negative videos;
- early safe regions of positive videos, before the alert interval.

The holdout split was not included.

## Generated Manifest

Full train/validation manifest:

```text
data/interim/product_event_windows_seq8_manifest.csv
```

Generation command:

```powershell
.\venv\Scripts\python.exe scripts\create_event_window_sequence_manifest.py `
  --frame-manifest data\interim\product_temporal_frames_224_manifest.csv `
  --split-manifest data\interim\full_train_product_splits.csv `
  --output data\interim\product_event_windows_seq8_manifest.csv `
  --sequence-length 8 `
  --pre-alert-margin 3 `
  --safe-gap-seconds 2 `
  --max-positive-windows-per-video 8 `
  --negative-windows-per-video 6 `
  --positive-safe-windows-per-video 2 `
  --random-state 42
```

Manifest summary:

| Split | Windows |
| --- | ---: |
| Train | 8324 |
| Validation | 1778 |

Target distribution:

| Target | Windows |
| ---: | ---: |
| 0 | 5100 |
| 1 | 5002 |

Window type distribution:

| Window type | Windows |
| --- | ---: |
| `negative_video` | 3828 |
| `positive_event` | 5002 |
| `positive_safe` | 1272 |

## Smoke Training Result

Smoke experiment:

```text
product_event_window_gru_seq8_resnet18_frozen_smoke
```

Command:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\product_event_windows_seq8_manifest.csv `
  --experiment-name product_event_window_gru_seq8_resnet18_frozen_smoke `
  --pretrained `
  --epochs 1 `
  --batch-size 16 `
  --learning-rate 0.0001 `
  --num-workers 2 `
  --sequence-length 8 `
  --rnn-type gru `
  --hidden-size 128 `
  --imbalance-strategy class_weight `
  --amp `
  --log-every-n-batches 100 `
  --no-mlflow
```

Result:

| Metric | Value |
| --- | ---: |
| Train sequences | 8324 |
| Validation sequences | 1778 |
| Trainable parameters | 246786 |
| Validation precision | 0.663 |
| Validation recall | 0.615 |
| Validation F1 | 0.638 |
| Validation ROC-AUC | 0.705 |

This is the first result above the frame-level backbone reference range.

Important caveat:

This metric is window-level validation, not final product alert evaluation. It
is promising enough to justify a proper multi-epoch sequence run, but it is not
yet a product candidate.

## Recommended Next Command

Run the full event-window GRU experiment:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\product_event_windows_seq8_manifest.csv `
  --experiment-name product_event_window_gru_seq8_resnet18_frozen `
  --pretrained `
  --epochs 6 `
  --batch-size 16 `
  --learning-rate 0.0001 `
  --num-workers 2 `
  --sequence-length 8 `
  --rnn-type gru `
  --hidden-size 128 `
  --imbalance-strategy class_weight `
  --monitor-metric roc_auc `
  --monitor-mode max `
  --patience 2 `
  --amp `
  --log-every-n-batches 100
```

If this run keeps validation ROC-AUC above the frame-level baselines, evaluate
it as a temporal alert model on the full validation videos:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_sequence_model.py `
  --manifest data\interim\product_temporal_frames_224_manifest.csv `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split-manifest data\interim\full_train_product_splits.csv `
  --checkpoint models\checkpoints\product_event_window_gru_seq8_resnet18_frozen_best_sequence.pt `
  --experiment-name product_event_window_gru_seq8_resnet18_frozen `
  --split val `
  --sequence-length 8 `
  --threshold 0.5 `
  --batch-size 16
```

Then run threshold sweeps on the generated risk scores before considering any
holdout evaluation.

## Product Gate Reminder

Validation candidate gate:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Holdout remains sealed until a validation candidate satisfies this gate.
