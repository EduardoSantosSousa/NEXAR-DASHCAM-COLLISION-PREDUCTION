# Temporal Label Progression

## Motivation

The first baseline used video-level labels for individual frames. This caused a
conceptual problem: every frame from a positive video was treated as risky, even
when it appeared long before the annotated alert time.

To address this, a new temporally labeled frame dataset was created.

## Temporal Label Definition

For negative videos:

```text
all frames -> temporal_target = 0
```

For positive videos:

```text
timestamp < time_of_alert - 3s       -> temporal_target = 0
time_of_alert - 3s <= timestamp <= time_of_event -> temporal_target = 1
```

This changes the model question from:

```text
Does this frame come from a positive video?
```

to:

```text
Is this frame in the alert-relevant temporal region?
```

## Generated Dataset

| Artifact | Location |
| --- | --- |
| Temporal frame manifest | `data/interim/temporal_frames_224_manifest.csv` |
| Temporal frames | `data/interim/temporal_frames_224/` |

The dataset was generated at 2 FPS with frames resized to 224x224.

| Class | Count |
| --- | ---: |
| `0` | 7,295 |
| `1` | 465 |

The class imbalance is expected because the alert/event window is much shorter
than the full video duration. Training therefore uses class-weighted cross entropy.

## Experiment

Command:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --experiment-name temporal_alert_224 --epochs 3 --batch-size 64 --learning-rate 0.0001 --num-workers 2
```

The model was evaluated as a temporal alert system:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_resnet18.pt --experiment-name temporal_alert_224 --fps 1 --threshold 0.5 --batch-size 32
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224
```

## Comparison Against Previous Baseline

At threshold `0.50`:

| Model | Alert Precision | Alert Recall | False Alarm Rate | Missed Event Rate | Mean Lead Time | Mean Alert Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Video-label baseline | 0.538 | 0.980 | 0.840 | 0.020 | 17.190 s | -15.606 s |
| Temporal-label model | 0.608 | 0.960 | 0.620 | 0.040 | 14.690 s | -13.150 s |

The temporal-label model reduces false alarms and improves alert precision while
keeping recall high. It still alerts earlier than the annotated alert time, but
the behavior is less extreme than the previous baseline.

## Threshold Trade-Off

For the temporal-label model:

| Threshold | Precision | Recall | False Alarm Rate | Missed Event Rate | Mean Lead Time | Mean Alert Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.608 | 0.960 | 0.620 | 0.040 | 14.690 | -13.150 |
| 0.70 | 0.708 | 0.920 | 0.380 | 0.080 | 11.622 | -10.069 |
| 0.90 | 0.902 | 0.740 | 0.080 | 0.260 | 7.828 | -6.214 |
| 0.95 | 1.000 | 0.600 | 0.000 | 0.400 | 4.605 | -3.002 |
| 0.99 | 1.000 | 0.120 | 0.000 | 0.880 | 2.922 | -0.997 |

## Conclusion

The progression confirms that temporally meaningful labels improve alert behavior.
However, the model remains limited because it still classifies individual frames
independently. The next step is to use transfer learning and then move toward
sequence-based modeling such as CNN + GRU/LSTM or short-window temporal
aggregation.

## Validation-Only Follow-Up

A fixed video-level split was created after the initial temporal-label run:

```text
data/interim/sample_100_videos_splits.csv
```

The corrected protocol evaluates only the 20 validation videos instead of all
100 sampled videos. This makes the results more conservative and more defensible.

At threshold `0.50`:

| Model | Eval split | Precision | Recall | False Alarm Rate | Missed Event Rate | Mean Lead Time | Mean Alert Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal-label ResNet18 from scratch | val only | 0.444 | 0.400 | 0.500 | 0.600 | 19.953 s | -17.960 s |
| Temporal-label ResNet18 pretrained | val only | 0.500 | 0.300 | 0.300 | 0.700 | 12.011 s | -9.774 s |

At the default threshold, transfer learning reduced false alarms and made alert
timing less extreme, but it also reduced recall. When thresholds are selected
under a matched minimum recall constraint of `0.50`, the pretrained model is
more favorable:

| Model | Threshold | Precision | Recall | False Alarm Rate | Mean Alert Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scratch | 0.38 | 0.455 | 0.500 | 0.600 | -16.876 s |
| Pretrained | 0.18 | 0.625 | 0.500 | 0.300 | -11.916 s |

Detailed results are documented in:

```text
reports/experiment_comparison_val.md
```
