# Product event-window GRU seq8 pre-alert phases experiment

Run date: 2026-07-13

## Objective

Improve the temporal target definition by separating positive videos into
explicit pre-alert phases instead of treating every event-window as the same
positive class.

Experiment:

```text
product_event_window_gru_seq8_prealert_phases
```

Best checkpoint:

```text
models/checkpoints/product_event_window_gru_seq8_prealert_phases_best_sequence.pt
```

## Implementation

Created:

```text
scripts/create_prealert_phase_sequence_manifest.py
```

The new manifest keeps the binary model but changes the phase definition:

| Window type | Target | Weight | Meaning |
| --- | ---: | ---: | --- |
| `negative_video` | 0 | 1.0 | Normal videos |
| `positive_safe` | 0 | 1.5 | Safe region in positive videos |
| `prealert_early` | 0 | 0.6 | Transition before `time_of_alert` |
| `prealert_late` | 1 | 1.0 | Alertable region after `time_of_alert` |
| `event_near` | 1 | 1.4 | Near-collision region |

This is intentionally stricter than the previous event-window labels: frames
before `time_of_alert` are no longer full positives.

## Manifest Setup

Output manifest:

```text
data/interim/product_event_windows_seq8_prealert_phases_manifest.csv
```

The manifest was generated with:

```text
--split-manifest data/interim/full_train_product_splits.csv
--allowed-splits train,val
```

The holdout remained sealed.

Manifest distribution:

| Split | Windows |
| --- | ---: |
| train | 13222 |
| val | 2837 |

| Target | Windows |
| --- | ---: |
| 0 | 13996 |
| 1 | 2063 |

| Window type | Windows |
| --- | ---: |
| event_near | 1577 |
| negative_video | 7656 |
| positive_safe | 3788 |
| prealert_early | 2552 |
| prealert_late | 486 |

## Training Setup

| Parameter | Value |
| --- | --- |
| Manifest | `data/interim/product_event_windows_seq8_prealert_phases_manifest.csv` |
| Manifest type | `explicit_windows` |
| Sequence length | `8` |
| Model | ResNet18 frozen encoder + GRU |
| Hidden size | `128` |
| Trainable parameters | `246786` |
| Batch size | `16` |
| AMP | `true` |
| Imbalance strategy | `class_weight` |
| Sample weight column | `sample_weight` |
| Monitor metric | `alert_selection_score` |
| Early stopping patience | `2` |

## Training Result

Training stopped early at epoch 3. The best checkpoint was epoch 1.

| Epoch | Train loss | Window F1 | Window ROC-AUC | Alert threshold | Alert precision | Alert recall | False alarm rate | Alert score | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.700 | 0.373 | 0.749 | 0.50 | 0.706 | 0.750 | 0.312 | 2.043 | Yes |
| 2 | 0.653 | 0.340 | 0.736 | 0.61 | 0.708 | 0.670 | 0.277 | 1.511 | No |
| 3 | 0.629 | 0.349 | 0.733 | 0.62 | 0.728 | 0.670 | 0.250 | 1.585 | No |

Best validation-window alert proxy:

| Metric | Value |
| --- | ---: |
| Alert threshold | 0.50 |
| Alert precision | 0.706 |
| Alert recall | 0.750 |
| False alarm rate | 0.312 |
| Missed event rate | 0.250 |
| Mean alert time error | -8.708 s |

Interpretation:

- Window ROC-AUC improved to `0.749`, the strongest window-level separation in
  the sequence experiments so far.
- The phase target reduced some temporal ambiguity, but it also made the binary
  task more conservative because only late/event-near windows are positive.
- The validation proxy still did not satisfy the recall requirement.

## Temporal Alert Evaluation At Threshold 0.50

The best checkpoint was evaluated on the full validation videos using the
original temporal frame manifest.

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 187 |
| True positive alerts | 99 |
| False positive alerts | 85 |
| Missed events | 13 |
| Alert precision | 0.538 |
| Alert recall | 0.884 |
| False alarm rate | 0.759 |
| Missed event rate | 0.116 |
| Mean predicted lead time | 15.087 s |
| Mean alert time error | -13.349 s |

At threshold `0.50`, the model remains too aggressive on full validation videos.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.65 | 0.649 | 0.545 | 0.295 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.52 | 0.532 | 0.821 | 0.723 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.59 | 0.677 | 0.580 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.49 | 0.551 | 0.821 | 0.670 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.57 | 0.655 | 0.491 | 0.259 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.45 | 0.558 | 0.812 | 0.643 |

## Product Gate

Validation candidate gate:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Result:

```text
Failed.
```

No threshold or tested temporal post-processing rule satisfied the product gate.

## Decision

Do not evaluate this model on holdout.

The phase-aware target improved window-level separation, but full-video alert
behavior still fails the product gate.

## Recommended Next Step

The result suggests that phase information is useful, but compressing it back
into a binary target loses too much structure. The next modeling step should be a
true multi-class phase model:

```text
product_event_window_phase_classifier_seq8
```

Expected idea:

- predict phase classes directly: safe/negative, early pre-alert, late
  pre-alert, event-near;
- derive the binary alert score from late/event-near probabilities;
- evaluate whether this creates a smoother and better calibrated risk curve.
