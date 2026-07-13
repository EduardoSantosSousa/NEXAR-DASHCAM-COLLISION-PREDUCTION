# Product event-window GRU seq8 alert-weighted experiment

Run date: 2026-07-13

## Objective

Test whether alert-aware checkpoint selection and hard-negative sample weighting
improve the product operating point for the event-window GRU model.

Experiment:

```text
product_event_window_gru_seq8_alert_weighted
```

Best checkpoint:

```text
models/checkpoints/product_event_window_gru_seq8_alert_weighted_best_sequence.pt
```

## Implementation Changes

This experiment added two product-oriented controls to the sequence training
pipeline:

- per-window sample weights through `sample_weight`;
- optional checkpoint selection by validation alert operating point through
  `--alert-metric-selection`.

The selected monitor metric was:

```text
alert_selection_score
```

The alert-selection rule searched thresholds from `0.40` to `0.80`, required
`2` consecutive windows above threshold, and scored each epoch against:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Window weights:

| Window type | Weight |
| --- | ---: |
| hard_negative_mined | 2.0 |
| negative_video | 1.0 |
| positive_safe | 1.0 |
| positive_event | 1.1 |

## Training Setup

| Parameter | Value |
| --- | --- |
| Manifest | `data/interim/product_event_windows_seq8_hard_negative_manifest.csv` |
| Manifest type | `explicit_windows` |
| Sequence length | `8` |
| Model | ResNet18 frozen encoder + GRU |
| Hidden size | `128` |
| Trainable parameters | `246786` |
| Batch size | `16` |
| AMP | `true` |
| Imbalance strategy | `class_weight` |
| Loss | `cross_entropy` |
| Monitor metric | `alert_selection_score` |
| Early stopping patience | `2` |

## Window-Level Training Result

Training stopped early at epoch 5. The best checkpoint was epoch 3.

| Epoch | Train loss | Window F1 | Window ROC-AUC | Alert threshold | Alert precision | Alert recall | False alarm rate | Alert score | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.696 | 0.077 | 0.667 | 0.40 | 0.812 | 0.580 | 0.134 | 1.188 | Yes |
| 2 | 0.681 | 0.626 | 0.689 | 0.50 | 0.730 | 0.750 | 0.277 | 2.177 | Yes |
| 3 | 0.668 | 0.526 | 0.685 | 0.45 | 0.717 | 0.812 | 0.321 | 2.447 | Yes |
| 4 | 0.661 | 0.506 | 0.686 | 0.42 | 0.685 | 0.795 | 0.366 | 2.081 | No |
| 5 | 0.647 | 0.581 | 0.657 | 0.50 | 0.664 | 0.759 | 0.384 | 1.644 | No |

Best validation-window alert proxy:

| Metric | Value |
| --- | ---: |
| Alert threshold | 0.45 |
| Alert precision | 0.717 |
| Alert recall | 0.812 |
| False alarm rate | 0.321 |
| Missed event rate | 0.188 |
| Mean alert time error | -5.446 s |

Interpretation:

- The alert-aware selector chose a different checkpoint from pure ROC-AUC
  selection.
- The validation-window proxy almost reached the product gate, but false alarm
  rate remained slightly above `0.30`.
- The window-level proxy was useful for model selection, but it was still too
  optimistic compared with full-video evaluation.

## Temporal Alert Evaluation At Threshold 0.50

The best checkpoint was evaluated on the full validation videos using the
original temporal frame manifest.

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 165 |
| True positive alerts | 93 |
| False positive alerts | 68 |
| Missed events | 19 |
| Alert precision | 0.578 |
| Alert recall | 0.830 |
| False alarm rate | 0.607 |
| Missed event rate | 0.170 |
| Mean predicted lead time | 15.034 s |
| Mean alert time error | -13.344 s |

Compared with `product_event_window_gru_seq8_hard_negative`, this model reduced
false alarm rate at threshold `0.50` from `0.830` to `0.607`, while keeping
recall above `0.80`. This is a meaningful improvement, but still not enough for
product use.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.57 | 0.652 | 0.518 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.51 | 0.592 | 0.804 | 0.554 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.54 | 0.652 | 0.518 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.49 | 0.600 | 0.804 | 0.536 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.52 | 0.645 | 0.536 | 0.295 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.47 | 0.596 | 0.804 | 0.545 |

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

No raw threshold or tested temporal post-processing rule satisfied the product
gate on full validation videos.

## Comparison Against Recent Sequence Models

At threshold `0.50` on full validation videos:

| Model | Precision | Recall | False alarm rate |
| --- | ---: | ---: | ---: |
| Event-window GRU seq8 | 0.557 | 0.866 | 0.688 |
| Event-window GRU seq8 hard-negative | 0.518 | 0.893 | 0.830 |
| Event-window GRU seq8 alert-weighted | 0.578 | 0.830 | 0.607 |

Best low-false-alarm operating point:

| Model | Rule | Threshold | Precision | Recall | False alarm rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Event-window GRU seq8 | Raw scores | 0.65 | 0.680 | 0.625 | 0.295 |
| Event-window GRU seq8 hard-negative | 2 consecutive frames | 0.59 | 0.696 | 0.634 | 0.277 |
| Event-window GRU seq8 alert-weighted | Moving average 3s + 2 frames | 0.52 | 0.645 | 0.536 | 0.295 |

Alert-aware checkpointing reduced false alarms at threshold `0.50`, but it did
not improve the best low-false-alarm recall. The main remaining issue is still
separation between true pre-collision windows and visually risky negative
driving scenes.

## Decision

Do not evaluate this model on holdout.

The holdout remains sealed because the validation product gate failed.

## Recommended Next Step

The current ResNet18 frozen encoder appears to be the limiting factor. The next
experiment should keep alert-aware selection but increase temporal/visual
capacity carefully:

```text
product_event_window_gru_seq8_alert_weighted_layer4
```

Recommended changes:

- start from the same hard-negative manifest;
- keep alert-aware checkpoint selection;
- fine-tune ResNet18 `layer4` with a lower CNN learning rate;
- use a lower head learning rate than the previous GRU runs;
- compare full-video validation sweeps before considering holdout.
