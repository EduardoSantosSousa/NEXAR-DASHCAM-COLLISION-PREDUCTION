# Product event-window GRU seq8 experiment

Run date: 2026-07-13

## Objective

Evaluate whether the event-centered temporal window formulation improves over
the frame-level baselines and becomes a product candidate.

Experiment:

```text
product_event_window_gru_seq8_resnet18_frozen
```

Best checkpoint:

```text
models/checkpoints/product_event_window_gru_seq8_resnet18_frozen_best_sequence.pt
```

## Training Setup

| Parameter | Value |
| --- | --- |
| Manifest | `data/interim/product_event_windows_seq8_manifest.csv` |
| Manifest type | `explicit_windows` |
| Sequence length | `8` |
| Model | ResNet18 frozen encoder + GRU |
| Hidden size | `128` |
| Trainable parameters | `246786` |
| Batch size | `16` |
| AMP | `true` |
| Monitor metric | `roc_auc` |
| Early stopping patience | `2` |

Dataset:

| Split | Sequences |
| --- | ---: |
| Train | 8324 |
| Validation | 1778 |

## Window-Level Training Result

Training stopped early at epoch 5. The best checkpoint was epoch 3.

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.670 | 0.638 | 0.705 | Yes |
| 2 | 0.646 | 0.629 | 0.706 | Yes |
| 3 | 0.631 | 0.638 | 0.707 | Yes |
| 4 | 0.618 | 0.655 | 0.698 | No |
| 5 | 0.602 | 0.504 | 0.697 | No |

Best validation window metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.681 |
| Recall | 0.600 |
| F1 | 0.638 |
| ROC-AUC | 0.707 |

Interpretation:

- This is the first model to clearly exceed the frame-level backbone ROC-AUC
  range on validation windows.
- Later epochs reduce training loss but do not improve validation ROC-AUC,
  indicating mild overfitting after epoch 3.

## Temporal Alert Evaluation At Threshold 0.50

The best checkpoint was evaluated on the full validation videos using the
original temporal frame manifest.

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 180 |
| True positive alerts | 97 |
| False positive alerts | 77 |
| Missed events | 15 |
| Alert precision | 0.557 |
| Alert recall | 0.866 |
| False alarm rate | 0.688 |
| Missed event rate | 0.134 |
| Mean alert time error | -14.964 s |

At threshold `0.50`, recall is strong, but false alarms remain too high for a
product candidate.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.65 | 0.680 | 0.625 | 0.295 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.54 | 0.566 | 0.804 | 0.616 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.62 | 0.684 | 0.598 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.51 | 0.588 | 0.804 | 0.563 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.60 | 0.688 | 0.589 | 0.268 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.48 | 0.588 | 0.804 | 0.563 |

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

This is the best direction so far and should become the next modeling branch,
but it is not yet reliable enough for product validation.

## Comparison Against Previous Best

Compared with the full-dataset ResNet18 baseline, the event-window GRU improves
the low-false-alarm operating point:

| Model | Rule | Threshold | Precision | Recall | False alarm rate |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet18 full dataset | 2 consecutive frames | 0.70 | 0.663 | 0.580 | 0.295 |
| Event-window GRU seq8 | Raw scores | 0.65 | 0.680 | 0.625 | 0.295 |

The improvement is meaningful but not sufficient. The main remaining problem is
still false alarms when recall is pushed above `0.80`.

## Recommended Next Step

Keep the event-window formulation and improve negative learning.

Recommended next experiment:

```text
product_event_window_gru_seq8_hard_negative
```

Rationale:

- The model detects many positive events, but fires too often on negative
  validation videos.
- Random negative windows are not enough.
- The next dataset should include more difficult negative windows, especially
  windows near high-risk visual patterns in negative videos and safe regions of
  positive videos.

Suggested changes:

- increase `negative_windows_per_video`;
- increase `positive_safe_windows_per_video`;
- add a hard-negative mining pass using high-scoring false-positive windows from
  the training split only;
- keep validation and holdout untouched for model selection.
