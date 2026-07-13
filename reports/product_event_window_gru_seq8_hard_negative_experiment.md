# Product event-window GRU seq8 hard-negative experiment

Run date: 2026-07-13

## Objective

Test whether adding mined hard-negative training windows reduces false alarms in
the event-window GRU formulation.

Experiment:

```text
product_event_window_gru_seq8_hard_negative
```

Best checkpoint:

```text
models/checkpoints/product_event_window_gru_seq8_hard_negative_best_sequence.pt
```

## Hard-Negative Mining Setup

Base manifest:

```text
data/interim/product_event_windows_seq8_manifest.csv
```

Augmented manifest:

```text
data/interim/product_event_windows_seq8_hard_negative_manifest.csv
```

Mining source checkpoint:

```text
models/checkpoints/product_event_window_gru_seq8_resnet18_frozen_best_sequence.pt
```

Only train-split negative windows were scored. Validation and holdout were not
used for hard-negative mining.

Mining result:

| Item | Value |
| --- | ---: |
| Scored train negative windows | 4204 |
| Selected hard negatives | 1024 |
| Minimum hard-negative score | 0.50 |
| Maximum hard negatives per video | 2 |

Augmented manifest distribution:

| Split | Windows |
| --- | ---: |
| train | 9348 |
| val | 1778 |

| Target | Windows |
| --- | ---: |
| 0 | 6124 |
| 1 | 5002 |

| Window type | Windows |
| --- | ---: |
| hard_negative_mined | 1024 |
| negative_video | 3828 |
| positive_event | 5002 |
| positive_safe | 1272 |

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
| Monitor metric | `roc_auc` |
| Early stopping patience | `2` |

## Window-Level Training Result

Training stopped early at epoch 4. The best checkpoint was epoch 2.

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.689 | 0.281 | 0.690 | Yes |
| 2 | 0.672 | 0.651 | 0.701 | Yes |
| 3 | 0.657 | 0.594 | 0.698 | No |
| 4 | 0.649 | 0.539 | 0.698 | No |

Best validation window metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.651 |
| Recall | 0.651 |
| F1 | 0.651 |
| ROC-AUC | 0.701 |

Interpretation:

- Hard-negative training slightly improved best validation F1 versus the first
  event-window GRU run.
- Validation ROC-AUC fell from `0.707` to `0.701`, so the mined negatives did
  not improve ranking quality overall.
- The model became more aggressive at threshold `0.50`, which increased recall
  but also increased false alarms.

## Temporal Alert Evaluation At Threshold 0.50

The best checkpoint was evaluated on the full validation videos using the
original temporal frame manifest.

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 201 |
| True positive alerts | 100 |
| False positive alerts | 93 |
| Missed events | 12 |
| Alert precision | 0.518 |
| Alert recall | 0.893 |
| False alarm rate | 0.830 |
| Missed event rate | 0.107 |
| Mean predicted lead time | 17.262 s |
| Mean alert time error | -15.543 s |

At threshold `0.50`, recall is higher than the previous event-window GRU, but
false alarms are too high for a product candidate.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.61 | 0.689 | 0.652 | 0.295 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.55 | 0.590 | 0.821 | 0.571 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.59 | 0.696 | 0.634 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.52 | 0.575 | 0.821 | 0.607 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.57 | 0.680 | 0.607 | 0.286 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.559 | 0.804 | 0.634 |

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
gate.

## Comparison Against Previous Event-Window GRU

Best low-false-alarm operating points:

| Model | Rule | Threshold | Precision | Recall | False alarm rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Event-window GRU seq8 | Raw scores | 0.65 | 0.680 | 0.625 | 0.295 |
| Event-window GRU seq8 hard-negative | Raw scores | 0.61 | 0.689 | 0.652 | 0.295 |
| Event-window GRU seq8 | 2 consecutive frames | 0.62 | 0.684 | 0.598 | 0.277 |
| Event-window GRU seq8 hard-negative | 2 consecutive frames | 0.59 | 0.696 | 0.634 | 0.277 |

Hard-negative mining helped the low-false-alarm region modestly, but it did not
solve the recall/high-false-alarm tradeoff required for a product candidate.

## Decision

Do not evaluate this model on holdout.

The holdout remains sealed because the validation product gate failed.

## Recommended Next Step

Keep the event-window formulation, but change the objective and sampling rather
than only duplicating hard negatives.

Recommended next experiment:

```text
product_event_window_gru_seq8_alert_weighted
```

Suggested changes:

- keep mined hard negatives in the training manifest;
- select checkpoints using validation alert metrics, not only window ROC-AUC;
- increase the cost of false positives from negative videos;
- add a validation-time operating-point selector that minimizes false alarm rate
  subject to recall constraints;
- consider a stronger temporal head only after the alert-metric selection loop is
  implemented.
