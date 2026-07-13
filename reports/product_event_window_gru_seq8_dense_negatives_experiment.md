# Product event-window GRU seq8 dense-negatives experiment

Run date: 2026-07-13

## Objective

Test whether a denser negative/safe-window formulation helps the event-window
GRU learn normal driving patterns and reduce false alarms.

Experiment:

```text
product_event_window_gru_seq8_dense_negatives
```

Best checkpoint:

```text
models/checkpoints/product_event_window_gru_seq8_dense_negatives_best_sequence.pt
```

## Data Safety Fix

During the first manifest generation attempt, `--allowed-splits train,val` was
provided without a split manifest. The script would have included all videos,
including holdout, because the frame manifest did not contain a `split` column.

The script was updated to fail fast when `allowed_splits` is provided but no
`split` column is available. It also drops an existing split column before
merging a split manifest to avoid duplicate split columns.

Correct manifest generation used:

```text
--split-manifest data/interim/full_train_product_splits.csv
--allowed-splits train,val
```

The holdout remained sealed.

## Manifest Setup

Output manifest:

```text
data/interim/product_event_windows_seq8_dense_negatives_manifest.csv
```

Sampling changes relative to the original event-window manifest:

| Parameter | Original | Dense negatives |
| --- | ---: | ---: |
| Negative windows per negative video | 6 | 20 |
| Positive safe windows per positive video | 2 | 8 |
| Max positive event windows per video | 8 | 8 |
| Sequence length | 8 | 8 |

Manifest distribution:

| Split | Windows |
| --- | ---: |
| train | 18775 |
| val | 4002 |

| Target | Windows |
| --- | ---: |
| 0 | 17775 |
| 1 | 5002 |

| Window type | Windows |
| --- | ---: |
| negative_video | 12760 |
| positive_event | 5002 |
| positive_safe | 5015 |

## Training Setup

| Parameter | Value |
| --- | --- |
| Manifest | `data/interim/product_event_windows_seq8_dense_negatives_manifest.csv` |
| Manifest type | `explicit_windows` |
| Sequence length | `8` |
| Model | ResNet18 frozen encoder + GRU |
| Hidden size | `128` |
| Trainable parameters | `246786` |
| Batch size | `16` |
| AMP | `true` |
| Imbalance strategy | `class_weight` |
| Loss | `cross_entropy` |
| Negative-video weight | `1.2` |
| Positive-safe weight | `1.5` |
| Positive-event weight | `1.0` |
| Monitor metric | `alert_selection_score` |
| Early stopping patience | `2` |

## Training Result

Training stopped early at epoch 5. The best checkpoint was epoch 3.

| Epoch | Train loss | Window F1 | Window ROC-AUC | Alert threshold | Alert precision | Alert recall | False alarm rate | Alert score | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.614 | 0.293 | 0.687 | 0.37 | 0.612 | 0.804 | 0.509 | 1.203 | Yes |
| 2 | 0.585 | 0.400 | 0.682 | 0.47 | 0.634 | 0.741 | 0.429 | 1.186 | No |
| 3 | 0.561 | 0.432 | 0.690 | 0.51 | 0.679 | 0.679 | 0.321 | 1.357 | Yes |
| 4 | 0.528 | 0.406 | 0.669 | 0.45 | 0.601 | 0.821 | 0.545 | 1.045 | No |
| 5 | 0.485 | 0.337 | 0.647 | 0.35 | 0.551 | 0.679 | 0.554 | -0.187 | No |

Best validation-window alert proxy:

| Metric | Value |
| --- | ---: |
| Alert threshold | 0.51 |
| Alert precision | 0.679 |
| Alert recall | 0.679 |
| False alarm rate | 0.321 |
| Missed event rate | 0.321 |
| Mean alert time error | -10.674 s |

Interpretation:

- Dense negatives improved window ROC-AUC slightly versus the previous
  alert-weighted run, but this did not translate into a better product operating
  point.
- The model became more conservative at the selected checkpoint: false alarms
  were lower in the proxy, but recall fell below the product target.

## Temporal Alert Evaluation At Threshold 0.50

The best checkpoint was evaluated on the full validation videos using the
original temporal frame manifest.

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 166 |
| True positive alerts | 89 |
| False positive alerts | 70 |
| Missed events | 23 |
| Alert precision | 0.560 |
| Alert recall | 0.795 |
| False alarm rate | 0.625 |
| Missed event rate | 0.205 |
| Mean predicted lead time | 15.354 s |
| Mean alert time error | -13.620 s |

At threshold `0.50`, the model remains too noisy for product use.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.64 | 0.674 | 0.571 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.49 | 0.556 | 0.804 | 0.643 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.60 | 0.677 | 0.598 | 0.286 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.45 | 0.562 | 0.804 | 0.625 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.57 | 0.667 | 0.554 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.40 | 0.552 | 0.804 | 0.652 |

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

Dense negative sampling improved coverage of normal windows, but it did not solve
the recall/false-alarm tradeoff on full validation videos.

## Recommended Next Step

The issue now appears less like a simple sampling-count problem and more like a
target-definition problem. The model needs stronger supervision for when an
alert should start, not only whether a window is positive or negative.

Recommended next experiment:

```text
product_event_window_gru_seq8_prealert_phases
```

Suggested direction:

- split positive windows into phases such as safe, early pre-alert, late
  pre-alert, and event-near;
- train either a multi-class phase model or a binary model with phase-dependent
  sample weights;
- evaluate whether the risk curve becomes more temporally aligned and less
  noisy before holdout.
