# Product event-window GRU seq8 alert-weighted layer4 experiment

Run date: 2026-07-13

## Objective

Evaluate whether conservative fine-tuning of the final ResNet18 visual block
improves the alert-weighted event-window GRU model.

Experiment:

```text
product_event_window_gru_seq8_alert_weighted_layer4
```

Best checkpoint:

```text
models/checkpoints/product_event_window_gru_seq8_alert_weighted_layer4_best_sequence.pt
```

## Implementation

The temporal model already supported `cnn_train_policy=layer4`. This experiment
made the fine-tuning behavior stricter by keeping frozen CNN blocks in eval mode
during training:

```text
cnn_train_policy=frozen -> full CNN eval
cnn_train_policy=layer4 -> lower CNN blocks eval, layer4 train
cnn_train_policy=full -> full CNN train
```

This avoids BatchNorm running-stat updates in frozen visual layers.

Trainable parameters:

| Parameter group | Count |
| --- | ---: |
| CNN layer4 | 8393728 |
| Temporal head | 246786 |
| Total trainable | 8640514 |

## Training Setup

| Parameter | Value |
| --- | --- |
| Manifest | `data/interim/product_event_windows_seq8_hard_negative_manifest.csv` |
| Manifest type | `explicit_windows` |
| Sequence length | `8` |
| Model | ResNet18 layer4 fine-tuned encoder + GRU |
| Hidden size | `128` |
| Batch size | `8` |
| Head learning rate | `0.00005` |
| CNN learning rate | `0.000005` |
| AMP | `true` |
| Loss | `cross_entropy` |
| Hard-negative weight | `2.0` |
| Positive-event weight | `1.1` |
| Monitor metric | `alert_selection_score` |
| Early stopping patience | `2` |

## Training Result

Training stopped early at epoch 3. The best checkpoint was epoch 1.

| Epoch | Train loss | Window F1 | Alert threshold | Alert precision | Alert recall | False alarm rate | Alert score | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.680 | 0.240 | 0.40 | 0.705 | 0.705 | 0.295 | 1.759 | Yes |
| 2 | 0.604 | 0.579 | 0.54 | 0.682 | 0.670 | 0.312 | 1.340 | No |
| 3 | 0.480 | 0.449 | 0.50 | 0.667 | 0.607 | 0.304 | 0.839 | No |

Interpretation:

- Training loss fell quickly, but alert selection score dropped after epoch 1.
- This indicates rapid overfitting after unfreezing `layer4`.
- The best proxy point controlled false alarms, but recall stayed too low.

## Temporal Alert Evaluation At Threshold 0.50

The best checkpoint was evaluated on the full validation videos using the
original temporal frame manifest.

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 100 |
| True positive alerts | 50 |
| False positive alerts | 35 |
| Missed events | 62 |
| Alert precision | 0.588 |
| Alert recall | 0.446 |
| False alarm rate | 0.312 |
| Missed event rate | 0.554 |
| Mean predicted lead time | 11.854 s |
| Mean alert time error | -10.061 s |

At threshold `0.50`, the model is too conservative and misses too many positive
events.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.51 | 0.623 | 0.429 | 0.259 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.43 | 0.585 | 0.830 | 0.589 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.49 | 0.595 | 0.393 | 0.268 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.41 | 0.583 | 0.812 | 0.580 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.47 | 0.556 | 0.357 | 0.286 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.39 | 0.573 | 0.804 | 0.598 |

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

Fine-tuning `layer4` with this manifest and learning-rate schedule did not
improve product-readiness. It reduced some false alarms but lost too much recall.

## Recommended Next Step

Do not keep escalating CNN fine-tuning on the current window labels.

The next modeling work should improve the target formulation and negatives:

```text
product_event_window_gru_seq8_dense_negatives
```

Recommended direction:

- generate more negative windows per negative video instead of duplicating only
  high-scoring hard negatives;
- include more safe windows from positive videos before the alert region;
- keep the frozen encoder as the reference model;
- keep alert-aware checkpoint selection;
- compare full-video validation metrics before any holdout evaluation.
