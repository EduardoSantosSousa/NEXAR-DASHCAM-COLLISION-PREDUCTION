# Product EfficientNet-B0 experiment

Run date: 2026-07-13

## Objective

Evaluate whether EfficientNet-B0 improves the product validation tradeoff over
the full-dataset ResNet18 baseline.

Experiment:

```text
product_temporal_alert_224_efficientnet_b0_frozen_amp
```

Candidate checkpoint:

```text
product_temporal_alert_224_efficientnet_b0_frozen_amp_best_efficientnet_b0.pt
```

## Implementation Notes

The baseline CNN training pipeline now supports multiple backbones:

- `resnet18`
- `efficientnet_b0`

Additional training controls were added for long product experiments:

- `--freeze-backbone`
- `--amp`
- `--log-every-n-batches`

An initial full fine-tuning attempt with EfficientNet-B0 was stopped before the
first checkpoint because it did not complete one epoch in a practical iteration
window on the available GPU. The completed experiment therefore used
EfficientNet-B0 as a frozen ImageNet feature extractor and trained only the
classifier head with CUDA mixed precision.

## Data

| Split | Videos | Negative | Positive |
| --- | ---: | ---: | ---: |
| Train | 1052 | 526 | 526 |
| Validation | 224 | 112 | 112 |

Frame-level distribution:

| Split | Frames |
| --- | ---: |
| Train | 80004 |
| Validation | 16987 |

## Training Configuration

| Parameter | Value |
| --- | --- |
| Backbone | `efficientnet_b0` |
| Pretrained | `true` |
| Freeze backbone | `true` |
| AMP enabled | `true` |
| Batch size | `64` |
| Learning rate | `0.0003` |
| Max epochs | `4` |
| Early stopping patience | `2` |
| Monitor metric | `roc_auc` |
| Trainable parameters | `2562 / 4010110` |

## Training Results

Training stopped early at epoch 4. The best checkpoint was epoch 2.

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.672 | 0.173 | 0.645 | Yes |
| 2 | 0.653 | 0.170 | 0.653 | Yes |
| 3 | 0.644 | 0.167 | 0.645 | No |
| 4 | 0.643 | 0.167 | 0.640 | No |

Interpretation:

- The frozen EfficientNet-B0 head trains efficiently with AMP.
- The best validation ROC-AUC is `0.653`, below the full-dataset ResNet18
  baseline best ROC-AUC of `0.667`.
- The model does not provide stronger frame-level separability in this setup.

## Validation Alert Metrics At Threshold 0.50

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 207 |
| True positive alerts | 105 |
| False positive alerts | 97 |
| Missed events | 7 |
| Alert precision | 0.520 |
| Alert recall | 0.938 |
| False alarm rate | 0.866 |
| Missed event rate | 0.063 |
| Mean alert time error | -15.439 s |

At threshold `0.50`, recall is high, but the false alarm rate is far beyond a
usable product threshold.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.73 | 0.571 | 0.357 | 0.268 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.58 | 0.523 | 0.813 | 0.741 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.66 | 0.540 | 0.304 | 0.259 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.563 | 0.839 | 0.652 |

## Score Distribution Diagnostic

Video-level maximum score quantiles:

| Video target | 10% | 25% | 50% | 75% | 90% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.471 | 0.577 | 0.679 | 0.735 | 0.798 |
| 1 | 0.582 | 0.642 | 0.729 | 0.773 | 0.827 |

The overlap between negative and positive maximum scores is large. Many
negative videos receive high peak risk scores, which causes false alerts when
the threshold is low enough to preserve recall.

## Product Gate

Target minimum for a product candidate:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Result:

```text
Failed.
```

No validation threshold satisfied the product gate.

## Decision

Do not evaluate this model on holdout as a product candidate.

The frozen EfficientNet-B0 experiment improves training practicality but does
not improve the validation tradeoff. The holdout should remain sealed.

## Recommended Next Step

Run a partial fine-tuning experiment only after improving observability and
runtime controls. Recommended next experiment:

```text
product_temporal_alert_224_efficientnet_b0_amp_finetune_last_blocks
```

Recommended setup:

- unfreeze the EfficientNet-B0 classifier and the last feature block;
- keep AMP enabled;
- keep `--log-every-n-batches`;
- use a lower learning rate such as `0.00005`;
- monitor validation ROC-AUC and stop early if it does not exceed the ResNet18
  reference of `0.667`.

If partial fine-tuning still fails, move to a temporal/product architecture
that uses sequence-level context or a stronger modern backbone such as
`convnext_tiny`, subject to GPU memory constraints.
