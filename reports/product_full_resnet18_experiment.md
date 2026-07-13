# Product full-dataset ResNet18 experiment

Run date: 2026-07-12

## Objective

Evaluate whether the strongest sample-level baseline survives on the full
product split.

Experiment:

```text
product_temporal_alert_224_pretrained
```

Candidate model:

```text
product_temporal_alert_224_pretrained_best
```

## Data

| Split | Videos | Negative | Positive |
| --- | ---: | ---: | ---: |
| Train | 1052 | 526 | 526 |
| Validation | 224 | 112 | 112 |

Frame-level distribution:

| Split | Negative frames | Positive frames |
| --- | ---: | ---: |
| Train | 75169 | 4835 |
| Validation | 15927 | 1060 |

## Training Results

Training stopped early at epoch 4. The best checkpoint was epoch 1.

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.561 | 0.192 | 0.667 | Yes |
| 2 | 0.313 | 0.160 | 0.628 | No |
| 3 | 0.194 | 0.146 | 0.614 | No |
| 4 | 0.146 | 0.171 | 0.667 | No |

Interpretation:

- The model learns a useful but limited frame-level signal.
- The best validation ROC-AUC is `0.667`, which is moderate.
- Training loss keeps decreasing while validation quality does not improve,
  indicating quick overfitting or limited feature capacity.

## Validation Alert Metrics At Threshold 0.50

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 196 |
| True positive alerts | 99 |
| False positive alerts | 90 |
| Missed events | 13 |
| Alert precision | 0.524 |
| Alert recall | 0.884 |
| False alarm rate | 0.804 |
| Missed event rate | 0.116 |
| Mean alert time error | -13.738 s |

At threshold `0.50`, recall is high, but false alarms are far too high for a
product candidate.

## Threshold Sweep Summary

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: |
| 0.81 | 0.659 | 0.536 | 0.277 | -10.748 s |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: |
| 0.64 | 0.563 | 0.804 | 0.625 | -11.770 s |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: |
| 0.70 | 0.663 | 0.580 | 0.295 | -9.644 s |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: |
| 0.43 | 0.533 | 0.804 | 0.705 | -13.035 s |

### Additional temporal post-processing

Several stricter post-processing rules were tested:

- 3 consecutive frames;
- 4 consecutive frames;
- moving average over 3 seconds;
- moving average over 5 seconds;
- moving average over 3 seconds plus 2 consecutive frames.

These rules reduce false positives, but they also reduce recall. None reaches
the product gate.

Best examples:

| Method | Threshold | Precision | Recall | False alarm rate |
| --- | ---: | ---: | ---: | ---: |
| 3 consecutive frames | 0.64 | 0.678 | 0.545 | 0.259 |
| 4 consecutive frames | 0.56 | 0.629 | 0.500 | 0.295 |
| Moving average 3s | 0.75 | 0.727 | 0.500 | 0.188 |
| Moving average 5s | 0.69 | 0.687 | 0.509 | 0.232 |
| Moving average 3s + consecutive 2 | 0.69 | 0.675 | 0.500 | 0.241 |

## Score Distribution Diagnostic

The model separates positive and negative videos somewhat, but not enough for a
reliable alert threshold.

Video-level maximum score quantiles:

| Video target | 10% | 25% | 50% | 75% | 90% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.371 | 0.564 | 0.738 | 0.821 | 0.918 |
| 1 | 0.637 | 0.744 | 0.863 | 0.927 | 0.959 |

The overlap is large. Many negative videos contain at least one high-risk frame,
which causes false alerts when using temporal thresholding.

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

The full-dataset ResNet18 model cannot simultaneously achieve high recall and
acceptable false alarm rate on validation.

## Decision

Do not evaluate this model on holdout as a product candidate.

The validation tradeoff is not strong enough. The holdout should remain sealed
until a better candidate is selected on validation.

## Recommended Next Step

Move to a stronger visual backbone before returning to temporal neural models.

Recommended next experiment:

```text
product_temporal_alert_224_efficientnet_b0
```

Rationale:

- The current best approach is still frame-level scoring plus temporal
  post-processing.
- The full-dataset ResNet18 has insufficient separation between positive and
  negative video risk peaks.
- Previous GRU/LSTM variants did not improve the alert tradeoff.
- A stronger image encoder is the most direct next test.

Candidate backbones:

1. `efficientnet_b0` for the first product-grade upgrade.
2. `convnext_tiny` if EfficientNet-B0 is still insufficient.
3. `efficientnet_b3` if GPU memory and training time are acceptable.

## Article Interpretation

This is an important result for the article:

- the small-sample baseline did not scale to product-level validation;
- full-dataset evaluation exposed the false alarm problem more clearly;
- temporal post-processing improves precision but does not solve separability;
- the next methodological step is stronger representation learning, not just
  threshold tuning.
