# Product context HN top1000 w15 phase classifier seq8 experiment

Run date: 2026-07-13

## Objective

Train the lightest contextual hard-negative ablation:

```text
product_context_hn_top1000_w15_phase_classifier_seq8
```

This variant uses:

- top 1000 contextual hard-negative windows;
- `sample_weight = 1.5`;
- four-class phase supervision;
- alert score `P(prealert_late) + P(event_near)`.

## Data

Manifest:

```text
data/interim/product_event_windows_seq8_context_hn_top1000_w15_manifest.csv
```

Train phase distribution:

| Phase index | Windows |
| ---: | ---: |
| 0 | 10437 |
| 1 | 2104 |
| 2 | 387 |
| 3 | 1294 |

Validation phase distribution:

| Phase index | Windows |
| ---: | ---: |
| 0 | 2007 |
| 1 | 448 |
| 2 | 99 |
| 3 | 283 |

The holdout remained sealed.

## Training Result

Training stopped early at epoch 4. The best checkpoint was epoch 2.

| Epoch | Train loss | Phase macro F1 | Alert threshold | Alert recall | False alarm rate | Alert score | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.263 | 0.242 | 0.56 | 0.777 | 0.402 | 1.683 | Yes |
| 2 | 1.220 | 0.258 | 0.59 | 0.750 | 0.330 | 1.931 | Yes |
| 3 | 1.193 | 0.286 | 0.55 | 0.804 | 0.446 | 1.607 | No |
| 4 | 1.169 | 0.286 | 0.55 | 0.714 | 0.357 | 1.429 | No |

Best validation-window alert proxy:

| Metric | Value |
| --- | ---: |
| Alert threshold | 0.59 |
| Alert precision | 0.694 |
| Alert recall | 0.750 |
| False alarm rate | 0.330 |
| Missed event rate | 0.250 |
| Mean alert time error | -9.496 s |

The proxy improved recall compared with the heavy hard-negative model, but it
still failed the product gate.

## Full-Video Validation At Threshold 0.50

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 217 |
| True positive alerts | 110 |
| False positive alerts | 106 |
| Missed events | 2 |
| Alert precision | 0.509 |
| Alert recall | 0.982 |
| False alarm rate | 0.946 |
| Missed event rate | 0.018 |
| Mean predicted lead time | 17.749 s |
| Mean alert time error | -16.083 s |

At threshold `0.50`, the model became too aggressive on full videos.

## Threshold Sweep Summary

No tested threshold or post-processing rule satisfied the product gate.

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.68 | 0.660 | 0.571 | 0.295 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.61 | 0.562 | 0.812 | 0.634 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.66 | 0.674 | 0.536 | 0.259 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.59 | 0.579 | 0.821 | 0.598 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.64 | 0.625 | 0.491 | 0.295 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.56 | 0.542 | 0.812 | 0.688 |

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

Do not evaluate this model on holdout.

## Comparison

Previous heavy contextual hard-negative model, best 2-consecutive low-FAR point:

```text
precision 0.674
recall    0.554
FAR       0.268
```

Current top1000 w15 model, best 2-consecutive low-FAR point:

```text
precision 0.674
recall    0.536
FAR       0.259
```

This ablation did not improve the best product operating point. It recovered
high recall at lower thresholds, but those thresholds still caused too many
false alarms.

## Decision

The lightest hard-negative ablation is not a product candidate.

Next ablation to try:

```text
product_context_hn_top1000_w20_phase_classifier_seq8
```

Reason:

Variant C reduced pressure too much and became over-sensitive at threshold
`0.50`. Variant B keeps the same reduced volume but increases hard-negative
weight from `1.5` to `2.0`, which may better balance recall and false alarms.
