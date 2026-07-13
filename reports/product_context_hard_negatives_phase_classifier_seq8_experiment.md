# Product context hard-negatives phase classifier seq8 experiment

Run date: 2026-07-13

## Objective

Test whether contextual hard negatives mined from train negative videos improve
the phase-classifier alert tradeoff.

Experiment:

```text
product_context_hard_negatives_phase_classifier_seq8
```

Best checkpoint:

```text
models/checkpoints/product_context_hard_negatives_phase_classifier_seq8_best_sequence.pt
```

## Data Setup

The model used:

```text
data/interim/product_event_windows_seq8_context_hard_negatives_manifest.csv
```

This manifest added 2202 contextual hard-negative windows to the previous
pre-alert phase manifest.

Training split distribution:

| Phase index | Windows |
| ---: | ---: |
| 0 | 11639 |
| 1 | 2104 |
| 2 | 387 |
| 3 | 1294 |

Validation split distribution was unchanged:

| Phase index | Windows |
| ---: | ---: |
| 0 | 2007 |
| 1 | 448 |
| 2 | 99 |
| 3 | 283 |

The holdout remained sealed.

## Training Result

Training stopped early at epoch 3. The best checkpoint was epoch 1.

| Epoch | Train loss | Phase macro F1 | Alert threshold | Alert precision | Alert recall | False alarm rate | Alert score | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.164 | 0.248 | 0.51 | 0.743 | 0.696 | 0.241 | 1.832 | Yes |
| 2 | 1.127 | 0.321 | 0.47 | - | 0.705 | 0.286 | 1.783 | No |
| 3 | 1.106 | 0.287 | 0.52 | - | 0.786 | 0.393 | 1.821 | No |

Best validation-window alert proxy:

| Metric | Value |
| --- | ---: |
| Alert threshold | 0.51 |
| Alert precision | 0.743 |
| Alert recall | 0.696 |
| False alarm rate | 0.241 |
| Missed event rate | 0.304 |
| Mean alert time error | -10.196 s |

Interpretation:

The contextual hard negatives reduced false alarms and improved precision in the
window-level proxy, but the model became too conservative and missed too many
positive videos.

## Full-Video Validation At Threshold 0.50

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 181 |
| True positive alerts | 91 |
| False positive alerts | 83 |
| Missed events | 21 |
| Alert precision | 0.523 |
| Alert recall | 0.812 |
| False alarm rate | 0.741 |
| Missed event rate | 0.188 |
| Mean predicted lead time | 16.412 s |
| Mean alert time error | -14.689 s |

At threshold `0.50`, the full-video behavior still has too many false alarms.

## Threshold Sweep Summary

No tested threshold or post-processing rule satisfied the product gate.

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.55 | 0.687 | 0.607 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.523 | 0.812 | 0.741 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.54 | 0.674 | 0.554 | 0.268 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.48 | 0.511 | 0.821 | 0.786 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.53 | 0.667 | 0.554 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.47 | 0.506 | 0.804 | 0.786 |

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

## Comparison With Previous Phase Classifier

Previous `product_event_window_phase_classifier_seq8` best low-false-alarm
point with 2 consecutive frames:

```text
precision 0.648
recall    0.509
FAR       0.277
```

Current contextual hard-negative model best low-false-alarm point with 2
consecutive frames:

```text
precision 0.674
recall    0.554
FAR       0.268
```

This is a modest improvement in the low-false-alarm region, but still far from
the required recall. At recall `>= 0.80`, both models still generate too many
false alarms.

## Decision

The mining direction is useful, but the current configuration is too blunt:
2202 extra negatives with weight `2.5` improved conservative operating points
but did not solve high-recall false alarms.

## Recommended Next Step

Run a lighter ablation instead of adding more capacity:

```text
product_context_hard_negatives_phase_classifier_seq8_weight_ablation
```

Recommended variants:

| Variant | Hard negatives | Weight | Purpose |
| --- | ---: | ---: | --- |
| A | 2202 | 1.5 | reduce over-penalization of alert classes |
| B | about 1000 | 2.0 | keep difficult negatives but reduce volume |
| C | about 1000 | 1.5 | most conservative data-pressure ablation |

The goal is to keep the false-alarm improvement while recovering recall.
