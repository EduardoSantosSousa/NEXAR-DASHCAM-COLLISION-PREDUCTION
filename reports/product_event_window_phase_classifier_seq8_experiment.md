# Product event-window phase classifier seq8 experiment

Run date: 2026-07-13

## Objective

Test whether predicting temporal phases directly improves alert calibration over
binary event-window supervision.

Experiment:

```text
product_event_window_phase_classifier_seq8
```

Best checkpoint:

```text
models/checkpoints/product_event_window_phase_classifier_seq8_best_sequence.pt
```

## Model Setup

The model used the existing ResNet18 frozen encoder plus GRU temporal head, but
changed the target from binary alert/no-alert to four phase classes:

| Class | Phase | Alert use |
| ---: | --- | --- |
| 0 | `negative_video` / `positive_safe` | no alert |
| 1 | `prealert_early` | no alert |
| 2 | `prealert_late` | alert |
| 3 | `event_near` | alert |

The product alert score was:

```text
P(class 2) + P(class 3)
```

Training command used:

```text
--num-classes 4
--target-column phase_index
--alert-class-indices 2,3
--imbalance-strategy class_weight
--sample-weight-column sample_weight
--alert-metric-selection
```

## Training Result

Training stopped early at epoch 6. The best checkpoint was epoch 4.

| Epoch | Train loss | Phase macro F1 | Alert threshold | Alert precision | Alert recall | False alarm rate | Alert score | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.277 | 0.236 | 0.58 | - | 0.705 | 0.295 | 1.759 | Yes |
| 2 | 1.227 | 0.225 | 0.66 | - | 0.741 | 0.339 | 1.790 | Yes |
| 3 | 1.195 | 0.263 | 0.64 | - | 0.741 | 0.348 | 1.729 | No |
| 4 | 1.166 | 0.286 | 0.52 | 0.707 | 0.732 | 0.304 | 1.946 | Yes |
| 5 | 1.143 | 0.297 | 0.46 | - | 0.830 | 0.455 | 1.652 | No |
| 6 | 1.116 | 0.279 | 0.47 | - | 0.786 | 0.384 | 1.881 | No |

Best validation-window alert proxy:

| Metric | Value |
| --- | ---: |
| Alert threshold | 0.52 |
| Alert precision | 0.707 |
| Alert recall | 0.732 |
| False alarm rate | 0.304 |
| Missed event rate | 0.268 |
| Mean alert time error | -7.814 s |

The best proxy nearly reached the false-alarm and precision constraints, but
recall remained below the product gate.

## Full-Video Validation At Threshold 0.50

The best checkpoint was evaluated on full validation videos using:

```text
outputs/predictions/product_event_window_phase_classifier_seq8_temporal_risk_scores.csv
```

| Metric | Value |
| --- | ---: |
| Videos | 224 |
| Positive videos | 112 |
| Negative videos | 112 |
| Predicted alerts | 187 |
| True positive alerts | 99 |
| False positive alerts | 83 |
| Missed events | 13 |
| Alert precision | 0.544 |
| Alert recall | 0.884 |
| False alarm rate | 0.741 |
| Missed event rate | 0.116 |
| Mean predicted lead time | 15.505 s |
| Mean alert time error | -13.775 s |

At threshold `0.50`, the model remains too aggressive on full videos.

## Threshold Sweep Summary

No tested threshold or post-processing rule satisfied the product gate.

### Raw scores

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.71 | 0.562 | 0.366 | 0.286 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.56 | 0.569 | 0.812 | 0.616 |

### Two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.64 | 0.648 | 0.509 | 0.277 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.51 | 0.562 | 0.804 | 0.625 |

### Moving average 3s + two consecutive frames

Best point with `false_alarm_rate <= 0.30`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.59 | 0.610 | 0.446 | 0.286 |

Lowest false alarm rate while keeping `recall >= 0.80`:

| Threshold | Precision | Recall | False alarm rate |
| ---: | ---: | ---: | ---: |
| 0.47 | 0.576 | 0.812 | 0.598 |

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

No holdout evaluation should be run for this model.

## Interpretation

The phase classifier is conceptually cleaner than the binary phase target, but
it did not improve the full-video product tradeoff. Compared with the binary
pre-alert phase model, the best low-false-alarm operating point had lower
recall.

The main pattern is still the same:

- when recall is high, false alarms are too frequent;
- when false alarms are controlled, recall drops too much;
- thresholding and light temporal smoothing are not enough to solve this alone.

## Recommended Next Step

Do not escalate the same GRU head again without diagnosing false positives.

Next recommended experiment:

```text
product_false_positive_error_analysis
```

Goal:

- identify which negative videos repeatedly trigger high `P(prealert_late) +
  P(event_near)`;
- inspect whether false positives are caused by visual ambiguity, phase-label
  timing, camera motion, traffic density, or sampling bias;
- use that evidence to decide between stronger temporal context, better
  negative mining, calibration, or label-window changes.
