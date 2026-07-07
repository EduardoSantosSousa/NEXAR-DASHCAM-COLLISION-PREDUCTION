# Temporal Alert Evaluation

## Objective

This stage evaluates whether the frame-level baseline can be used as a temporal
alert system. Instead of producing only one prediction per frame or video, the
model is applied across each sampled video at 1 FPS to generate a risk curve:

```text
video timestamp -> risk_score
```

The first timestamp whose `risk_score` is greater than or equal to a threshold is
treated as the predicted alert time:

```text
predicted_alert_time = first timestamp where risk_score >= threshold
```

## Generated Artifacts

| Artifact | Location |
| --- | --- |
| Temporal risk scores | `outputs/predictions/temporal_risk_scores.csv` |
| Alert predictions | `outputs/predictions/alert_predictions.csv` |
| Alert metrics | `models/reports/alert_metrics.json` |
| Threshold sweep | `models/reports/alert_threshold_sweep.csv` |
| Threshold sweep figure | `outputs/figures/alert_threshold_sweep.png` |
| Example risk curves | `outputs/figures/temporal_risk_curve_*.png` |

## KPI Definitions

| KPI | Meaning |
| --- | --- |
| Alert precision | Fraction of predicted alerts that correspond to positive videos rather than false alarms. |
| Alert recall | Fraction of positive videos where the model emitted an alert before or at the event. |
| False alarm rate | Fraction of negative videos where the model emitted an alert. |
| Missed event rate | Fraction of positive videos with no valid alert before the event. |
| Predicted lead time | `time_of_event - predicted_alert_time`. Higher means earlier alert. |
| Alert time error | `predicted_alert_time - time_of_alert`. Negative values mean the model alerts earlier than the dataset alert timestamp. |

## Initial Results

The first run used:

- Sample: 100 videos.
- Sampling rate: 1 FPS.
- Device: CUDA.
- Model: baseline ResNet18 checkpoint.

At threshold `0.50`:

| KPI | Value |
| --- | ---: |
| Alert precision | 0.538 |
| Alert recall | 0.980 |
| False alarm rate | 0.840 |
| Missed event rate | 0.020 |
| Mean predicted lead time | 17.190 s |
| Mean alert time error | -15.606 s |

This operating point detects almost all positive videos but produces many false
alarms and alerts much earlier than the annotated alert timestamp.

## Threshold Trade-Off

| Threshold | Precision | Recall | False Alarm Rate | Missed Event Rate | Mean Lead Time | Mean Alert Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.538 | 0.980 | 0.840 | 0.020 | 17.190 | -15.606 |
| 0.70 | 0.554 | 0.920 | 0.740 | 0.080 | 16.813 | -15.188 |
| 0.90 | 0.618 | 0.840 | 0.520 | 0.160 | 16.139 | -14.460 |
| 0.95 | 0.635 | 0.800 | 0.460 | 0.200 | 14.654 | -13.010 |
| 0.99 | 0.718 | 0.560 | 0.220 | 0.440 | 13.198 | -11.674 |

## Interpretation

The current model can generate temporal risk curves, but it is not yet a reliable
alert system. The main limitation is calibration: many videos cross the alert
threshold too early, including negative examples. This behavior is expected from
the current baseline because it was trained with video-level labels assigned to
individual frames, not with temporally precise frame labels.

The next scientific step is to train a temporally aware model or relabel frames
according to their distance from `time_of_alert` and `time_of_event`.
