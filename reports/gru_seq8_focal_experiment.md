# GRU seq8 focal experiment

Run date: 2026-07-09

## Objective

Test whether focal loss improves the GRU sequence model after the
`temporal_alert_224_pretrained_gru_seq8_sampler` experiment failed to beat the
current alert baseline.

The experiment name is:

```text
temporal_alert_224_pretrained_gru_seq8_focal
```

This run keeps the stronger sequence setup:

- sequence length 8;
- ImageNet-pretrained ResNet18 encoder;
- partial CNN fine-tuning with `cnn_train_policy=layer4`;
- separate CNN/head learning rates.

It changes the imbalance handling:

- from `weighted_sampler + cross_entropy`;
- to `class_weight + focal`.

It also changes checkpoint selection from ROC-AUC to validation F1, because the
previous sequence run improved frame-level ranking but did not improve the alert
operating point.

## Configuration

| Setting | Value |
| --- | --- |
| Manifest | `data/interim/temporal_frames_224_manifest.csv` |
| Split manifest | `data/interim/sample_100_videos_splits.csv` |
| Evaluation split | `val` |
| Train videos | 80 |
| Validation videos | 20 |
| Sequence length | 8 |
| RNN | GRU |
| Hidden size | 128 |
| CNN encoder | ResNet18 ImageNet pretrained |
| CNN train policy | `layer4` |
| Imbalance strategy | `class_weight` |
| Loss | `focal` |
| Focal gamma | 2.0 |
| Batch size | 8 |
| Epochs | 6 |
| Head learning rate | 0.0001 |
| CNN learning rate | 0.00001 |
| Monitor metric | validation F1 |
| Early stopping patience | 2 |

## Baseline To Beat

The current best operating point remains:

| Model | Post-processing | Threshold | Precision | Recall | False alarm rate | Mean alert error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_pretrained_best` | 2 consecutive frames | 0.13 | 0.727 | 0.800 | 0.300 | -7.991 s |

Success criteria:

1. `recall >= 0.80` and `false_alarm_rate < 0.30`;
2. `recall >= 0.80` and `precision > 0.727`;
3. `recall >= 0.70` and `false_alarm_rate <= 0.20`.

## Training Results

Training stopped early at epoch 3. The best checkpoint was epoch 1.

| Metric | Value |
| --- | ---: |
| Best epoch | 1 |
| Best validation F1 | 0.178 |
| Best validation ROC-AUC | 0.530 |
| Best frame precision | 0.295 |
| Best frame recall | 0.127 |
| Final frame F1 | 0.113 |
| Final frame ROC-AUC | 0.570 |
| Train sequences | 3130 |
| Validation sequences | 1551 |
| Trainable parameters | 8640514 |
| Trainable CNN parameters | 8393728 |
| Trainable head parameters | 246786 |

Training history:

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Selected best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.180 | 0.178 | 0.530 | Yes |
| 2 | 0.128 | 0.130 | 0.546 | No |
| 3 | 0.091 | 0.113 | 0.570 | No |

Interpretation:

- Focal loss produced a higher best frame F1 than the previous seq8 sampler run.
- However, the best F1 checkpoint has weak ROC-AUC and weak recall.
- Later epochs improved ROC-AUC but reduced F1, which triggered early stopping.

## Alert Evaluation At Threshold 0.50

| Metric | Value |
| --- | ---: |
| Videos | 20 |
| Positive videos | 10 |
| Negative videos | 10 |
| Predicted alerts | 5 |
| True positive alerts | 2 |
| False positive alerts | 3 |
| Missed events | 8 |
| Alert precision | 0.400 |
| Alert recall | 0.200 |
| False alarm rate | 0.300 |
| Missed event rate | 0.800 |
| Mean alert error | -13.374 s |

At threshold `0.50`, the model is still too conservative for recall and too weak
to replace the baseline.

## Threshold Sweep Results

### Raw Scores

Best point with false alarm rate at or below the current baseline ceiling
(`<= 0.30`):

| Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.57 | 1.000 | 0.200 | 0.000 | 0.800 | -8.874 s |
| 0.54 | 0.667 | 0.200 | 0.100 | 0.800 | -9.374 s |
| 0.53 | 0.500 | 0.200 | 0.200 | 0.800 | -9.374 s |
| 0.51 | 0.500 | 0.200 | 0.200 | 0.800 | -13.374 s |

Best high-recall points:

| Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.34 | 0.500 | 0.800 | 0.800 | 0.200 | -10.671 s |
| 0.33 | 0.526 | 1.000 | 0.900 | 0.000 | -12.028 s |
| 0.26 | 0.500 | 1.000 | 1.000 | 0.000 | -14.828 s |

### Two Consecutive Frames

Best point with false alarm rate at or below the current baseline ceiling
(`<= 0.30`):

| Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.56 | 1.000 | 0.200 | 0.000 | 0.800 | -8.624 s |
| 0.53 | 0.667 | 0.200 | 0.100 | 0.800 | -8.874 s |
| 0.52 | 0.500 | 0.200 | 0.200 | 0.800 | -8.874 s |
| 0.49 | 0.400 | 0.200 | 0.300 | 0.800 | -13.124 s |

Best high-recall points:

| Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.32 | 0.556 | 1.000 | 0.800 | 0.000 | -11.028 s |
| 0.30 | 0.526 | 1.000 | 0.900 | 0.000 | -12.228 s |
| 0.24 | 0.500 | 1.000 | 1.000 | 0.000 | -14.378 s |

## Score Calibration Diagnostic

The focal model produced a narrow score distribution:

| Group | Mean risk score | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| All validation frames | 0.335 | 0.159 | 0.587 |
| Positive frame targets | 0.335 | 0.173 | 0.587 |
| Negative frame targets | 0.335 | 0.159 | 0.543 |

The mean risk scores for positive and negative frame targets are nearly
identical. This explains the alert tradeoff:

- high thresholds produce very few alerts and miss most positive videos;
- lower thresholds recover recall but also trigger many negative videos;
- the model does not separate positive and negative temporal regions well enough.

## Verdict

This experiment is useful, but it is not good enough to replace the current
baseline.

It improves over the seq8 sampler in one narrow way:

- it can produce precision `1.0` at low recall and zero false alarms.

But it fails the project criteria:

- with `false_alarm_rate <= 0.30`, recall is only `0.20`;
- at recall `0.80`, false alarm rate is `0.80`;
- at recall `1.00`, false alarm rate is between `0.80` and `1.00`;
- it does not beat the current baseline recall/false-alarm tradeoff.

Decision:

```text
Do not replace temporal_alert_224_pretrained_best + consecutive2.
Do not continue iterating on this exact GRU seq8 setup as the main priority.
```

## Updated Recommendation

The project has now tested:

- independent frame baseline;
- best checkpoint selection;
- temporal post-processing;
- CNN + GRU/LSTM seq4;
- GRU seq8 with weighted sampler;
- GRU seq8 with focal loss and F1 checkpointing.

The strongest result is still the pretrained frame model with simple temporal
post-processing.

Recommended next priority:

1. Freeze modeling work for the current small validation split.
2. Build a stronger evaluation protocol:
   - holdout split if more videos are available;
   - or video-level cross-validation.
3. Update the comparison notebook from MLflow and local artifacts.
4. Build the Streamlit dashboard around the current best baseline.
5. Use the sequence results as a future-work/negative-result section in the
   report or article.

If modeling must continue, the next meaningful change should not be another
minor GRU hyperparameter tweak. A more useful modeling direction would be
window sampling centered around pre-alert/event regions, with the checkpoint
selected by video-level alert metrics instead of frame-level metrics.

## Artifacts

Primary artifacts:

```text
models/reports/temporal_alert_224_pretrained_gru_seq8_focal_sequence_metrics.json
models/reports/temporal_alert_224_pretrained_gru_seq8_focal_alert_metrics.json
models/reports/temporal_alert_224_pretrained_gru_seq8_focal_alert_threshold_sweep.csv
models/reports/temporal_alert_224_pretrained_gru_seq8_focal_consecutive2_alert_threshold_sweep.csv
outputs/predictions/temporal_alert_224_pretrained_gru_seq8_focal_sequence_predictions.csv
outputs/predictions/temporal_alert_224_pretrained_gru_seq8_focal_temporal_risk_scores.csv
```

Diagnostic low-threshold artifacts:

```text
models/reports/temporal_alert_224_pretrained_gru_seq8_focal_low_threshold_alert_threshold_sweep.csv
models/reports/temporal_alert_224_pretrained_gru_seq8_focal_consecutive2_low_threshold_alert_threshold_sweep.csv
```

## Caveats

- The validation set has only 20 videos.
- Thresholds are selected on the same validation split.
- The sequence models may still improve with different window sampling and a
  video-level objective.
- Current evidence is strong enough to avoid more small GRU/LSTM tweaks before
  improving evaluation quality.
