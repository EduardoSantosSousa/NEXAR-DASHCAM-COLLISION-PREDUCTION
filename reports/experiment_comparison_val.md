# Experiment Comparison on Validation Split

## Objective

This report compares the two most recent temporal alert experiments using the
fixed video-level validation split:

- `temporal_alert_224_split`: ResNet18 trained from scratch.
- `temporal_alert_224_pretrained`: ResNet18 initialized with ImageNet weights.

The goal is to check whether transfer learning improved the temporal alert
behavior after correcting the evaluation protocol to use only validation videos.

## Evaluation Protocol

The split file is:

```text
data/interim/sample_100_videos_splits.csv
```

Validation set:

| Split | Target | Videos |
| --- | ---: | ---: |
| val | 0 | 10 |
| val | 1 | 10 |

Training set:

| Split | Target | Videos |
| --- | ---: | ---: |
| train | 0 | 40 |
| train | 1 | 40 |

Both models were trained on the same temporally labeled frame manifest:

```text
data/interim/temporal_frames_224_manifest.csv
```

Both temporal alert evaluations used:

| Setting | Value |
| --- | ---: |
| Evaluation split | `val` |
| Videos evaluated | 20 |
| Positive videos | 10 |
| Negative videos | 10 |
| Sampling rate | 1 FPS |
| Default threshold | 0.50 |

## Generated Artifacts

| Experiment | Artifact | Path |
| --- | --- | --- |
| Scratch | Training metrics | `models/reports/temporal_alert_224_split_metrics.json` |
| Scratch | Alert metrics | `models/reports/temporal_alert_224_split_alert_metrics.json` |
| Scratch | Threshold sweep | `models/reports/temporal_alert_224_split_alert_threshold_sweep.csv` |
| Scratch | Risk scores | `outputs/predictions/temporal_alert_224_split_temporal_risk_scores.csv` |
| Pretrained | Training metrics | `models/reports/temporal_alert_224_pretrained_metrics.json` |
| Pretrained | Alert metrics | `models/reports/temporal_alert_224_pretrained_alert_metrics.json` |
| Pretrained | Threshold sweep | `models/reports/temporal_alert_224_pretrained_alert_threshold_sweep.csv` |
| Pretrained | Risk scores | `outputs/predictions/temporal_alert_224_pretrained_temporal_risk_scores.csv` |

## Frame-Level Validation Results

| Model | Pretrained | Epochs | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_split` | No | 3 | 0.792 | 0.083 | 0.216 | 0.120 | 0.501 |
| `temporal_alert_224_pretrained` | Yes | 8 | 0.908 | 0.082 | 0.039 | 0.053 | 0.565 |

Frame-level interpretation:

- The pretrained model improved ROC-AUC from `0.501` to `0.565`.
- Accuracy increased, but this is partly caused by class imbalance: most frames
  are non-alert frames.
- Recall and F1 dropped in the final pretrained epoch, meaning the model became
  more conservative and missed more positive temporal frames.
- The pretrained run had better intermediate ROC-AUC around epochs 2 and 3
  (`0.604` and `0.600`), so the final epoch may not be the best checkpoint.

## Alert Results at Threshold 0.50

| Model | Threshold | Predicted alerts | True positive alerts | False positive alerts | Missed events | Precision | Recall | False alarm rate | Missed event rate | Mean lead time | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_split` | 0.50 | 10 | 4 | 5 | 6 | 0.444 | 0.400 | 0.500 | 0.600 | 19.953 s | -17.960 s |
| `temporal_alert_224_pretrained` | 0.50 | 8 | 3 | 3 | 7 | 0.500 | 0.300 | 0.300 | 0.700 | 12.011 s | -9.774 s |

Threshold `0.50` interpretation:

- The pretrained model reduced false alarm rate from `0.500` to `0.300`.
- Alert precision improved from `0.444` to `0.500`.
- Mean alert error improved from `-17.960 s` to `-9.774 s`, so alerts became
  less extremely early.
- Recall dropped from `0.400` to `0.300`, and missed event rate increased from
  `0.600` to `0.700`.

At the default threshold, transfer learning produced a more conservative model:
fewer false alarms, but also fewer detected positive events.

## Threshold Sweep With Recall Constraints

The automatic candidate from the sweep can be misleading because it may choose a
threshold that emits no alerts. A threshold with zero alerts has zero false
alarms, but it is not useful for collision anticipation.

The table below chooses operating points by applying a minimum recall constraint
first, then minimizing false alarm rate.

| Model | Minimum recall constraint | Chosen threshold | Precision | Recall | False alarm rate | Missed event rate | Mean lead time | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_split` | 0.70 | 0.19 | 0.467 | 0.700 | 0.800 | 0.300 | 16.267 s | -14.352 s |
| `temporal_alert_224_pretrained` | 0.70 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `temporal_alert_224_split` | 0.60 | 0.31 | 0.462 | 0.600 | 0.700 | 0.400 | 15.908 s | -13.836 s |
| `temporal_alert_224_pretrained` | 0.60 | 0.10 | 0.600 | 0.600 | 0.400 | 0.400 | 14.118 s | -12.338 s |
| `temporal_alert_224_split` | 0.50 | 0.38 | 0.455 | 0.500 | 0.600 | 0.500 | 18.682 s | -16.876 s |
| `temporal_alert_224_pretrained` | 0.50 | 0.18 | 0.625 | 0.500 | 0.300 | 0.500 | 13.856 s | -11.916 s |
| `temporal_alert_224_split` | 0.40 | 0.70 | 0.667 | 0.400 | 0.200 | 0.600 | 17.203 s | -15.210 s |
| `temporal_alert_224_pretrained` | 0.40 | 0.18 | 0.625 | 0.500 | 0.300 | 0.500 | 13.856 s | -11.916 s |
| `temporal_alert_224_split` | 0.30 | 0.89 | 1.000 | 0.300 | 0.000 | 0.700 | 16.126 s | -14.394 s |
| `temporal_alert_224_pretrained` | 0.30 | 0.54 | 0.600 | 0.300 | 0.200 | 0.700 | 12.011 s | -9.774 s |

Sweep interpretation:

- If the minimum acceptable recall is `0.70`, only the scratch model reaches it,
  but with a high false alarm rate of `0.800`.
- If the minimum acceptable recall is `0.60`, the pretrained model is better:
  it keeps the same recall and reduces false alarm rate from `0.700` to `0.400`.
- If the minimum acceptable recall is `0.50`, the pretrained model is clearly
  better: precision increases from `0.455` to `0.625`, false alarm rate drops
  from `0.600` to `0.300`, and alert error becomes less negative.
- If recall can be as low as `0.30`, both models miss many events. The scratch
  model can reach zero false alarms, but only detects 3 of 10 positive videos.

## Did Transfer Learning Improve the Model?

Short answer: partially, yes, but not enough to call it a strong final model.

What improved:

- Better ROC-AUC at frame level: `0.501` to `0.565`.
- Lower false alarm rate at threshold `0.50`: `0.500` to `0.300`.
- Better alert precision at threshold `0.50`: `0.444` to `0.500`.
- Alerts became less extremely early: mean alert error improved from
  `-17.960 s` to `-9.774 s`.
- Under matched recall constraints of `0.50` or `0.60`, the pretrained model
  has better precision and lower false alarm rate.

What got worse:

- Recall at threshold `0.50` dropped from `0.400` to `0.300`.
- Missed event rate increased from `0.600` to `0.700`.
- Final frame-level F1 dropped from `0.120` to `0.053`.
- The pretrained model does not reach recall `0.70` at any tested threshold.

Conclusion:

Transfer learning improved calibration and reduced false alarms, but it made the
model more conservative. The most defensible statement is:

```text
The pretrained ResNet18 improved false alarm control and temporal calibration,
especially at matched recall thresholds, but it did not improve event recall.
The current model is therefore better calibrated but still not reliable enough
as a final collision alert system.
```

## Recommended Operating Point

For the current validation results, a reasonable operating point is:

| Model | Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_pretrained` | 0.18 | 0.625 | 0.500 | 0.300 | 0.500 | -11.916 s |

Why:

- It detects 5 of 10 positive validation videos.
- It keeps false alarms to 3 of 10 negative validation videos.
- It improves precision and false alarm rate compared with the scratch model at
  the same recall level.
- Its alerts are still early, but less extreme than the scratch model.

This should not be treated as a production threshold. It is only the best
current operating point on a small validation sample.

## Best-Checkpoint Follow-Up

Best-checkpoint selection and early stopping were added after this comparison.
The training script now saves both the final checkpoint and the best checkpoint
according to validation ROC-AUC.

New artifacts:

| Experiment | Artifact |
| --- | --- |
| Scratch best checkpoint | `models/checkpoints/temporal_alert_224_split_best_resnet18.pt` |
| Pretrained best checkpoint | `models/checkpoints/temporal_alert_224_pretrained_best_resnet18.pt` |
| Follow-up report | `reports/best_checkpoint_experiment.md` |
| Review notebook | `notebooks/08_best_checkpoint_mlflow_review.ipynb` |

At threshold `0.50`, the scratch best checkpoint increased alert recall to
`0.800`, but false alarm rate also increased to `0.700`. The pretrained best
checkpoint stayed conservative at threshold `0.50`, with recall `0.300` and
false alarm rate `0.300`.

The stronger result appears after threshold tuning:

| Model | Minimum recall constraint | Threshold | Precision | Recall | False alarm rate | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_split_best` | 0.70 | 0.65 | 0.636 | 0.700 | 0.400 | -14.495 s |
| `temporal_alert_224_pretrained_best` | 0.70 | 0.22 | 0.667 | 0.800 | 0.400 | -8.116 s |

This means best-checkpoint selection materially improves the pretrained
experiment: it reaches higher recall at the same false alarm rate and alerts less
extremely early than the scratch best checkpoint.

## Limitations

- Validation set is small: only 20 videos.
- Positive temporal frames are heavily imbalanced against non-alert frames.
- The model still classifies frames independently, without sequence context.
- The pretrained model was evaluated using the final epoch, but intermediate
  epochs showed stronger frame-level ROC-AUC.
- Threshold tuning was performed on the validation split, so another holdout or
  cross-validation protocol would be needed for a stronger claim.

## Next Steps

1. Add checkpoint selection by best validation ROC-AUC or best validation F1.
2. Add early stopping to avoid keeping an over-conservative final epoch.
3. Try focal loss or stronger positive-frame sampling for the temporal labels.
4. Evaluate short-window temporal aggregation before implementing GRU/LSTM.
5. Implement CNN + GRU/LSTM only after the baseline uses best-checkpoint
   selection and a documented threshold criterion.
6. Update `reports/methodology.md` with the corrected validation-only protocol.
7. Update `reports/temporal_label_progression.md` with the comparison table from
   this report.
