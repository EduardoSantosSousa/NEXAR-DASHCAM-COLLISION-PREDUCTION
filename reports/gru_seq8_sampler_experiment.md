# GRU seq8 sampler experiment

Run date: 2026-07-09

## Objective

Test whether a longer causal sequence model with partial CNN fine-tuning and
sequence-level weighted sampling can improve the temporal collision alert
baseline.

The experiment name is:

```text
temporal_alert_224_pretrained_gru_seq8_sampler
```

This experiment was designed to test three changes over the first GRU/LSTM
sequence runs:

- increase sequence length from 4 to 8 frames;
- fine-tune only `layer4` of the ImageNet-pretrained ResNet18 encoder;
- use `WeightedRandomSampler` to reduce the effect of positive/negative
  imbalance during sequence training.

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
| Imbalance strategy | `weighted_sampler` |
| Loss | `cross_entropy` |
| Batch size | 8 |
| Epochs | 6 |
| Head learning rate | 0.0001 |
| CNN learning rate | 0.00001 |
| Monitor metric | validation ROC-AUC |

The validation split has 10 positive videos and 10 negative videos.

## Baseline To Beat

The current best operating point remains:

| Model | Post-processing | Threshold | Precision | Recall | False alarm rate | Mean alert error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_pretrained_best` | 2 consecutive frames | 0.13 | 0.727 | 0.800 | 0.300 | -7.991 s |

For this project stage, a new experiment is good enough to replace the baseline
only if it improves the video-level alert tradeoff, not just frame-level metrics.

Success criteria:

1. `recall >= 0.80` and `false_alarm_rate < 0.30`;
2. `recall >= 0.80` and `precision > 0.727`;
3. `recall >= 0.70` and `false_alarm_rate <= 0.20`.

## Training Results

The training completed successfully on CUDA.

| Metric | Value |
| --- | ---: |
| Best epoch | 6 |
| Best validation ROC-AUC | 0.612 |
| Final frame accuracy | 0.926 |
| Final frame precision | 0.158 |
| Final frame recall | 0.029 |
| Final frame F1 | 0.050 |
| Train sequences | 3130 |
| Validation sequences | 1551 |
| Trainable parameters | 8640514 |
| Trainable CNN parameters | 8393728 |
| Trainable head parameters | 246786 |

Interpretation:

- The frame-level ROC-AUC improved compared with the first frozen GRU seq4 run
  (`0.567`) and the previous pretrained frame baseline best checkpoint
  (`0.599`).
- However, the thresholded frame-level recall and F1 are very low at the default
  decision threshold.
- The model ranks positives somewhat better, but the score calibration is not
  yet useful enough for robust alerting.

## Alert Evaluation At Threshold 0.50

| Metric | Value |
| --- | ---: |
| Videos | 20 |
| Positive videos | 10 |
| Negative videos | 10 |
| Predicted alerts | 3 |
| True positive alerts | 1 |
| False positive alerts | 1 |
| Missed events | 9 |
| Alert precision | 0.500 |
| Alert recall | 0.100 |
| False alarm rate | 0.100 |
| Missed event rate | 0.900 |
| Mean alert error | -16.633 s |

At threshold `0.50`, the model is too conservative. It misses 9 of the 10
positive validation videos.

## Default Threshold Sweep

The default sweep uses thresholds from `0.10` to `0.99`.

Best raw operating points observed:

| Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.16 | 0.750 | 0.300 | 0.100 | 0.700 | -7.774 s |
| 0.13 | 0.750 | 0.300 | 0.100 | 0.700 | -7.941 s |
| 0.12 | 0.600 | 0.300 | 0.200 | 0.700 | -11.274 s |
| 0.10 | 0.500 | 0.300 | 0.300 | 0.700 | -11.274 s |

With 2 consecutive frames:

| Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.13 | 0.667 | 0.200 | 0.100 | 0.800 | -6.895 s |
| 0.12 | 0.667 | 0.200 | 0.100 | 0.800 | -6.895 s |
| 0.10 | 0.667 | 0.200 | 0.100 | 0.800 | -6.895 s |
| 0.50 | 0.500 | 0.100 | 0.100 | 0.900 | -16.133 s |

Interpretation:

- Under the standard sweep range, raw scores only reach recall `0.30`.
- The 2-consecutive-frame rule reduces recall further, reaching only `0.20`.
- This does not meet any success criterion for replacing the current baseline.

## Expanded Low-Threshold Diagnostic Sweep

Because the default sweep starts at `0.10`, an additional diagnostic sweep was
generated from `0.01` to `0.99` to check whether the model was simply
under-calibrated.

Raw scores:

| Minimum recall target | Selected threshold | Precision | Recall | False alarm rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.70 | 0.01 | 0.500 | 0.800 | 0.800 | -8.370 s |
| 0.60 | 0.02 | 0.462 | 0.600 | 0.700 | -9.486 s |
| 0.50 | 0.04 | 0.455 | 0.500 | 0.600 | -9.597 s |
| 0.40 | 0.06 | 0.500 | 0.400 | 0.400 | -12.318 s |
| 0.30 | 0.16 | 0.750 | 0.300 | 0.100 | -7.774 s |

With 2 consecutive frames:

| Minimum recall target | Selected threshold | Precision | Recall | False alarm rate | Mean alert error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.03 | 0.500 | 0.500 | 0.500 | -8.397 s |
| 0.40 | 0.04 | 0.500 | 0.400 | 0.400 | -8.818 s |
| 0.30 | 0.09 | 0.750 | 0.300 | 0.100 | -7.608 s |

Interpretation:

- The raw model can reach recall `0.80`, but only at threshold `0.01`.
- At that operating point, false alarm rate rises to `0.80`, which is far worse
  than the current baseline false alarm rate of `0.30`.
- The consecutive-frame rule does not rescue the model; it caps useful recall at
  `0.50` in the expanded sweep.
- This suggests the model has some ranking signal but poor alert-level
  separability and calibration.

## Verdict

This experiment is useful, but it is not good enough to replace the current
baseline.

It is good enough as an engineering and research milestone because:

- the new sequence training path runs end to end;
- partial CNN fine-tuning works;
- sequence-level weighted sampling works;
- MLflow/artifacts are generated correctly;
- frame-level ROC-AUC improved to `0.612`.

It is not good enough as the selected alert model because:

- at the current baseline recall level (`0.80`), false alarm rate is `0.80`
  instead of `0.30`;
- the best low-false-alarm operating point has recall only `0.30`;
- the default threshold `0.50` misses 9 of 10 positive validation videos;
- the 2-consecutive-frame rule makes recall worse for this model.

Decision:

```text
Do not replace temporal_alert_224_pretrained_best + consecutive2.
Keep temporal_alert_224_pretrained_gru_seq8_sampler as an informative failed
experiment.
```

## Why This Happened

The most likely explanation is that optimizing checkpoint selection by
frame-level ROC-AUC improved ranking but did not optimize the alert operating
point that matters for the project.

The training curve supports this:

- ROC-AUC improves from epoch 1 to epoch 6;
- frame F1 collapses to zero for several epochs;
- the selected best checkpoint is the ROC-AUC-best epoch, but it is very
  conservative at normal thresholds.

In other words, the model can rank some risky frames above normal frames, but it
does not produce a clean enough score distribution for video-level alerts.

## Recommended Next Step

Do not keep pushing the same configuration blindly.

Recommended next experiment:

```text
temporal_alert_224_pretrained_gru_seq8_focal
```

Suggested changes:

- use focal loss with class weights;
- keep `sequence_length=8`;
- keep `cnn_train_policy=layer4`;
- lower the CNN learning rate if training remains unstable;
- monitor `f1` or run checkpoint selection from a metric closer to alert
  usefulness, instead of only ROC-AUC;
- keep the expanded threshold sweep from `0.01` to `0.99` for sequence models.

Candidate command:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py `
  --manifest data\interim\temporal_frames_224_manifest.csv `
  --split-manifest data\interim\sample_100_videos_splits.csv `
  --experiment-name temporal_alert_224_pretrained_gru_seq8_focal `
  --pretrained `
  --sequence-length 8 `
  --train-sequence-stride 2 `
  --val-sequence-stride 1 `
  --rnn-type gru `
  --epochs 6 `
  --batch-size 8 `
  --learning-rate 0.0001 `
  --head-learning-rate 0.0001 `
  --cnn-learning-rate 0.00001 `
  --hidden-size 128 `
  --num-workers 0 `
  --cnn-train-policy layer4 `
  --imbalance-strategy class_weight `
  --loss-name focal `
  --focal-gamma 2.0 `
  --monitor-metric f1 `
  --monitor-mode max `
  --patience 2
```

After evaluation, compare again against:

```text
precision = 0.727
recall = 0.800
false_alarm_rate = 0.300
mean_alert_time_error = -7.991 s
```

## Artifacts

Primary artifacts:

```text
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_sequence_metrics.json
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_alert_metrics.json
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_alert_threshold_sweep.csv
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_consecutive2_alert_threshold_sweep.csv
outputs/predictions/temporal_alert_224_pretrained_gru_seq8_sampler_sequence_predictions.csv
outputs/predictions/temporal_alert_224_pretrained_gru_seq8_sampler_temporal_risk_scores.csv
```

Diagnostic low-threshold artifacts:

```text
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_low_threshold_alert_threshold_sweep.csv
models/reports/temporal_alert_224_pretrained_gru_seq8_sampler_consecutive2_low_threshold_alert_threshold_sweep.csv
```

## Caveats

- The validation set has only 20 videos.
- Thresholds are selected on the same validation split.
- This is not enough for a final scientific claim.
- A separate holdout split or video-level cross-validation is still needed
  before writing a strong article conclusion.
