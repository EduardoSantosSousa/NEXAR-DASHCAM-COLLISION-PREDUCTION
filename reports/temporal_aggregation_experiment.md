# Temporal Aggregation Experiment

Run date: 2026-07-07

## Objective

This experiment tests simple temporal post-processing before moving to new model
architectures or imbalance-handling strategies. The goal is to check whether
risk-score smoothing or consecutive-frame alert rules can reduce false alarms
without retraining the model.

The base model is:

```text
temporal_alert_224_pretrained_best
```

The evaluation split is:

```text
data/interim/sample_100_videos_splits.csv --split val
```

## Implemented Methods

The threshold sweep script now supports:

- raw scores, unchanged;
- causal moving average over a trailing time window;
- causal rolling max over a trailing time window;
- alert only after N consecutive frames above the threshold.

Relevant CLI options:

```powershell
--aggregation raw
--aggregation moving_average --aggregation-window-seconds 3
--aggregation rolling_max --aggregation-window-seconds 3
--min-consecutive-frames 2
```

These methods operate on existing risk-score CSVs, so they do not require video
rescoring or model retraining.

## Commands

Moving average, 3 seconds:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --aggregation moving_average --aggregation-window-seconds 3
```

Moving average, 5 seconds:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --aggregation moving_average --aggregation-window-seconds 5
```

Rolling max, 3 seconds:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --aggregation rolling_max --aggregation-window-seconds 3
```

Two consecutive frames:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --min-consecutive-frames 2
```

Moving average, 3 seconds, plus two consecutive frames:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --aggregation moving_average --aggregation-window-seconds 3 --min-consecutive-frames 2
```

## Results at Minimum Recall 0.70

The table below selects an operating point by requiring alert recall at least
`0.70`, then minimizing false alarm rate.

| Method | Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw pretrained best | 0.23 | 0.667 | 0.800 | 0.400 | 0.200 | -7.616 s |
| Moving average 3s | 0.17 | 0.700 | 0.700 | 0.300 | 0.300 | -7.182 s |
| Moving average 5s | 0.13 | 0.667 | 0.800 | 0.400 | 0.200 | -7.241 s |
| Rolling max 3s | 0.23 | 0.667 | 0.800 | 0.400 | 0.200 | -7.616 s |
| Consecutive 2 frames | 0.13 | 0.727 | 0.800 | 0.300 | 0.200 | -7.991 s |
| Moving average 3s + consecutive 2 | 0.12 | 0.667 | 0.800 | 0.400 | 0.200 | -9.116 s |

## Interpretation

The strongest post-processing rule is:

```text
raw risk scores + alert after 2 consecutive frames above threshold
```

At threshold `0.13`, it keeps recall at `0.80`, increases precision from `0.667`
to `0.727`, and reduces false alarm rate from `0.400` to `0.300`.

Moving average over 3 seconds also reduces false alarm rate to `0.300`, but it
drops recall to `0.700`. Moving average over 5 seconds and rolling max over 3
seconds do not improve the best raw operating point. Combining moving average 3s
with consecutive-2 does not improve over consecutive-2 alone.

## Current Best Operating Point

For the current validation split, the most useful operating point is:

| Model | Post-processing | Threshold | Precision | Recall | False alarm rate | Mean alert error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `temporal_alert_224_pretrained_best` | 2 consecutive frames | 0.13 | 0.727 | 0.800 | 0.300 | -7.991 s |

This is now the strongest baseline before testing imbalance handling or sequence
models.

## Caveats

- The validation set contains only 20 videos.
- Threshold and post-processing choice are selected on the validation split.
- A separate holdout split or cross-validation would be needed for a stronger
  scientific claim.
- The method is causal because moving windows only use current and previous risk
  scores.
