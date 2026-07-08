# CNN + GRU/LSTM Sequence Experiment

Run date: 2026-07-07

## Objective

This experiment implements and evaluates a first causal CNN + recurrent model for
temporal collision alert prediction.

The goal is to move beyond independent frame classification by using a short
history of frames before each timestamp:

```text
frames[t-3:t] -> ResNet18 encoder -> GRU/LSTM -> alert-region score at t
```

The model is causal: it only uses the current and previous frames, not future
frames.

## Implementation

Added:

- `FrameSequenceCollisionDataset`
- `CnnRnnCollisionModel`
- `scripts/train_sequence_model.py`
- `scripts/evaluate_sequence_model.py`

The sequence model supports:

- ResNet18 frame encoder;
- GRU or LSTM temporal head;
- pretrained ImageNet weights;
- frozen or fine-tuned CNN encoder;
- best-checkpoint selection;
- early stopping;
- MLflow logging;
- temporal alert evaluation;
- threshold sweeps;
- consecutive-frame post-processing.

## Experiment Setup

Common configuration:

| Setting | Value |
| --- | --- |
| Manifest | `data/interim/temporal_frames_224_manifest.csv` |
| Split file | `data/interim/sample_100_videos_splits.csv` |
| Evaluation split | `val` |
| Sequence length | 4 frames |
| CNN encoder | ResNet18 pretrained |
| CNN policy | Frozen |
| Hidden size | 128 |
| Train sequence stride | 2 |
| Validation sequence stride | 1 |
| Batch size | 16 |
| Epochs | 4 |
| Monitor metric | validation ROC-AUC |

Why this setup:

- Sequence length 4 at 2 FPS gives roughly 2 seconds of context.
- Frozen CNN keeps the first sequence experiment reasonably fast.
- Train stride 2 reduces overlapping training samples while validation still
  scores every timestamp.

## Commands

GRU:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_pretrained_gru_seq4 --pretrained --sequence-length 4 --train-sequence-stride 2 --val-sequence-stride 1 --rnn-type gru --epochs 4 --batch-size 16 --learning-rate 0.0001 --hidden-size 128 --num-workers 0 --monitor-metric roc_auc --monitor-mode max --patience 2
.\venv\Scripts\python.exe scripts\evaluate_sequence_model.py --checkpoint models\checkpoints\temporal_alert_224_pretrained_gru_seq4_best_sequence.pt --experiment-name temporal_alert_224_pretrained_gru_seq4 --manifest data\interim\temporal_frames_224_manifest.csv --sample-csv data\interim\sample_100_videos_splits.csv --split-manifest data\interim\sample_100_videos_splits.csv --split val --threshold 0.5 --batch-size 16
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_gru_seq4 --sample-csv data\interim\sample_100_videos_splits.csv --split val
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_gru_seq4 --sample-csv data\interim\sample_100_videos_splits.csv --split val --min-consecutive-frames 2
```

LSTM:

```powershell
.\venv\Scripts\python.exe scripts\train_sequence_model.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_pretrained_lstm_seq4 --pretrained --sequence-length 4 --train-sequence-stride 2 --val-sequence-stride 1 --rnn-type lstm --epochs 4 --batch-size 16 --learning-rate 0.0001 --hidden-size 128 --num-workers 0 --monitor-metric roc_auc --monitor-mode max --patience 2
.\venv\Scripts\python.exe scripts\evaluate_sequence_model.py --checkpoint models\checkpoints\temporal_alert_224_pretrained_lstm_seq4_best_sequence.pt --experiment-name temporal_alert_224_pretrained_lstm_seq4 --manifest data\interim\temporal_frames_224_manifest.csv --sample-csv data\interim\sample_100_videos_splits.csv --split-manifest data\interim\sample_100_videos_splits.csv --split val --threshold 0.5 --batch-size 16
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_lstm_seq4 --sample-csv data\interim\sample_100_videos_splits.csv --split val
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_lstm_seq4 --sample-csv data\interim\sample_100_videos_splits.csv --split val --min-consecutive-frames 2
```

## Frame-Level Validation Results

| Model | Best epoch | Best ROC-AUC | Final F1 | Final ROC-AUC | Early stopped |
| --- | ---: | ---: | ---: | ---: | --- |
| CNN + GRU seq4 | 4 | 0.567 | 0.142 | 0.567 | No |
| CNN + LSTM seq4 | 1 | 0.549 | 0.100 | 0.540 | Yes, epoch 3 |

The GRU performed better than the LSTM in this first configuration.

## Alert Results at Threshold 0.50

| Model | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN + GRU seq4 | 0.667 | 0.400 | 0.200 | 0.600 | -7.204 s |
| CNN + LSTM seq4 | 0.500 | 0.200 | 0.200 | 0.800 | -4.017 s |

At threshold `0.50`, both sequence models are conservative. They reduce false
alarms but miss too many positive events.

## Sweep Results at Minimum Recall 0.70

The table below applies the same operating-point selection rule used in previous
experiments: require recall at least `0.70`, then minimize false alarm rate.

| Model | Post-processing | Threshold | Precision | Recall | False alarm rate | Missed event rate | Mean alert error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current best baseline | 2 consecutive frames | 0.13 | 0.727 | 0.800 | 0.300 | 0.200 | -7.991 s |
| CNN + GRU seq4 | Raw | 0.33 | 0.500 | 0.800 | 0.800 | 0.200 | -12.108 s |
| CNN + GRU seq4 | 2 consecutive frames | 0.25 | 0.556 | 1.000 | 0.800 | 0.000 | -13.328 s |
| CNN + LSTM seq4 | Raw | 0.45 | 0.615 | 0.800 | 0.500 | 0.200 | -6.725 s |
| CNN + LSTM seq4 | 2 consecutive frames | 0.43 | 0.583 | 0.700 | 0.500 | 0.300 | -9.047 s |

## Conclusion

The CNN + GRU/LSTM pipeline is now implemented and evaluable end to end, but this
first frozen-encoder sequence setup does not beat the current best baseline.

Current best remains:

```text
temporal_alert_224_pretrained_best + 2 consecutive frames
threshold = 0.13
precision = 0.727
recall = 0.800
false_alarm_rate = 0.300
```

The sequence models are useful as a foundation, but the first configuration
appears under-calibrated: at high recall, false alarms increase too much.

## Recommended Next Sequence Experiments

1. Fine-tune the last ResNet block instead of freezing the full CNN.
2. Increase sequence length from 4 to 8 frames.
3. Use a lower learning rate for CNN parameters and higher rate for the RNN head.
4. Add `WeightedRandomSampler` or focal loss for sequence training.
5. Sample windows more deliberately around alert/event regions instead of using
   every timestamp equally.

The most promising next run is:

```text
temporal_alert_224_pretrained_gru_seq8_sampler
```

with sequence length 8, GRU, pretrained CNN, partial fine-tuning, and imbalance
handling.
