# Methodology

Document the dataset, preprocessing steps, frame extraction strategy, model architecture, and evaluation protocol here.

## Initial Dataset Understanding

The project uses the Nexar Collision Prediction dataset stored locally in `data/raw/`.
The initial inspection found `train.csv`, `test.csv`, `sample_submission.csv`, and
matching MP4 files under `data/raw/train/` and `data/raw/test/`.

The training split contains 1,500 videos and is balanced: 750 negative samples and
750 positive samples. Positive samples include `time_of_event` and `time_of_alert`,
which allows the project to be framed as both a video-level classification task and
an early risk anticipation task.

For the article, the early anticipation framing is especially relevant because the
average lead time between alert and event is approximately 1.6 seconds. This creates
a measurable temporal objective: detect risk before the critical event occurs.

See `reports/dataset_initial_analysis.md` for the full initial inspection.

## Stratified Visual Sample

A stratified sample was created to support qualitative inspection before model
training. The sample contains 100 training videos:

- 50 positive videos.
- 50 negative videos.

The sample manifest is stored at `data/interim/sample_100_videos.csv`.

For each positive video, five frames were extracted around the risk event:

- `event_minus_5s`
- `event_minus_3s`
- `event_minus_1s`
- `alert`
- `event`

For each negative video, five frames were extracted at fixed relative positions:

- `video_20pct`
- `video_40pct`
- `video_60pct`
- `video_80pct`
- `video_95pct`

This produced 500 extracted frames, documented in
`data/interim/sample_frames_manifest.csv`.

The first visual outputs were exported to `outputs/figures/`:

- `positive_timeline_00364.png`
- `negative_timeline_01840.png`
- `positive_vs_negative_example.png`

This step supports the article by providing qualitative evidence before
introducing model results. It also validates that the event timing annotations can
be used to construct a temporal risk anticipation protocol.

## Baseline Frame Classifier

The first modeling experiment uses the 500 extracted frames from the stratified
visual sample. The task is binary frame-level classification:

```text
frame -> ResNet18 -> collision risk target
```

The split is performed by video ID, not by individual frame. This prevents frames
from the same video appearing in both training and validation sets.

Initial protocol:

- Input: `data/interim/sample_frames_manifest.csv`
- Train videos: 80
- Validation videos: 20
- Train frames: 400
- Validation frames: 100
- Model: ResNet18 with a binary classification head
- Epochs: 3
- Batch size: 16
- Device: CUDA GPU

Initial validation results:

| Level | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frame | 0.55 | 0.538 | 0.700 | 0.609 | 0.560 |
| Video, mean aggregation | 0.55 | 0.538 | 0.700 | 0.609 | 0.620 |
| Video, max aggregation | 0.45 | 0.467 | 0.700 | 0.560 | 0.530 |

These results should be interpreted only as a first technical baseline. The model
was trained on a small sample and without transfer learning weights. The result is
useful because it validates the training pipeline, GPU execution, video-level
splitting, prediction export, and metric generation.

Artifacts:

- `models/reports/baseline_metrics.json`
- `models/checkpoints/baseline_resnet18.pt`
- `outputs/predictions/baseline_frame_predictions.csv`
- `outputs/predictions/baseline_video_predictions.csv`
- `outputs/figures/baseline_frame_confusion_matrix.png`
- `outputs/figures/baseline_frame_roc_curve.png`

## Temporal Alert Evaluation

The baseline model was also evaluated as an alert system by applying it across
sampled videos at 1 FPS. This produces a temporal risk curve for each video:

```text
timestamp -> risk_score
```

The predicted alert time is defined as the first timestamp where the risk score
crosses a chosen threshold.

At threshold `0.50`, the model reached high alert recall but also produced many
false alarms:

| KPI | Value |
| --- | ---: |
| Alert precision | 0.538 |
| Alert recall | 0.980 |
| False alarm rate | 0.840 |
| Missed event rate | 0.020 |
| Mean predicted lead time | 17.190 s |
| Mean alert time error | -15.606 s |

A threshold sweep from `0.10` to `0.99` showed the expected trade-off: increasing
the threshold reduces false alarms but also increases missed events. For example,
at threshold `0.99`, false alarm rate decreases to `0.220`, but alert recall also
drops to `0.560`.

These results indicate that the current frame-level baseline can produce temporal
risk scores, but it is not yet calibrated enough for reliable alert prediction.
The next modeling step should use transfer learning and/or temporal labels that
distinguish early, alert, and event-adjacent frames.

## Temporal Label Progression

To reduce the mismatch between video-level labels and temporal alert prediction,
a new frame dataset was created with temporally refined labels. Positive frames
are now defined as frames inside the window from `time_of_alert - 3s` to
`time_of_event`. Frames before this window, and all frames from negative videos,
are labeled as non-alert frames.

This produced 7,760 frames:

| Temporal Target | Count |
| --- | ---: |
| `0` | 7,295 |
| `1` | 465 |

A new experiment, `temporal_alert_224`, was trained with class-weighted cross
entropy. Compared with the previous video-label baseline at threshold `0.50`, the
temporal-label model reduced false alarm rate from `0.840` to `0.620`, while
keeping alert recall high at `0.960`.

The model remains imperfect, but this is a meaningful improvement because it
aligns training labels more closely with the actual alert prediction objective.

See `reports/temporal_label_progression.md` for details.

## Validation-Only Experiment Comparison

The temporal alert protocol was later corrected to evaluate only videos from a
fixed validation split instead of evaluating all 100 sampled videos. The split is
stored at:

```text
data/interim/sample_100_videos_splits.csv
```

This split contains 80 training videos and 20 validation videos, stratified by
video-level target. The corrected evaluation compares:

- `temporal_alert_224_split`: ResNet18 trained from scratch.
- `temporal_alert_224_pretrained`: ResNet18 initialized with ImageNet weights.

At threshold `0.50`, the pretrained model reduced false alarm rate from `0.500`
to `0.300` and improved mean alert time error from `-17.960 s` to `-9.774 s`,
but recall dropped from `0.400` to `0.300`.

Under a matched recall constraint of at least `0.50`, the pretrained model was
stronger: threshold `0.18` achieved precision `0.625`, recall `0.500`, and false
alarm rate `0.300`, compared with the scratch model at threshold `0.38`, which
achieved precision `0.455`, recall `0.500`, and false alarm rate `0.600`.

The detailed comparison is documented in:

```text
reports/experiment_comparison_val.md
```
