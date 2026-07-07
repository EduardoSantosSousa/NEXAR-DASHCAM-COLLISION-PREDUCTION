# Initial Dataset Analysis

## Data Location

The raw Nexar Collision Prediction dataset is stored locally in:

```text
data/raw/
```

The raw folder contains:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `train/`
- `test/`

## Metadata Schema

The training metadata contains four columns:

| Column | Description |
| --- | --- |
| `id` | Video identifier. It matches the MP4 filename without extension. |
| `time_of_event` | Timestamp of the collision or near-collision event for positive samples. |
| `time_of_alert` | Timestamp at which an alert should be raised for positive samples. |
| `target` | Binary label, where `1` indicates collision/near-collision risk and `0` indicates normal driving. |

The test metadata contains only:

| Column | Description |
| --- | --- |
| `id` | Video identifier. |

The sample submission contains:

| Column | Description |
| --- | --- |
| `id` | Test video identifier. |
| `target` | Predicted binary label or score, depending on the competition submission format. |

## Initial Counts

| Split | Rows | Videos |
| --- | ---: | ---: |
| Train | 1,500 | 1,500 |
| Test | 1,344 | 1,344 |
| Sample submission | 1,344 | N/A |

The training set is balanced:

| Target | Meaning | Count |
| --- | --- | ---: |
| `0` | Normal driving / no event | 750 |
| `1` | Collision or near-collision event | 750 |

All train CSV IDs match MP4 files in `data/raw/train/`, and all test CSV IDs match MP4 files in `data/raw/test/`.

## File Size Summary

| Split | Total Size | Average Video Size |
| --- | ---: | ---: |
| Train | ~23.78 GB | ~16.23 MB |
| Test | ~5.45 GB | ~4.15 MB |

The dataset is large enough that the first experiments should use a small, stratified sample instead of processing all videos at once.

## Event Timing

For positive training samples:

| Field | Value |
| --- | ---: |
| Positive samples with timing labels | 750 |
| Average event timestamp | 19.102 s |
| Average alert timestamp | 17.501 s |
| Average lead time | 1.600 s |
| Minimum lead time | 0.033 s |
| Maximum lead time | 4.466 s |

Lead time is defined as:

```text
lead_time = time_of_event - time_of_alert
```

No positive sample had negative or zero lead time in the initial inspection.

## Scientific Implications

This dataset supports two related research tasks:

1. **Video-level risk classification**: predict whether a video contains a collision or near-collision event.
2. **Early risk anticipation**: estimate risk before the event timestamp, using the alert timestamp as a temporal reference.

For the scientific article, the second task is more interesting because it studies the ability of visual models to anticipate critical traffic events before they occur.

## Recommended First Experimental Protocol

Start with a small stratified subset:

- 50 positive videos.
- 50 negative videos.

For positive videos, extract frames in windows relative to `time_of_alert` and `time_of_event`.

Suggested first windows:

| Window | Purpose |
| --- | --- |
| `time_of_event - 5s` to `time_of_event` | Capture visual cues before the event. |
| `time_of_alert - 3s` to `time_of_alert` | Study early warning context. |
| Uniform sampling at 1 FPS | Build a simple baseline quickly. |

The first baseline should be simple and defensible:

```text
sampled frames -> pretrained CNN features -> aggregation -> binary classifier
```

This creates a reference point before moving to temporal models such as CNN + LSTM, CNN + GRU, 3D CNNs, or video transformers.

## Visual Sample Created

A stratified visual sample was generated after the initial dataset inspection.

| Artifact | Location |
| --- | --- |
| Sample manifest | `data/interim/sample_100_videos.csv` |
| Extracted frames | `data/interim/frames_sample/` |
| Frame manifest | `data/interim/sample_frames_manifest.csv` |
| Example figures | `outputs/figures/` |

The sample contains 100 videos:

| Target | Count |
| --- | ---: |
| `0` | 50 |
| `1` | 50 |

Five frames were extracted per video, resulting in 500 frames.

| Target | Frame Labels |
| --- | --- |
| `1` | `event_minus_5s`, `event_minus_3s`, `event_minus_1s`, `alert`, `event` |
| `0` | `video_20pct`, `video_40pct`, `video_60pct`, `video_80pct`, `video_95pct` |

This sample is the recommended input for the first qualitative analysis notebook:

```text
notebooks/02_data_exploration.ipynb
```
