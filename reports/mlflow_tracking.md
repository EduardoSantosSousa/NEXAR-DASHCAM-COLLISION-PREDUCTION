# MLflow Experiment Tracking

This project now supports local MLflow tracking for training, temporal alert
evaluation, and threshold sweeps.

## What Gets Logged

Training runs log:

- training parameters;
- frame-level validation metrics per epoch;
- final frame-level metrics;
- final video-level metrics;
- model checkpoint;
- frame predictions;
- video predictions;
- confusion matrix and ROC curve figures.

Alert evaluation runs log:

- sample CSV;
- checkpoint path;
- split;
- FPS;
- threshold;
- alert precision;
- alert recall;
- false alarm rate;
- missed event rate;
- lead-time metrics;
- temporal risk scores;
- alert predictions;
- risk curve figures.

Threshold sweep runs log:

- threshold range;
- sweep CSV;
- sweep figure;
- automatic candidate point;
- operating points selected under minimum recall constraints.

## Install

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Current Validation Experiments with MLflow

Create the split:

```powershell
.\venv\Scripts\python.exe scripts\create_video_split.py --input data\interim\sample_100_videos.csv --output data\interim\sample_100_videos_splits.csv --val-size 0.2 --random-state 42
```

Train from scratch:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_split --epochs 3 --batch-size 64 --learning-rate 0.0001 --num-workers 2
```

Evaluate from scratch:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_split_resnet18.pt --experiment-name temporal_alert_224_split --sample-csv data\interim\sample_100_videos_splits.csv --split val --fps 1 --threshold 0.5 --batch-size 32
```

Sweep from scratch:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_split --sample-csv data\interim\sample_100_videos_splits.csv --split val
```

Train pretrained:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_pretrained --pretrained --epochs 8 --batch-size 64 --learning-rate 0.00005 --num-workers 2
```

Evaluate pretrained:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_pretrained_resnet18.pt --experiment-name temporal_alert_224_pretrained --sample-csv data\interim\sample_100_videos_splits.csv --split val --fps 1 --threshold 0.5 --batch-size 32
```

Sweep pretrained:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained --sample-csv data\interim\sample_100_videos_splits.csv --split val
```

## Open the MLflow UI

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## Useful CLI Options

Disable MLflow for a run:

```powershell
--no-mlflow
```

Use a custom MLflow experiment name:

```powershell
--mlflow-experiment-name nexar-validation
```

Use a custom run name:

```powershell
--mlflow-run-name temporal_alert_224_pretrained_eval_v1
```

Use a custom tracking URI:

```powershell
--mlflow-tracking-uri sqlite:///C:/path/to/mlflow.db
```
