# Nexar Dashcam Collision Prediction

Project scaffold for building a dashcam collision prediction pipeline with data preparation, frame extraction, baseline modeling, evaluation, reporting, and a Streamlit dashboard.

## Structure

- `data/`: raw, interim, processed, and external datasets.
- `notebooks/`: exploratory and experiment notebooks.
- `src/nexar_collision/`: reusable Python package code.
- `models/`: checkpoints and model reports.
- `outputs/`: generated figures, predictions, and submissions.
- `app/`: Streamlit dashboard.
- `reports/`: methodology notes, paper draft, and references.
- `scripts/`: command-line entry points for common workflows.

## Quick Start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the dashboard:

```powershell
streamlit run app/streamlit_app.py
```

## Baseline Training

Create the visual sample and extracted frames first:

```powershell
.\venv\Scripts\python.exe scripts\create_sample.py
.\venv\Scripts\python.exe scripts\extract_sample_frames.py
```

Train the initial ResNet18 frame baseline:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --epochs 3 --batch-size 16
```

The training script automatically uses CUDA when a GPU-enabled PyTorch build is
installed.

Evaluate temporal alerts:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py --fps 1 --threshold 0.5 --batch-size 32
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py
```

## Experiment Tracking with MLflow

The training, temporal evaluation, and threshold sweep scripts can log
parameters, metrics, checkpoints, CSV outputs, and figures to MLflow.

Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the MLflow UI:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

MLflow logging is enabled by default. Use `--no-mlflow` to disable it for any
experiment script.

Review videos, frames, risk curves, and KPIs in:

```text
notebooks/06_results_review.ipynb
```

Current roadmap:

```text
reports/next_steps_roadmap.md
```

Best-checkpoint follow-up:

```text
reports/best_checkpoint_experiment.md
reports/temporal_aggregation_experiment.md
notebooks/08_best_checkpoint_mlflow_review.ipynb
notebooks/09_temporal_aggregation_review.ipynb
```

Train the temporal-label progression model:

```powershell
.\venv\Scripts\python.exe scripts\create_temporal_frame_dataset.py --fps 2 --pre-alert-margin 3 --image-size 224 --output-dir data\interim\temporal_frames_224 --manifest data\interim\temporal_frames_224_manifest.csv
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --experiment-name temporal_alert_224 --epochs 3 --batch-size 64 --learning-rate 0.0001 --num-workers 2
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_resnet18.pt --experiment-name temporal_alert_224 --fps 1 --threshold 0.5 --batch-size 32
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224
```

Train with best-checkpoint selection and early stopping:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py --manifest data\interim\temporal_frames_224_manifest.csv --split-manifest data\interim\sample_100_videos_splits.csv --experiment-name temporal_alert_224_pretrained --pretrained --epochs 8 --batch-size 64 --learning-rate 0.00005 --num-workers 2 --monitor-metric roc_auc --monitor-mode max --patience 3
.\venv\Scripts\python.exe scripts\evaluate_model.py --checkpoint models\checkpoints\temporal_alert_224_pretrained_best_resnet18.pt --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val --fps 1 --threshold 0.5 --batch-size 32
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py --experiment-name temporal_alert_224_pretrained_best --sample-csv data\interim\sample_100_videos_splits.csv --split val
```
