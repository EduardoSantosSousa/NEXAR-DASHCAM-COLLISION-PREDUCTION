# Next Steps Roadmap

This document records the recommended next steps after adding MLflow tracking and
running the corrected validation-only experiments.

The project is now in the experiment-management stage: the main priority is to
make each result comparable, reproducible, and useful for later article writing.

## Current State

Implemented:

- Initial dataset inspection.
- Stratified 100-video sample.
- Frame extraction.
- Frame-level ResNet18 baseline.
- Temporal label progression dataset.
- Fixed train/validation split by video.
- Validation-only temporal alert evaluation.
- Threshold sweeps.
- Experiment comparison report.
- Jupyter analysis notebook.
- MLflow tracking for training, alert evaluation, and threshold sweeps.

Important files:

```text
data/interim/sample_100_videos_splits.csv
data/interim/temporal_frames_224_manifest.csv
reports/experiment_comparison_val.md
reports/best_checkpoint_experiment.md
reports/temporal_aggregation_experiment.md
reports/cnn_rnn_sequence_experiment.md
reports/mlflow_tracking.md
notebooks/07_validation_experiment_analysis.ipynb
notebooks/08_best_checkpoint_mlflow_review.ipynb
mlflow.db
```

## Progress Update - 2026-07-07

Completed after the original roadmap:

- Best-checkpoint selection for training.
- Optional early stopping controlled by `--patience`.
- MLflow logging of `best_epoch`, best validation metrics, early-stopping state,
  final checkpoint, and best checkpoint.
- New best-checkpoint evaluations:
  - `temporal_alert_224_split_best`
  - `temporal_alert_224_pretrained_best`
- New review notebook:
  `notebooks/08_best_checkpoint_mlflow_review.ipynb`.
- Simple temporal aggregation threshold sweeps:
  - moving average 3s;
  - moving average 5s;
  - rolling max 3s;
  - 2 consecutive frames above threshold;
  - moving average 3s plus 2 consecutive frames.

Key result:

The pretrained best checkpoint reached recall `0.80` at threshold `0.23` on the
validation split, with precision `0.667`, false alarm rate `0.400`, and mean
alert error `-7.616 s`.

After temporal aggregation, the strongest operating point is now the pretrained
best checkpoint with a 2-consecutive-frame alert rule: threshold `0.13`, recall
`0.80`, precision `0.727`, false alarm rate `0.300`, and mean alert error
`-7.991 s`.

The next modeling priority is now imbalance handling.

Additional sequence-model progress:

- Implemented CNN + GRU/LSTM sequence training and evaluation.
- Ran first frozen-encoder sequence experiments:
  - `temporal_alert_224_pretrained_gru_seq4`
  - `temporal_alert_224_pretrained_lstm_seq4`
- The sequence pipeline works end to end, but does not yet beat the current best
  baseline.

## Progress Update - 2026-07-09

Completed after the first sequence experiments:

- Implemented sequence-level `WeightedRandomSampler`.
- Added partial ResNet18 fine-tuning through `cnn_train_policy=layer4`.
- Added separate CNN/head learning rates for sequence training.
- Ran:
  - `temporal_alert_224_pretrained_gru_seq8_sampler`
  - `temporal_alert_224_pretrained_gru_seq8_focal`
- Registered the analysis in:
  - `reports/gru_seq8_sampler_experiment.md`
  - `reports/gru_seq8_focal_experiment.md`

Key result:

The GRU seq8 sampler run improved frame-level ROC-AUC to `0.612`, but it did not
beat the current alert baseline. In an expanded threshold sweep, it reached
recall `0.80` only at threshold `0.01`, with false alarm rate `0.80`. Its best
low-false-alarm operating point had recall only `0.30`.

The GRU seq8 focal run improved the best frame-level F1 to `0.178`, but also did
not beat the alert baseline. With false alarm rate constrained to `0.30`, recall
was only `0.20`. At recall `0.80`, false alarm rate rose to `0.80`.

Decision:

Keep `temporal_alert_224_pretrained_best` with the 2-consecutive-frame rule as
the current best baseline. Treat `temporal_alert_224_pretrained_gru_seq8_sampler`
and `temporal_alert_224_pretrained_gru_seq8_focal` as informative failed
experiments rather than replacement models.

## Progress Update - 2026-07-13

Completed after the full product dataset was generated:

- Ran the full-dataset ResNet18 product experiment:
  `product_temporal_alert_224_pretrained`.
- Registered the analysis in:
  `reports/product_full_resnet18_experiment.md`.
- Added EfficientNet-B0 support to the baseline CNN pipeline.
- Added product training controls:
  - `--backbone`;
  - `--freeze-backbone`;
  - `--unfreeze-last-n-blocks`;
  - `--amp`;
  - `--log-every-n-batches`.
- Ran:
  `product_temporal_alert_224_efficientnet_b0_frozen_amp`.
- Ran:
  `product_temporal_alert_224_efficientnet_b0_amp_finetune_last2`.
- Added `convnext_tiny` to the baseline CNN backbones.
- Ran:
  `product_temporal_alert_224_convnext_tiny_frozen_amp`.
- Ran:
  `product_temporal_alert_224_convnext_tiny_amp_finetune_last1`.
- Registered the analysis in:
  `reports/product_efficientnet_b0_experiment.md`.
- Registered the partial fine-tuning analysis in:
  `reports/product_efficientnet_b0_finetune_last2_experiment.md`.
- Registered the ConvNeXt-Tiny analysis in:
  `reports/product_convnext_tiny_experiment.md`.

Key full-dataset ResNet18 result:

At threshold `0.50`, alert recall was `0.884`, but false alarm rate was
`0.804`. With false alarm rate constrained to `0.30`, the best recall was
`0.580` using 2 consecutive frames. The model failed the product gate.

Key EfficientNet-B0 result:

The frozen EfficientNet-B0 experiment trained efficiently with AMP, but best
validation ROC-AUC was only `0.653`, below the ResNet18 reference of `0.667`.
At threshold `0.50`, alert recall was `0.938`, but false alarm rate was
`0.866`. No threshold or 2-consecutive-frame rule satisfied the product gate.

Key EfficientNet-B0 partial fine-tuning result:

Unfreezing the last 2 EfficientNet-B0 feature blocks made `1131954 / 4010110`
parameters trainable. Training loss fell from `0.651` to `0.528`, but validation
ROC-AUC fell from `0.649` to `0.645`. The best checkpoint remained below both
the frozen EfficientNet-B0 result and the ResNet18 reference.

Key ConvNeXt-Tiny result:

The frozen ConvNeXt-Tiny run reached best validation ROC-AUC `0.657`.
Partial fine-tuning of the last ConvNeXt feature block reached best validation
ROC-AUC `0.659`, but then overfit quickly: training loss fell from `0.592` to
`0.254` while validation ROC-AUC fell from `0.659` to `0.642`. ConvNeXt-Tiny
also failed to exceed the ResNet18 reference of `0.667`.

Decision:

Do not evaluate either full-dataset ResNet18 or frozen EfficientNet-B0 on
holdout. The holdout remains sealed until a validation candidate satisfies the
product criteria.

Current product gate:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

Recommended next modeling step:

Move away from isolated frame-level image classification and build an
event-centered temporal formulation:

```text
product_temporal_event_window_seq_model
```

The next dataset should sample short temporal windows around the pre-alert
interval and train/evaluate at sequence or video-window level, instead of
treating every frame as an independent sample.

Preparation completed:

- Created `scripts/create_event_window_sequence_manifest.py`.
- Created `data/interim/product_event_windows_seq8_manifest.csv`.
- Added explicit-window sequence dataset support.
- Added AMP and batch logging to sequence training.
- Ran a 1-epoch smoke test:
  `product_event_window_gru_seq8_resnet18_frozen_smoke`.

Smoke result:

```text
validation window ROC-AUC = 0.705
validation window F1 = 0.638
```

This is promising, but it is window-level validation only. It still needs
multi-epoch training and temporal alert evaluation on full validation videos.

Multi-epoch result:

- Trained `product_event_window_gru_seq8_resnet18_frozen`.
- Best checkpoint: epoch 3.
- Best validation window ROC-AUC: `0.707`.
- Temporal alert evaluation at threshold `0.50`:
  - precision `0.557`;
  - recall `0.866`;
  - false alarm rate `0.688`.
- Best low-false-alarm operating point:
  - raw scores, threshold `0.65`;
  - precision `0.680`;
  - recall `0.625`;
  - false alarm rate `0.295`.

Decision:

Do not evaluate on holdout. The event-window GRU is the strongest modeling
direction so far, but it still fails the product gate because false alarms
remain too high at recall `>= 0.80`.

Hard-negative result:

- Implemented `scripts/mine_hard_negative_windows.py`.
- Mined 1024 high-scoring train negative windows from the previous event-window
  GRU checkpoint.
- Created:
  `data/interim/product_event_windows_seq8_hard_negative_manifest.csv`.
- Trained:
  `product_event_window_gru_seq8_hard_negative`.
- Best checkpoint: epoch 2.
- Best validation window ROC-AUC: `0.701`.
- Temporal alert evaluation at threshold `0.50`:
  - precision `0.518`;
  - recall `0.893`;
  - false alarm rate `0.830`.
- Best low-false-alarm operating point:
  - 2 consecutive frames, threshold `0.59`;
  - precision `0.696`;
  - recall `0.634`;
  - false alarm rate `0.277`.

Decision:

Do not evaluate the hard-negative model on holdout. It modestly improved the
low-false-alarm operating region, but still failed the validation product gate.

Alert-weighted result:

- Added per-window sample weighting to sequence training.
- Added optional alert-metric checkpoint selection through
  `--alert-metric-selection`.
- Trained:
  `product_event_window_gru_seq8_alert_weighted`.
- Best checkpoint: epoch 3, selected by `alert_selection_score`.
- Best validation-window alert proxy:
  - precision `0.717`;
  - recall `0.812`;
  - false alarm rate `0.321`.
- Full-video temporal alert evaluation at threshold `0.50`:
  - precision `0.578`;
  - recall `0.830`;
  - false alarm rate `0.607`.
- Best full-video low-false-alarm operating point:
  - moving average 3s + 2 consecutive frames, threshold `0.52`;
  - precision `0.645`;
  - recall `0.536`;
  - false alarm rate `0.295`.

Decision:

Do not evaluate the alert-weighted model on holdout. The new selection strategy
reduced false alarms at threshold `0.50`, but still failed the validation
product gate on full videos.

Alert-weighted layer4 result:

- Tightened temporal model fine-tuning so frozen CNN blocks stay in eval mode
  while `layer4` trains.
- Trained:
  `product_event_window_gru_seq8_alert_weighted_layer4`.
- Best checkpoint: epoch 1, selected by `alert_selection_score`.
- Trainable parameters:
  - CNN layer4: `8393728`;
  - temporal head: `246786`.
- Best validation-window alert proxy:
  - precision `0.705`;
  - recall `0.705`;
  - false alarm rate `0.295`.
- Full-video temporal alert evaluation at threshold `0.50`:
  - precision `0.588`;
  - recall `0.446`;
  - false alarm rate `0.312`.
- Best full-video low-false-alarm operating point:
  - raw scores, threshold `0.51`;
  - precision `0.623`;
  - recall `0.429`;
  - false alarm rate `0.259`.

Decision:

Do not evaluate the layer4 model on holdout. Fine-tuning the visual block
overfit quickly and did not improve the validation product tradeoff.

Dense-negatives result:

- Added a safety check to `scripts/create_event_window_sequence_manifest.py` so
  `--allowed-splits` cannot silently include holdout when the manifest has no
  split column.
- Regenerated the dense manifest with:
  `--split-manifest data/interim/full_train_product_splits.csv`.
- Created:
  `data/interim/product_event_windows_seq8_dense_negatives_manifest.csv`.
- Manifest distribution:
  - train windows: `18775`;
  - validation windows: `4002`;
  - targets: `17775` negative, `5002` positive.
- Trained:
  `product_event_window_gru_seq8_dense_negatives`.
- Best checkpoint: epoch 3, selected by `alert_selection_score`.
- Best validation-window alert proxy:
  - precision `0.679`;
  - recall `0.679`;
  - false alarm rate `0.321`.
- Full-video temporal alert evaluation at threshold `0.50`:
  - precision `0.560`;
  - recall `0.795`;
  - false alarm rate `0.625`.
- Best full-video low-false-alarm operating point:
  - 2 consecutive frames, threshold `0.60`;
  - precision `0.677`;
  - recall `0.598`;
  - false alarm rate `0.286`.

Decision:

Do not evaluate the dense-negatives model on holdout. Denser negative/safe
sampling improved coverage, but it did not solve the recall versus false-alarm
tradeoff.

Pre-alert phase target result:

- Created `scripts/create_prealert_phase_sequence_manifest.py`.
- Created:
  `data/interim/product_event_windows_seq8_prealert_phases_manifest.csv`.
- The manifest split positive videos into:
  - `positive_safe`;
  - `prealert_early`;
  - `prealert_late`;
  - `event_near`.
- Binary target rule:
  - `negative_video`, `positive_safe`, and `prealert_early` as target `0`;
  - `prealert_late` and `event_near` as target `1`.
- Manifest distribution:
  - train windows: `13222`;
  - validation windows: `2837`;
  - targets: `13996` negative, `2063` positive.
- Trained:
  `product_event_window_gru_seq8_prealert_phases`.
- Best checkpoint: epoch 1, selected by `alert_selection_score`.
- Best validation-window alert proxy:
  - precision `0.706`;
  - recall `0.750`;
  - false alarm rate `0.312`.
- Full-video temporal alert evaluation at threshold `0.50`:
  - precision `0.538`;
  - recall `0.884`;
  - false alarm rate `0.759`.
- Best full-video low-false-alarm operating point:
  - 2 consecutive frames, threshold `0.59`;
  - precision `0.677`;
  - recall `0.580`;
  - false alarm rate `0.277`.

Decision:

Do not evaluate the pre-alert phase binary model on holdout. The phase-aware
target improved window-level separation, reaching window ROC-AUC `0.749`, but
compressing the phases back into a binary target still failed the validation
product gate.

Phase-classifier result:

- Trained:
  `product_event_window_phase_classifier_seq8`.
- The model predicted four phase classes directly and used
  `P(prealert_late) + P(event_near)` as the alert score.
- Best checkpoint: epoch 4, selected by `alert_selection_score`.
- Best validation-window alert proxy:
  - precision `0.707`;
  - recall `0.732`;
  - false alarm rate `0.304`.
- Full-video temporal alert evaluation at threshold `0.50`:
  - precision `0.544`;
  - recall `0.884`;
  - false alarm rate `0.741`.
- Best full-video low-false-alarm operating point:
  - 2 consecutive frames, threshold `0.64`;
  - precision `0.648`;
  - recall `0.509`;
  - false alarm rate `0.277`.

Decision:

Do not evaluate the phase-classifier model on holdout. Explicit phase
prediction is conceptually cleaner, but it did not improve the full-video
product tradeoff.

Product-interface direction:

Do not prioritize Streamlit for the product interface. The intended product
path is a custom HTML/Tailwind frontend backed by an API service, such as
FastAPI, after a validation-approved model candidate is selected.

## Immediate Next Step

Stop evaluating failed candidates on holdout and stop escalating isolated
frame-level backbones. The goal now is to make the learning target match the
product behavior: early temporal collision anticipation.

Recommended command family:

```text
product_temporal_event_window_seq_model
```

Completed command:

```text
product_event_window_phase_classifier_seq8
```

Current next modeling step:

```text
product_context_hard_negatives_phase_classifier
```

The false-positive analysis showed that phase supervision alone is not enough.
The next step should mine contextual hard negatives from dense traffic,
close-vehicle, brake-light, low-light, and degraded-visual-quality scenes.

Contextual hard-negative mining completed:

- Created `scripts/mine_context_hard_negative_windows.py`.
- Scored train negative videos with
  `product_event_window_phase_classifier_seq8`.
- Selected 513 high-risk peaks from 240 train negative videos.
- Added 2202 contextual hard-negative windows to the phase manifest.
- Created:
  `data/interim/product_event_windows_seq8_context_hard_negatives_manifest.csv`.
- Documented the mining run in:
  `reports/product_context_hard_negative_mining.md`.

Contextual hard-negative training result:

- Trained:
  `product_context_hard_negatives_phase_classifier_seq8`.
- Best validation-window alert proxy:
  - precision `0.743`;
  - recall `0.696`;
  - false alarm rate `0.241`.
- Full-video evaluation at threshold `0.50`:
  - precision `0.523`;
  - recall `0.812`;
  - false alarm rate `0.741`.
- Best full-video low-false-alarm point with 2 consecutive frames:
  - precision `0.674`;
  - recall `0.554`;
  - false alarm rate `0.268`.
- Completed analysis is documented in:
  `reports/product_context_hard_negatives_phase_classifier_seq8_experiment.md`.

Decision:

Do not evaluate on holdout. Hard-negative mining helped the low-false-alarm
region only modestly and still missed too many positives.

Hard-negative pressure ablation prepared:

- Created `scripts/create_hard_negative_ablation_manifests.py`.
- Created three ablation manifests:
  - `data/interim/product_event_windows_seq8_context_hn_all_w15_manifest.csv`;
  - `data/interim/product_event_windows_seq8_context_hn_top1000_w20_manifest.csv`;
  - `data/interim/product_event_windows_seq8_context_hn_top1000_w15_manifest.csv`.
- All ablation manifests contain `holdout_rows = 0`.
- Commands are documented in:
  `reports/product_context_hard_negative_ablation_plan.md`.

Hard-negative pressure ablation C result:

- Trained:
  `product_context_hn_top1000_w15_phase_classifier_seq8`.
- Full-video validation at threshold `0.50`:
  - precision `0.509`;
  - recall `0.982`;
  - false alarm rate `0.946`.
- Best full-video low-false-alarm point with 2 consecutive frames:
  - precision `0.674`;
  - recall `0.536`;
  - false alarm rate `0.259`.
- Completed analysis is documented in:
  `reports/product_context_hn_top1000_w15_phase_classifier_seq8_experiment.md`.

Decision:

Do not evaluate on holdout. Continue with the next ablation:
`product_context_hn_top1000_w20_phase_classifier_seq8`.

Phase-classifier implementation and analysis:

- `scripts/train_sequence_model.py` now supports `--num-classes`,
  `--target-column`, and `--alert-class-indices`.
- `scripts/evaluate_sequence_model.py` now supports phase-model alert scoring.
- Commands and expected artifacts are documented in:
  `reports/product_event_window_phase_classifier_seq8_next_step.md`.
- Completed analysis is documented in:
  `reports/product_event_window_phase_classifier_seq8_experiment.md`.
- False-positive analysis is documented in:
  `reports/product_false_positive_error_analysis.md`.

Use MLflow while comparing candidates:

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open MLflow:

```text
http://127.0.0.1:5000
```

Use MLflow and local artifacts to update:

```text
notebooks/09_temporal_aggregation_review.ipynb
reports/experiment_comparison_val.md
reports/methodology.md
reports/product_full_resnet18_experiment.md
reports/product_efficientnet_b0_experiment.md
```

## Priority 1 - Best Checkpoint and Early Stopping

Problem:

The current training script saves the final epoch. This is not ideal because the
pretrained experiment had stronger intermediate validation ROC-AUC than the final
epoch.

Status:

Completed on 2026-07-07. See:

```text
reports/best_checkpoint_experiment.md
```

Goal:

- Save the best checkpoint according to validation ROC-AUC or validation F1.
- Log the best epoch and best metric to MLflow.
- Optionally stop training early when the metric does not improve.

Recommended implementation:

- Add `--monitor-metric`, default `roc_auc`.
- Add `--monitor-mode`, default `max`.
- Add `--patience`, default `3`.
- Save:

```text
models/checkpoints/<experiment_name>_best_resnet18.pt
```

Suggested MLflow metrics:

```text
best_epoch
best_val_roc_auc
best_val_f1
```

Decision criterion:

This was implemented before adding CNN + GRU/LSTM. A better checkpoint selection
policy improved the pretrained operating points without changing the architecture.

## Priority 2 - Better Imbalance Handling

Problem:

The temporal frame dataset is highly imbalanced:

```text
0: 7295 frames
1: 465 frames
```

Class-weighted cross entropy helps, but current recall is still weak.

Candidate experiments:

1. WeightedRandomSampler.
2. Focal loss.
3. Positive-frame oversampling.
4. Stronger data augmentation for positive frames.

Recommended experiment names:

```text
temporal_alert_224_pretrained_sampler
temporal_alert_224_pretrained_focal
temporal_alert_224_pretrained_oversample
```

Metrics to prioritize:

```text
alert_recall
false_alarm_rate
alert_precision
mean_alert_time_error
```

Decision criterion:

Keep any method that improves false alarm rate at matched recall, or improves
recall without causing false alarms to explode.

## Priority 3 - Short-Window Temporal Aggregation

Problem:

The current model classifies frames independently. This can produce noisy risk
curves and unstable alerts.

Status:

Completed on 2026-07-07. See:

```text
reports/temporal_aggregation_experiment.md
```

Before implementing GRU/LSTM, test simple temporal post-processing:

- moving average over 3 seconds;
- moving average over 5 seconds;
- rolling max over 3 seconds;
- alert only after N consecutive frames above threshold.

Suggested experiment names:

```text
temporal_alert_224_pretrained_ma3
temporal_alert_224_pretrained_ma5
temporal_alert_224_pretrained_consecutive2
```

Why this matters:

Simple temporal aggregation may reduce false positives and make alert timing more
stable without adding a new neural architecture.

Result:

The 2-consecutive-frame rule improved precision from `0.667` to `0.727` and
reduced false alarm rate from `0.400` to `0.300` while preserving recall `0.80`.

## Priority 4 - CNN + GRU/LSTM

Only start this after:

- best checkpoint selection is implemented;
- imbalance experiments are logged in MLflow;
- short-window temporal aggregation is tested.

Status:

Started on 2026-07-07. See:

```text
reports/cnn_rnn_sequence_experiment.md
```

The first frozen-encoder GRU/LSTM experiments are complete. They validate the
sequence pipeline but do not yet outperform the ResNet18 best-checkpoint model
with the 2-consecutive-frame rule.

Proposed architecture:

```text
frame sequence -> CNN encoder -> feature sequence -> GRU/LSTM -> alert score
```

Possible input design:

- 8 to 16 frames per sequence;
- 1 FPS or 2 FPS;
- positive windows centered around alert/event region;
- negative windows sampled from normal videos.

Important:

The split must remain video-level. Frames or windows from the same video must not
appear in both train and validation.

## Priority 5 - Notebook and Report Updates

Update:

```text
notebooks/07_validation_experiment_analysis.ipynb
reports/experiment_comparison_val.md
reports/methodology.md
reports/temporal_label_progression.md
```

Recommended notebook improvement:

- Load runs directly from MLflow.
- Compare all runs with a single table.
- Generate article-ready charts from MLflow metrics.

## Priority 6 - Product Interface Prototype

Do not use Streamlit as the product interface. The product direction is a
custom web frontend with HTML/Tailwind backed by an API service such as FastAPI.

Build this only after selecting a validation-approved model candidate.

Recommended product features:

- experiment selector;
- video selector;
- risk curve plot;
- threshold slider;
- true alert/event markers;
- predicted alert marker;
- alert metrics summary;
- model comparison table.

The interface can initially consume existing artifacts:

```text
outputs/predictions/*_temporal_risk_scores.csv
outputs/predictions/*_alert_predictions.csv
models/reports/*_alert_metrics.json
```

Later, the API can read MLflow runs and model registry metadata directly.

## Priority 7 - Paper Draft

Start turning the results into the article narrative.

Suggested structure:

1. Introduction.
2. Dataset and task definition.
3. Temporal alert framing.
4. Preprocessing and temporal labeling.
5. Baseline frame-level model.
6. Validation-only protocol.
7. Transfer learning comparison.
8. Threshold and alert-time analysis.
9. Limitations.
10. Future work with sequence models.

Core claim for now:

```text
Temporally meaningful labels and transfer learning improve calibration and false
alarm control, but frame-independent classification remains insufficient for
reliable early collision anticipation.
```

## Recommended Order of Work

1. Run a hard-negative pressure ablation.
2. Completed:
   `product_context_hn_top1000_w15_phase_classifier_seq8`.
3. Next:
   `product_context_hn_top1000_w20_phase_classifier_seq8`.
4. Compare validation alert tradeoff against
   `product_event_window_phase_classifier_seq8`.
5. Compare validation alert tradeoff against
   `product_context_hard_negatives_phase_classifier_seq8`.
6. Only run holdout if a candidate satisfies the validation product gate.
7. Update the experiment comparison notebook and reports from MLflow/local
   artifacts.
8. Start writing the methodology/results sections with failed models as
   negative but informative evidence.
9. Build the HTML/Tailwind/FastAPI product interface after selecting a model
   candidate.

## Resume Here Next Time

If continuing from a fresh day, start with:

```powershell
cd "C:\Users\z004hn4c\Documents\Estudo\LLMOps And AIOps Bootcamp With 8 End To End Projects\nexar-dashcam-collision-prediction"
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then rerun or inspect the current experiments in MLflow.
