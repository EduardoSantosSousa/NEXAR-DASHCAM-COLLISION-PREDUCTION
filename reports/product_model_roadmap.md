# Product-grade model and interface roadmap

Created: 2026-07-09

## Why The Project Needs A Pivot

The current best experimental model is:

```text
temporal_alert_224_pretrained_best + 2 consecutive frames
threshold = 0.13
precision = 0.727
recall = 0.800
false_alarm_rate = 0.300
mean_alert_time_error = -7.991 s
```

This is useful for an article baseline, but it is not yet enough for a product.

The main reason is evaluation quality:

- most strong experiments so far used a 100-video sample;
- validation used only 20 videos;
- thresholds were selected on that same validation split;
- sequence models did not beat the frame-level baseline with temporal
  post-processing.

For a product, the next priority is not another small GRU/LSTM tweak. The next
priority is to create a reliable full-dataset modeling and evaluation pipeline.

## Product Split

A full train/validation/holdout split was created with:

```powershell
.\venv\Scripts\python.exe scripts\create_product_split.py --val-size 0.15 --holdout-size 0.15 --random-state 42
```

Output:

```text
data/interim/full_train_product_splits.csv
```

Split summary:

| Split | Negative videos | Positive videos | Total |
| --- | ---: | ---: | ---: |
| train | 526 | 526 | 1052 |
| val | 112 | 112 | 224 |
| holdout | 112 | 112 | 224 |

Rules:

- Use `train` for fitting model parameters.
- Use `val` for checkpoint selection, threshold selection, calibration, and
  iteration.
- Use `holdout` only once a candidate is selected.
- Never tune threshold or architecture after seeing holdout results unless a new
  holdout is created.

## Reliability Criteria For A Product Candidate

A model should not be called "final" unless it passes a stronger standard than
the current sample-level experiments.

Minimum candidate criteria on validation:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.25
alert_precision >= 0.75
```

Preferred candidate criteria on validation:

```text
alert_recall >= 0.85
false_alarm_rate <= 0.20
alert_precision >= 0.80
```

Final holdout gate:

```text
alert_recall >= 0.80
false_alarm_rate <= 0.30
alert_precision >= 0.70
```

The holdout gate is intentionally close to the current best sample baseline. If
the model cannot preserve that tradeoff on 224 unseen holdout videos, it should
not be treated as product-ready.

## Modeling Strategy

### Stage 1 - Full-Dataset Baseline

First, rerun the strongest current baseline on the full split:

- temporal frame labels;
- ImageNet-pretrained ResNet18;
- best checkpoint selection;
- validation-only threshold sweep;
- 2-consecutive-frame post-processing;
- final holdout evaluation only after selecting the validation threshold.

This answers the most important question:

```text
Does the current best baseline survive when trained and evaluated on the full dataset?
```

Recommended data build:

```powershell
.\venv\Scripts\python.exe scripts\create_temporal_frame_dataset.py `
  --sample-csv data\interim\full_train_product_splits.csv `
  --fps 2 `
  --pre-alert-margin 3 `
  --image-size 224 `
  --output-dir data\interim\product_temporal_frames_224 `
  --manifest data\interim\product_temporal_frames_224_manifest.csv
```

Recommended training:

```powershell
.\venv\Scripts\python.exe scripts\train_baseline.py `
  --manifest data\interim\product_temporal_frames_224_manifest.csv `
  --split-manifest data\interim\full_train_product_splits.csv `
  --experiment-name product_temporal_alert_224_pretrained `
  --pretrained `
  --epochs 8 `
  --batch-size 64 `
  --learning-rate 0.00005 `
  --num-workers 2 `
  --monitor-metric roc_auc `
  --monitor-mode max `
  --patience 3
```

Recommended validation evaluation:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py `
  --checkpoint models\checkpoints\product_temporal_alert_224_pretrained_best_resnet18.pt `
  --experiment-name product_temporal_alert_224_pretrained_best `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split val `
  --fps 1 `
  --threshold 0.5 `
  --batch-size 32
```

Recommended validation threshold sweeps:

```powershell
.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py `
  --experiment-name product_temporal_alert_224_pretrained_best `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split val

.\venv\Scripts\python.exe scripts\sweep_alert_thresholds.py `
  --experiment-name product_temporal_alert_224_pretrained_best `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split val `
  --min-consecutive-frames 2
```

After choosing the validation threshold, run holdout once:

```powershell
.\venv\Scripts\python.exe scripts\evaluate_model.py `
  --checkpoint models\checkpoints\product_temporal_alert_224_pretrained_best_resnet18.pt `
  --experiment-name product_temporal_alert_224_pretrained_best_holdout `
  --sample-csv data\interim\full_train_product_splits.csv `
  --split holdout `
  --fps 1 `
  --threshold SELECTED_VAL_THRESHOLD `
  --batch-size 32
```

### Stage 2 - Stronger Image Backbone

If the full-dataset ResNet18 baseline is not good enough, the next modeling
change should be a stronger image backbone, not another GRU variant.

Recommended candidates:

- `efficientnet_b0` as the first performance/latency upgrade;
- `efficientnet_b3` if GPU memory allows;
- `convnext_tiny` if latency is acceptable.

Why:

- the current best model is frame-level plus temporal post-processing;
- sequence models did not improve alert tradeoffs;
- a better visual encoder may improve risk score separation before adding
  temporal complexity.

Implementation requirement:

```text
src/nexar_collision/models/baseline_cnn.py
```

should support a `--backbone` argument before running this stage.

### Stage 3 - Product-Oriented Temporal Model

Only return to temporal neural models after changing the data formulation.

The next temporal attempt should use:

- event-centered positive windows;
- hard-negative windows from negative videos and early safe segments;
- video-level checkpoint selection;
- validation alert metrics as the final selection criterion.

Avoid selecting sequence checkpoints only by frame-level ROC-AUC. The previous
experiments showed that frame-level ROC-AUC can improve without producing a
better alert system.

Current Stage 3 status:

- `product_event_window_gru_seq8_resnet18_frozen` improved the low-false-alarm
  operating point but failed the product gate.
- `product_event_window_gru_seq8_hard_negative` modestly improved the
  low-false-alarm region again, reaching precision `0.696`, recall `0.634`, and
  false alarm rate `0.277` with 2 consecutive frames at threshold `0.59`.
- `product_event_window_gru_seq8_alert_weighted` added hard-negative sample
  weighting plus alert-aware checkpoint selection. It reduced full-video false
  alarm rate at threshold `0.50` to `0.607` while keeping recall `0.830`, but it
  still failed the product gate.
- `product_event_window_gru_seq8_alert_weighted_layer4` fine-tuned the final
  ResNet18 visual block with conservative learning rates. It overfit quickly:
  the best checkpoint was epoch 1, and full-video recall at threshold `0.50`
  fell to `0.446`.
- `product_event_window_gru_seq8_dense_negatives` increased negative-video
  windows to 20 per video and positive-safe windows to 8 per positive video. It
  improved coverage but still failed the product gate: the best low-false-alarm
  point with 2 consecutive frames reached precision `0.677`, recall `0.598`,
  and false alarm rate `0.286`.
- `product_event_window_gru_seq8_prealert_phases` improved the temporal target
  definition by splitting positive videos into safe, early pre-alert, late
  pre-alert, and event-near windows. It improved window-level ROC-AUC to
  `0.749`, but still failed the product gate. The best low-false-alarm point
  with 2 consecutive frames reached precision `0.677`, recall `0.580`, and
  false alarm rate `0.277`.
- `product_event_window_phase_classifier_seq8` predicted four temporal phases
  directly and used `P(prealert_late) + P(event_near)` as the alert score. It
  failed the product gate: at threshold `0.50`, full-video precision was
  `0.544`, recall `0.884`, and false alarm rate `0.741`. The best
  low-false-alarm point with 2 consecutive frames reached precision `0.648`,
  recall `0.509`, and false alarm rate `0.277`.
- No current candidate satisfies recall `>= 0.80`, false alarm rate `<= 0.30`,
  and precision `>= 0.70` on validation.
- The holdout remains sealed.

Current Stage 3 priority:

```text
product_context_hard_negatives_phase_classifier
```

False-positive analysis showed that negative validation videos trigger high
alert scores mostly in hard negative contexts: dense traffic, close vehicles,
brake lights, low-light scenes, and visual degradation. The next data
formulation should explicitly mine and oversample these contextual hard
negatives.

Contextual hard-negative mining completed:

- Created `scripts/mine_context_hard_negative_windows.py`.
- Scored 526 train negative videos from
  `product_event_window_phase_classifier_seq8`.
- Selected 513 high-risk peaks from 240 train negative videos.
- Added 2202 contextual hard-negative windows with `phase_index=0` and
  `sample_weight=2.5`.
- Created:
  `data/interim/product_event_windows_seq8_context_hard_negatives_manifest.csv`.
- Documented the mining run in:
  `reports/product_context_hard_negative_mining.md`.

Contextual hard-negative training result:

- Trained:
  `product_context_hard_negatives_phase_classifier_seq8`.
- Best checkpoint: epoch 1, selected by `alert_selection_score`.
- Best validation-window alert proxy:
  - precision `0.743`;
  - recall `0.696`;
  - false alarm rate `0.241`.
- Full-video temporal alert evaluation at threshold `0.50`:
  - precision `0.523`;
  - recall `0.812`;
  - false alarm rate `0.741`.
- Best full-video low-false-alarm point with 2 consecutive frames:
  - precision `0.674`;
  - recall `0.554`;
  - false alarm rate `0.268`.
- The completed experiment analysis is documented in:
  `reports/product_context_hard_negatives_phase_classifier_seq8_experiment.md`.

Decision:

Do not evaluate on holdout. The mined negatives modestly improved the
low-false-alarm region, but still failed the product gate. The next step should
be a lighter hard-negative ablation to recover recall.

Hard-negative pressure ablation prepared:

- Created `scripts/create_hard_negative_ablation_manifests.py`.
- Created three ablation manifests:
  - `data/interim/product_event_windows_seq8_context_hn_all_w15_manifest.csv`;
  - `data/interim/product_event_windows_seq8_context_hn_top1000_w20_manifest.csv`;
  - `data/interim/product_event_windows_seq8_context_hn_top1000_w15_manifest.csv`.
- All ablation manifests contain `holdout_rows = 0`.
- Documented the plan and commands in:
  `reports/product_context_hard_negative_ablation_plan.md`.

Hard-negative pressure ablation C result:

- Trained:
  `product_context_hn_top1000_w15_phase_classifier_seq8`.
- Best checkpoint: epoch 2, selected by `alert_selection_score`.
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

Do not evaluate on holdout. Variant C reduced pressure too much and became
over-sensitive at lower thresholds. The next ablation should try the same
top-1000 volume with stronger weight `2.0`.

Phase-classifier implementation and analysis:

- `scripts/train_sequence_model.py` now supports `--num-classes`,
  `--target-column`, and `--alert-class-indices`.
- `scripts/evaluate_sequence_model.py` now supports phase-model alert scoring.
- The execution plan and commands are documented in:
  `reports/product_event_window_phase_classifier_seq8_next_step.md`.
- The completed experiment analysis is documented in:
  `reports/product_event_window_phase_classifier_seq8_experiment.md`.

## Calibration And Thresholding

For product use, threshold selection must be explicit.

Required artifacts for a product candidate:

```text
models/reports/<experiment>_alert_threshold_sweep.csv
models/reports/<experiment>_chosen_threshold.json
models/reports/<experiment>_holdout_alert_metrics.json
```

The chosen threshold file should include:

```json
{
  "experiment_name": "product_temporal_alert_224_pretrained_best",
  "selected_on": "val",
  "threshold": 0.13,
  "post_processing": "min_consecutive_frames=2",
  "selection_rule": "recall>=0.80, minimize false_alarm_rate, maximize precision",
  "do_not_tune_on_holdout": true
}
```

Optional but recommended:

- temperature scaling or isotonic calibration on validation;
- bootstrap confidence intervals for alert recall and false alarm rate;
- per-scenario error analysis for false positives and missed events.

## Product Architecture Without Streamlit

The product should not use Streamlit.

Recommended architecture:

```text
frontend/
  Vite
  HTML
  Tailwind CSS
  TypeScript

backend/
  FastAPI
  PyTorch or ONNX Runtime inference
  background video processing job
  static artifact serving
```

Core product flow:

1. User uploads a dashcam video.
2. Backend extracts frames at the configured FPS.
3. Model scores each frame.
4. Backend applies calibrated threshold and temporal post-processing.
5. Frontend displays:
   - uploaded video;
   - risk curve;
   - selected threshold;
   - predicted alert timestamp;
   - confidence/status;
   - downloadable JSON report.

Recommended frontend screens:

- upload/analyze screen;
- analysis result screen;
- risk curve timeline;
- model/version metadata panel;
- threshold and post-processing explanation;
- batch/history screen for previous analyses.

Recommended backend endpoints:

```text
POST /api/analyses
GET  /api/analyses/{analysis_id}
GET  /api/analyses/{analysis_id}/risk-scores
GET  /api/analyses/{analysis_id}/report
GET  /api/model-info
```

Model packaging path:

1. start with PyTorch `.pt` inference for correctness;
2. export to TorchScript or ONNX after metrics are stable;
3. benchmark latency on representative videos;
4. only then optimize frontend/backend performance.

## Immediate Next Steps

1. Build a lighter hard-negative ablation.
2. Train the prepared variants in this order:
   - completed: `product_context_hn_top1000_w15_phase_classifier_seq8`;
   - next: `product_context_hn_top1000_w20_phase_classifier_seq8`;
   - `product_context_hn_all_w15_phase_classifier_seq8`.
3. Keep phase-classifier alert score as `P(prealert_late) + P(event_near)`.
4. Use validation-only threshold selection and compare against
   `product_context_hard_negatives_phase_classifier_seq8`.
5. Evaluate the selected candidate on holdout only if validation passes the
   product gate.
6. If holdout passes the product gate, freeze the model version.
7. If validation still fails, revisit the target formulation before adding
   product UI polish.
8. Start the custom HTML/Tailwind/FastAPI product only after a candidate model is
   selected, or build it against the current baseline as a prototype with a clear
   "experimental model" label.

## Current Decision

No current model is product-grade.

The strongest current direction is the event-window GRU family, but the base,
hard-negative, alert-weighted, layer4, dense-negative, binary pre-alert phase,
phase-classifier, and contextual hard-negative variants all fail the validation
product gate. The next work should tune hard-negative pressure before claiming
model reliability or investing heavily in UI polish.
