# Product ConvNeXt-Tiny experiment

Run date: 2026-07-13

## Objective

Evaluate whether ConvNeXt-Tiny provides stronger frame-level separability than
the previous product candidates.

Reference to beat before running alert-level evaluation:

```text
Full-dataset ResNet18 validation ROC-AUC = 0.667
```

Experiments:

```text
product_temporal_alert_224_convnext_tiny_frozen_amp
product_temporal_alert_224_convnext_tiny_amp_finetune_last1
```

## Implementation

The baseline CNN pipeline now supports:

```text
resnet18
efficientnet_b0
convnext_tiny
```

ConvNeXt-Tiny uses ImageNet pretrained weights from torchvision and replaces
the final classifier layer with a binary collision head.

## Frozen Feature Extractor

Configuration:

| Parameter | Value |
| --- | --- |
| Backbone | `convnext_tiny` |
| Pretrained | `true` |
| Freeze backbone | `true` |
| AMP enabled | `true` |
| Trainable parameters | `3074 / 27821666` |
| Batch size | `64` |
| Learning rate | `0.0003` |
| Early stopping patience | `2` |

Training results:

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.662 | 0.165 | 0.657 | Yes |
| 2 | 0.642 | 0.167 | 0.653 | No |
| 3 | 0.632 | 0.165 | 0.650 | No |

Best validation ROC-AUC:

```text
0.6566
```

## Partial Fine-Tuning

Configuration:

| Parameter | Value |
| --- | --- |
| Backbone | `convnext_tiny` |
| Pretrained | `true` |
| Unfrozen feature blocks | `1` |
| AMP enabled | `true` |
| Trainable parameters | `14292482 / 27821666` |
| Batch size | `16` |
| Learning rate | `0.00002` |
| Early stopping patience | `2` |

Training results:

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.592 | 0.166 | 0.659 | Yes |
| 2 | 0.406 | 0.162 | 0.651 | No |
| 3 | 0.254 | 0.131 | 0.642 | No |

Best validation ROC-AUC:

```text
0.6594
```

Interpretation:

- ConvNeXt-Tiny improves over EfficientNet-B0 variants, but still does not beat
  the ResNet18 validation reference.
- Partial fine-tuning reduces training loss quickly, but validation ROC-AUC
  degrades after epoch 1.
- This is another sign that the current frame-level formulation overfits before
  learning a product-grade alert signal.

## Decision

Do not run alert threshold sweeps or holdout evaluation for these ConvNeXt-Tiny
checkpoints.

The continuation rule was:

```text
Run alert-level evaluation only if validation ROC-AUC > 0.667.
```

Neither ConvNeXt-Tiny run reached that threshold.

## Artifacts

Best frozen checkpoint:

```text
models/checkpoints/product_temporal_alert_224_convnext_tiny_frozen_amp_best_convnext_tiny.pt
```

Best partial fine-tuning checkpoint:

```text
models/checkpoints/product_temporal_alert_224_convnext_tiny_amp_finetune_last1_best_convnext_tiny.pt
```

Metrics:

```text
models/reports/product_temporal_alert_224_convnext_tiny_frozen_amp_metrics.json
models/reports/product_temporal_alert_224_convnext_tiny_amp_finetune_last1_metrics.json
```

## Recommended Next Step

Stop testing stronger frame-level backbones as isolated image classifiers.

The evidence now points to the formulation rather than only backbone capacity:

- ResNet18: best validation ROC-AUC `0.667`
- EfficientNet-B0 frozen: best validation ROC-AUC `0.653`
- EfficientNet-B0 partial fine-tuning: best validation ROC-AUC `0.649`
- ConvNeXt-Tiny frozen: best validation ROC-AUC `0.657`
- ConvNeXt-Tiny partial fine-tuning: best validation ROC-AUC `0.659`

Recommended next modeling direction:

```text
event-centered temporal dataset + sequence/video-level model
```

The next experiment should sample short windows around the pre-alert interval
and train/evaluate at sequence level, instead of treating every frame as an
independent classification example.
