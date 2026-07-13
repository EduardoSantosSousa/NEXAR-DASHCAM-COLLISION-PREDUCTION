# Product EfficientNet-B0 partial fine-tuning experiment

Run date: 2026-07-13

## Objective

Test whether partial fine-tuning of EfficientNet-B0 improves validation
separability beyond the full-dataset ResNet18 reference.

Experiment:

```text
product_temporal_alert_224_efficientnet_b0_amp_finetune_last2
```

Reference to beat:

```text
ResNet18 validation ROC-AUC = 0.667
Frozen EfficientNet-B0 validation ROC-AUC = 0.653
```

## Configuration

| Parameter | Value |
| --- | --- |
| Backbone | `efficientnet_b0` |
| Pretrained | `true` |
| AMP enabled | `true` |
| Unfrozen feature blocks | `2` |
| Trainable parameters | `1131954 / 4010110` |
| Batch size | `48` |
| Learning rate | `0.00005` |
| Max epochs | `6` |
| Early stopping patience | `2` |
| Monitor metric | `roc_auc` |

## Training Results

Training stopped early at epoch 3. The best checkpoint was epoch 1.

| Epoch | Train loss | Validation F1 | Validation ROC-AUC | Best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.651 | 0.171 | 0.649 | Yes |
| 2 | 0.590 | 0.170 | 0.649 | No |
| 3 | 0.528 | 0.168 | 0.645 | No |

Interpretation:

- Partial fine-tuning trained successfully and was operationally manageable.
- Training loss decreased substantially, but validation ROC-AUC did not improve.
- The best validation ROC-AUC, `0.649`, is below both the frozen EfficientNet-B0
  run and the ResNet18 reference.
- This indicates overfitting or insufficient visual separability for this
  frame-level formulation.

## Decision

Do not run alert threshold sweeps or holdout evaluation for this checkpoint.

The predefined continuation rule was to proceed to alert-level evaluation only
if validation ROC-AUC exceeded the ResNet18 reference of `0.667`. This model did
not meet that condition.

## Artifact

Best checkpoint:

```text
models/checkpoints/product_temporal_alert_224_efficientnet_b0_amp_finetune_last2_best_efficientnet_b0.pt
```

Metrics:

```text
models/reports/product_temporal_alert_224_efficientnet_b0_amp_finetune_last2_metrics.json
```

## Recommended Next Step

Move away from EfficientNet-B0 for the product candidate path.

Recommended next experiment:

```text
product_temporal_alert_224_convnext_tiny_amp
```

Rationale:

- ResNet18, frozen EfficientNet-B0, and partially fine-tuned EfficientNet-B0 all
  fail to create enough separation between positive and negative videos.
- Threshold tuning cannot fix insufficient score separability.
- A stronger modern visual backbone is now a more useful experiment than
  additional EfficientNet-B0 tuning.

If ConvNeXt-Tiny still fails to exceed the ResNet18 reference, the next article
and product direction should shift toward explicit temporal modeling and
event-centered sequence sampling.
