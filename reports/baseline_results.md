# Baseline Results

## Objective

The first baseline evaluates whether individual extracted frames contain visual
signals associated with collision or near-collision risk.

## Protocol

The experiment uses the 100-video stratified sample created during the visual
analysis stage. Five frames were extracted per video, resulting in 500 frames.

The split is video-based:

| Split | Videos | Frames |
| --- | ---: | ---: |
| Train | 80 | 400 |
| Validation | 20 | 100 |

Model:

```text
ResNet18 -> binary classification head
```

Training:

| Setting | Value |
| --- | --- |
| Epochs | 3 |
| Batch size | 16 |
| Device | CUDA |
| Pretrained weights | No |

## Initial Results

| Level | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frame | 0.55 | 0.538 | 0.700 | 0.609 | 0.560 |
| Video, mean aggregation | 0.55 | 0.538 | 0.700 | 0.609 | 0.620 |
| Video, max aggregation | 0.45 | 0.467 | 0.700 | 0.560 | 0.530 |

## Interpretation

This result is not intended to be a strong final model. It is a sanity-check
baseline that confirms:

- the GPU training pipeline works;
- the split avoids video leakage;
- predictions can be exported at frame and video level;
- metrics and diagnostic figures are generated reproducibly.

The next improvement should use transfer learning with pretrained ImageNet
weights or a frozen feature extractor before moving to temporal modeling.
