# False positive error analysis - product_event_window_phase_classifier_seq8

## Objective

Identify negative validation videos that repeatedly trigger high alert
scores, then prepare visual artifacts for manual error categorization.

## Configuration

- Threshold used for ranking: `0.640`
- Negative validation videos analyzed: `112`
- Negative videos with at least one false alert: `50`
- False-alert video rate at this threshold: `0.446`
- Figures directory: `C:\Users\z004hn4c\Documents\Estudo\LLMOps And AIOps Bootcamp With 8 End To End Projects\nexar-dashcam-collision-prediction\outputs\figures\product_false_positive_error_analysis`

## Error Type Taxonomy

- `camera_motion`
- `close_vehicle_no_collision`
- `traffic_density`
- `lane_change_or_turn`
- `visual_occlusion`
- `night_or_low_quality`
- `label_or_timing_suspicion`
- `background_bias`
- `unknown`

## Top False Positive Videos

| Rank | Video ID | Max risk | Mean risk | Frames above threshold | First false alert | Segments | Suggested type |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 01599 | 0.761 | 0.640 | 48 | 5.000 | 8 | unknown |
| 2 | 01283 | 0.847 | 0.612 | 37 | 0.000 | 6 | unknown |
| 3 | 01242 | 0.785 | 0.464 | 33 | 0.000 | 4 | unknown |
| 4 | 01581 | 0.764 | 0.614 | 31 | 0.500 | 8 | unknown |
| 5 | 01114 | 0.772 | 0.540 | 30 | 12.500 | 9 | unknown |
| 6 | 01365 | 0.768 | 0.550 | 25 | 19.000 | 1 | unknown |
| 7 | 01271 | 0.736 | 0.463 | 20 | 8.000 | 5 | unknown |
| 8 | 01959 | 0.771 | 0.529 | 18 | 21.000 | 4 | unknown |
| 9 | 01072 | 0.816 | 0.528 | 16 | 3.500 | 8 | unknown |
| 10 | 01711 | 0.733 | 0.503 | 14 | 11.000 | 4 | unknown |
| 11 | 01704 | 0.768 | 0.487 | 13 | 3.500 | 6 | unknown |
| 12 | 01717 | 0.731 | 0.477 | 12 | 0.000 | 4 | unknown |
| 13 | 01075 | 0.724 | 0.480 | 10 | 0.000 | 3 | unknown |
| 14 | 01236 | 0.763 | 0.509 | 9 | 3.000 | 5 | unknown |
| 15 | 01807 | 0.704 | 0.465 | 9 | 18.000 | 5 | unknown |
| 16 | 01420 | 0.690 | 0.502 | 9 | 14.500 | 6 | unknown |
| 17 | 01673 | 0.742 | 0.492 | 8 | 14.000 | 3 | unknown |
| 18 | 02083 | 0.772 | 0.451 | 7 | 26.500 | 6 | unknown |
| 19 | 01287 | 0.718 | 0.385 | 6 | 12.000 | 4 | unknown |
| 20 | 01592 | 0.748 | 0.335 | 5 | 0.000 | 2 | unknown |

## Positive Reference Videos

Use these examples to compare high-risk true positives against false
positive videos.

| Rank | Video ID | Max risk | Mean risk | Frames above threshold | Top risk timestamp |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 00071 | 0.883 | 0.677 | 50 | 2.500 |
| 2 | 00004 | 0.879 | 0.680 | 55 | 30.000 |
| 3 | 00819 | 0.841 | 0.711 | 66 | 30.500 |
| 4 | 00143 | 0.834 | 0.726 | 70 | 12.000 |
| 5 | 00809 | 0.834 | 0.597 | 31 | 12.500 |
| 6 | 00013 | 0.828 | 0.582 | 51 | 16.000 |
| 7 | 00174 | 0.824 | 0.693 | 67 | 1.000 |
| 8 | 00659 | 0.823 | 0.450 | 11 | 6.500 |
| 9 | 00194 | 0.814 | 0.551 | 25 | 29.000 |
| 10 | 00123 | 0.808 | 0.713 | 73 | 19.500 |

## Manual Review Instructions

1. Open each top false-positive folder under the figures directory.
2. Compare the risk curve with the copied top-risk frames.
3. Fill `manual_error_type` and `notes` in the review CSV.
4. Use the dominant error type to choose the next modeling change.

Recommended decision rules:

- many `close_vehicle_no_collision`: mine harder safe negatives;
- many `camera_motion` or `lane_change_or_turn`: add stronger temporal context;
- many `label_or_timing_suspicion`: revise phase windows or labels;
- many visually ambiguous cases: consider risk-level product UX before binary alerting.

## Initial Visual Review

Reviewed the peak-risk frames for the first high-priority false-positive
videos. The current dominant pattern is not random noise: the model is firing on
hard negative scenes that visually resemble pre-collision situations.

Initial reviewed labels:

| Video ID | Initial type | Observation |
| --- | --- | --- |
| 01599 | `traffic_density` | Urban narrow road with many nearby parked/moving vehicles and side occlusions. |
| 01283 | `night_or_low_quality` | Night congested traffic with close leading vehicle and strong brake-light reflections. |
| 01242 | `close_vehicle_no_collision` | Daylight multi-lane traffic with close vehicles ahead and on adjacent lanes. |
| 01581 | `traffic_density` | Dense night urban traffic with multiple nearby vehicles and brake lights. |
| 01114 | `night_or_low_quality` | Low-light stop-and-go traffic with close leading vehicles. |
| 01365 | `night_or_low_quality` | Dusk/night highway traffic with close leading vehicle and brake lights. |
| 01959 | `close_vehicle_no_collision` | Highway scene with close adjacent/front vehicles and degraded visual quality. |
| 01072 | `night_or_low_quality` | Night highway traffic with close adjacent vehicle and leading traffic. |
| 01711 | `traffic_density` | Night urban street with close leading vehicle, parked cars, and strong brake lights. |

Initial interpretation:

- the model is over-sensitive to vehicle proximity, brake lights, dense traffic,
  and low-light scenes;
- these are plausible hard negatives, not obviously broken labels;
- the next data/modeling fix should make these safe-but-risky contexts explicit
  in training instead of only adding generic negative windows.

Recommended next modeling direction:

```text
product_context_hard_negatives_phase_classifier
```

The next manifest should oversample negative windows that match these patterns:

- high-risk negative windows from the current phase-classifier scores;
- nearby frames around each false-positive peak;
- safe positive-video windows that visually resemble traffic density but occur
  before the alertable interval.
