# Experiments

## Experiment 001 — EuRoC loading

Goal: verify that camera, IMU, and ground-truth metadata load correctly.

```bash
python scripts/run_euroc.py --sequence datasets/euroc/MH_01_easy
```

## Experiment 002 — Feature tracking

Goal: evaluate visual feature quality and inlier ratio between consecutive frames.

## Experiment 003 — Uncertainty-aware control

Goal: connect covariance-derived uncertainty to navigation modes:

- nominal
- cautious
- recovery

## Future experiments

- motion blur stress test
- low-light stress test
- feature-poor scene test
- covariance consistency analysis
- uncertainty-aware navigation ablation
