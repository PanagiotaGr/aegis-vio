# Experiments

AEGIS-VIO is evaluated as an uncertainty-aware visual-inertial navigation system. The experimental protocol focuses on two questions:

1. **Estimation quality:** how much localization error appears under nominal and degraded visual conditions?
2. **Risk response:** does the uncertainty signal correctly trigger cautious, recovery, or halt behavior when perception becomes unreliable?

## 1. Synthetic stress-test suite

The synthetic suite is the fastest way to produce reproducible, presentation-ready results without downloading external datasets. It generates a 2-D reference trajectory, injects environment-dependent visual degradation, propagates covariance, computes risk, and maps risk to navigation modes.

### Run all scenarios

```bash
python scripts/run_experiment_suite.py --seeds 0 1 2 3 4 --plots
```

### Outputs

```text
results/experiment_suite/
├── summary_metrics.csv      # aggregate metrics for every scenario/seed
├── summary_report.md        # paper-style summary table and interpretation
└── runs/
    ├── nominal_seed_0/
    │   ├── trace.csv
    │   ├── risk_quality.png
    │   ├── position_error.png
    │   └── trajectory.png
    └── ...
```

### Scenarios

| Scenario | Description | Expected behavior |
|---|---|---|
| `nominal` | High visual quality baseline | Low risk, mostly `NORMAL` mode |
| `blur` | Temporary feature degradation due to motion blur | Risk increase during blur interval |
| `low_light` | Poor illumination reduces visual evidence | Longer cautious/recovery behavior |
| `feature_poor` | Textureless scene weakens feature tracking | Strong uncertainty growth |
| `mixed` | Combined blur, low-light and feature-poor intervals | Most complete stress test |

### Metrics

| Metric | Meaning |
|---|---|
| `mean_position_error_m` | Average localization error |
| `max_position_error_m` | Worst-case localization error |
| `mean_risk` / `max_risk` | Normalized uncertainty-derived risk |
| `high_risk_ratio` | Fraction of steps with risk ≥ 0.65 |
| `cautious_ratio` | Fraction of steps in cautious mode |
| `recovery_ratio` | Fraction of steps in recovery mode |
| `halt_ratio` | Fraction of steps in halt mode |

## 2. Single-scenario simulation

For a quick demo:

```bash
python scripts/run_simulation.py --scenario mixed --output_dir results/simulation
```

This creates a trace CSV and three plots:

- `risk_quality.png`: visual quality versus risk response
- `position_error.png`: localization error over time
- `trajectory.png`: ground truth versus estimated trajectory

## 3. Real-dataset protocol

The real-data evaluation should use EuRoC and, later, TUM-VI sequences.

### EuRoC loading check

```bash
python scripts/run_euroc.py --sequence datasets/euroc/MH_01_easy
```

Goal: verify that camera, IMU, and ground-truth metadata load correctly.

### Trajectory evaluation

```bash
python scripts/evaluate_results.py --csv results/trajectory.csv
```

Expected CSV columns:

```text
est_x, est_y, est_z, gt_x, gt_y, gt_z
```

## 4. Presentation recommendations

For a clear presentation, show results in this order:

1. **Nominal scenario:** demonstrates that the system behaves normally when perception is reliable.
2. **Mixed scenario:** demonstrates the full uncertainty-response loop.
3. **Mode-ratio table:** shows how often the system enters `CAUTIOUS`, `RECOVERY`, and `HALT`.
4. **Ablation comparison:** compare uncertainty-aware behavior against a naive constant-speed baseline.

## 5. Future experiments

- Motion blur stress test on real EuRoC/TUM-VI sequences.
- Low-light stress test with brightness-degraded image streams.
- Feature-poor scene test using masked or texture-reduced images.
- NEES/NIS covariance consistency analysis.
- Uncertainty-aware navigation ablation against constant-speed navigation.
- Learned visual uncertainty prediction from image patches.
