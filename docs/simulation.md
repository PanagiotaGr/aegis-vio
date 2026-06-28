# Synthetic simulation

This project can be demonstrated without downloading a full visual-inertial dataset by running a deterministic synthetic simulation.

The simulation generates a smooth 2-D reference trajectory, injects odometry drift, reduces visual quality during challenging intervals, propagates a simple covariance model, computes an uncertainty-derived risk score, and maps that risk into navigation behavior.

## Run

```bash
python scripts/run_simulation.py --scenario mixed --output_dir results/simulation
```

Other scenarios:

```bash
python scripts/run_simulation.py --scenario nominal
python scripts/run_simulation.py --scenario blur
python scripts/run_simulation.py --scenario low_light
python scripts/run_simulation.py --scenario feature_poor
```

Useful options:

```bash
python scripts/run_simulation.py --scenario mixed --steps 700 --dt 0.1 --seed 11
```

## Outputs

The output directory contains:

- `simulation_results.csv`: step-by-step state, visual quality, covariance trace, entropy, risk, navigation mode, speed scale, safety-margin scale, and position error.
- `risk_quality.png`: risk response compared with visual quality.
- `position_error.png`: localization error over time.
- `trajectory.png`: ground-truth and estimated 2-D trajectories.

## Interpretation

The core research idea is that degraded perception should not only affect pose estimation; it should also change robot behavior. When synthetic visual quality drops, covariance increases. The risk score rises, and the policy moves from `NORMAL` to `CAUTIOUS`, `RECOVERY`, or `HALT`.

This is not a replacement for EuRoC/TUM-VI benchmarking. It is a lightweight demonstration for presentations, debugging, and early validation of the uncertainty-aware navigation loop.
