# Experiment 001 — Synthetic Uncertainty-Aware Navigation

## Purpose

This experiment validates the central AegisVIO research idea before running on a full visual-inertial dataset.

The goal is to test whether a robot can use an uncertainty score to detect degraded perception and switch between navigation modes.

## Research question

Can uncertainty estimates act as an early warning signal for navigation risk?

## Method

A synthetic 2D trajectory is generated. The estimator receives noisy pseudo-visual measurements. During selected time windows, visual quality is degraded to simulate:

- motion blur
- low light
- feature-poor scenes
- temporary visual failure

The estimator logs:

- ground-truth position
- estimated position
- position error
- covariance trace
- visual quality
- uncertainty score
- navigation mode

## Navigation modes

- `nominal`: normal confidence
- `cautious`: increased uncertainty
- `recovery`: severe uncertainty / degraded perception

## How to run

```bash
python scripts/run_uncertainty_demo.py
```

## Outputs

```text
results/synthetic_uncertainty_demo/synthetic_uncertainty_results.csv
plots/synthetic_uncertainty_demo/trajectory.png
plots/synthetic_uncertainty_demo/error_uncertainty.png
plots/synthetic_uncertainty_demo/visual_quality.png
```

## Expected interpretation

When visual quality decreases, the uncertainty score should increase. If the uncertainty score rises before or together with position error, it can be used as a signal for risk-aware navigation.

This experiment is not the final VIO system. It is a controlled validation of the uncertainty-aware decision layer.
