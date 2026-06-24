# AegisVIO Research Plan

## Title

**AegisVIO: Uncertainty-Aware Visual-Inertial Navigation for Autonomous Robots**

## Core idea

AegisVIO explores how visual-inertial state estimation uncertainty can be used as an active signal for safer robotic navigation.

Most VIO pipelines estimate pose. This project asks a second question:

> How confident is the robot about its pose, and how should navigation change when confidence decreases?

## Initial research questions

1. Can covariance-derived uncertainty metrics predict upcoming VIO degradation?
2. Do uncertainty metrics correlate with visual failure modes such as blur, low texture, fast motion, or low light?
3. Can a simple uncertainty-aware policy reduce navigation risk compared with a pose-only baseline?
4. Can active re-observation or keyframe selection reduce uncertainty growth?

## First milestone

Build a reliable EuRoC MAV data pipeline:

- load camera frames
- load IMU packets
- load ground truth
- run ORB feature detection
- visualize feature matches and inliers

## Planned modules

- `src/data/euroc_loader.py`
- `src/frontend/orb_tracker.py`
- `src/estimation/`
- `src/uncertainty/`
- `src/navigation/`
- `src/evaluation/`

## Evaluation targets

- ATE / RPE
- feature track count
- inlier ratio
- covariance trace
- covariance entropy
- NEES / NIS
- tracking failure rate

## Long-term direction

The strongest PhD-aligned version of the project is:

**Uncertainty-triggered active perception for visual-inertial robotic navigation.**

When uncertainty grows, the robot should not only slow down. It should actively seek better visual information.
