# AegisVIO

Uncertainty-Aware Visual-Inertial Navigation for Autonomous Robots

AegisVIO is a research-oriented robotics project that explores how visual perception quality and inertial sensing can be combined to estimate uncertainty and support safer navigation decisions in autonomous systems.

---

## Research Motivation

Visual-Inertial Odometry (VIO) systems often operate in challenging environments where perception quality may degrade due to motion blur, low texture, illumination changes, or sensor noise.

The goal of AegisVIO is to investigate:

- Visual feature quality estimation
- Uncertainty-aware perception
- Multi-modal uncertainty fusion
- Risk-aware navigation policies
- Autonomous decision making under uncertainty

---

## Current Pipeline

EuRoC MAV Dataset

↓

ORB Feature Tracking

↓

Feature Quality Estimation

↓

Visual Uncertainty Estimation

↓

IMU Stability Analysis

↓

Multi-Modal Uncertainty Fusion

↓

Adaptive Navigation Policy

---

## Dataset

This project currently uses the EuRoC MAV Dataset from ETH Zurich.

Dataset:
https://doi.org/10.3929/ethz-b-000690084

Sequences currently tested:

- MH_01_easy

---

## Implemented Experiments

### Experiment 001
EuRoC Dataset Loading

Results:

- 3682 stereo frames
- 36820 IMU packets
- Ground-truth available

### Experiment 002
Synthetic Feature Tracking

ORB feature extraction and matching on synthetic scenes.

### Experiment 003
Real EuRoC Feature Tracking

Results (MH_01_easy):

- Matches: 1029
- Inlier Ratio: 0.708

### Experiment 004
Feature Quality Analysis

300 frame pairs evaluated.

Results:

- Mean Matches: 871
- Mean Inlier Ratio: 0.716

### Experiment 005
Visual Uncertainty Estimation

Visual uncertainty derived from feature quality degradation.

### Experiment 006
Multi-Modal Uncertainty

Fusion of:

- Visual uncertainty
- IMU instability

---

## Repository Structure

```text
src/
ros2_ws/
scripts/
tests/
datasets/
results/
plots/
docs/



---








 ###nFuture Work
EKF-based uncertainty propagation
Full Visual-Inertial Odometry pipeline
Uncertainty-aware path planning
Risk-aware navigation
ROS2 deployment
Real robot experiments
Author

Panagiota Grosdouli

Electrical & Computer Engineering
Democritus University of Thrace

Research Interests:

Robotics
Visual-Inertial Odometry
Motion Prediction
Autonomous Systems
Uncertainty-Aware AI
