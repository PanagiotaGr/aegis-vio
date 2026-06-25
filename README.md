# AEGIS-VIO

**Uncertainty-Aware Visual-Inertial Navigation in Challenging Environments**

A research-oriented Visual-Inertial Odometry (VIO) pipeline that estimates
not only robot pose but also calibrated estimation **uncertainty**, and
uses that uncertainty to **adapt navigation behavior** (speed, safety
margins, halt/recovery) in real time. Built as a portfolio/thesis-level
project in the spirit of ETH Zurich's Vision for Robotics Lab (V4RL)
research direction on active perception and risk-aware autonomy.

---

## Why uncertainty-aware navigation?

Most VIO/SLAM front-ends silently degrade in motion blur, low light,
feature-poor corridors, or under fast motion — the pose estimate keeps
being published even as its true error grows. AEGIS-VIO instead exposes
a calibrated, real-time **risk score** derived from the EKF error-state
covariance and couples it directly to navigation: the robot slows down,
inflates safety margins, or halts *before* it silently drifts into a
collision, rather than after.

```
 IMU  ──┐
        ├──▶  ErrorStateEKF  ──▶  Uncertainty Metrics  ──▶  Uncertainty-Aware
 Camera─┘     (state + cov)       (trace/det/entropy/        Navigator
              ▲                    risk score)               (speed/margin/
              │                                                halt decisions)
        Feature Tracker
        (vision-quality score)
```

## Repository layout

```
aegis-vio/
├── src/                          # core, ROS2-independent Python library
│   ├── dataset_loader.py         # EuRoC/TUM-VI dataset reader
│   ├── feature_tracker.py        # KLT optical-flow front-end + quality score
│   ├── imu_integrator.py         # strap-down IMU mechanization + Jacobians
│   ├── vio_estimator.py          # front-end + EKF orchestration
│   ├── ekf.py                    # error-state EKF (predict/update)
│   ├── uncertainty.py            # trace/det/entropy/Mahalanobis/NEES/NIS/risk
│   ├── navigation.py             # uncertainty -> speed/margin/mode mapping
│   ├── uncertainty_aware_controller.py  # top-level control loop
│   ├── evaluator.py              # ATE/RPE/consistency metrics
│   └── visualization.py          # publication-quality plots
├── ros2_ws/src/aegis_vio/        # ROS2 package (live operation)
│   ├── aegis_vio/                # vio_node, navigation_node, uncertainty_node
│   ├── launch/aegis_vio.launch.py
│   ├── config/{config,dataset,ekf,navigation}.yaml
│   └── msg/{StateEstimate,UncertaintyMetrics}.msg
├── scripts/                      # run_euroc.py, evaluate_results.py, download_euroc.py
├── tests/                        # pytest unit tests for every src/ module
├── datasets/ models/ results/ plots/   # working directories (gitkept, gitignored content)
├── docs/                         # installation.md, usage.md, experiments.md
├── requirements.txt
├── setup.py
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt && pip install -e .
python scripts/download_euroc.py --sequence MH_01_easy
python scripts/run_euroc.py --dataset_root ./datasets/MH_01_easy/mav0 --output_dir ./results/MH_01_easy
python scripts/evaluate_results.py --results_dir ./results/MH_01_easy --dataset_root ./datasets/MH_01_easy/mav0
pytest tests/ -v
```

See `docs/installation.md` and `docs/usage.md` for full details, and
`docs/experiments.md` for the challenging-environment evaluation protocol
(motion blur, low light, dynamic objects, fast motion, feature-poor
scenes, sensor noise, IMU drift).

## Method summary

- **State estimation:** 16-dim nominal state (position, velocity,
  orientation quaternion, gyro/accel biases), 15-dim error-state
  Extended Kalman Filter following Solà (2017)'s quaternion
  error-state formulation.
- **Front-end:** Shi-Tomasi/KLT pyramidal optical flow with a composite
  vision-quality score (feature count, track age, spatial spread) that
  modulates measurement noise — poorer visual conditions are mapped to
  larger innovation covariance, which is the key mechanism coupling
  front-end health to propagated uncertainty.
- **Uncertainty quantification:** covariance trace/determinant,
  differential entropy, Mahalanobis distance, 95%-confidence ellipsoids,
  and filter-consistency diagnostics (NEES/NIS with chi-square bounds).
- **Navigation:** a thresholded risk-score state machine
  (NORMAL → CAUTIOUS → RECOVERY → HALT) that scales commanded velocity
  and inflates obstacle safety margins; includes a greedy active-
  perception waypoint-selection heuristic.
- **Evaluation:** ATE/RPE via Umeyama Sim(3)/SE(3) alignment, tracking
  failure rate, and NEES/NIS consistency testing, matching standard
  VIO/SLAM benchmarking practice (EuRoC MAV, TUM-VI conventions).

## Status / research extensions

The shipped pseudo-measurement update in `vio_estimator.py` is an
intentionally simplified, documented placeholder for a full multi-view
reprojection-error update (à la MSCKF/OpenVINS). Documented next steps
(see `docs/experiments.md` and the accompanying research proposal):

1. Learned uncertainty prediction (CNN/Transformer mapping image
   patches → predicted observation noise).
2. Full landmark triangulation + reprojection-error EKF/factor-graph
   back-end (GTSAM integration point already scaffolded in
   `requirements.txt`).
3. Active viewpoint selection for next-best-view exploration.
4. Risk-aware trajectory planning (cost-map inflation is scaffolded in
   `navigation.inflate_obstacle_costmap`).
5. Uncertainty-aware keyframe/marginalization selection.

## License

MIT
