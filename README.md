<div align="center">

#  AEGIS-VIO

### Uncertainty-Aware Visual-Inertial Navigation in Challenging Environments

*A robot that doesn't just estimate where it is — it knows how much to trust itself.*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble%20%7C%20Jazzy-22314E.svg)](https://docs.ros.org/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#license)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#status--roadmap)

</div>

---

##  The idea

Almost every VIO/SLAM system publishes a pose estimate every frame — even
when that estimate has quietly become garbage. Motion blur, low light,
a feature-poor corridor, an aggressive turn: the numbers keep coming out,
but nobody's watching whether they can still be trusted.

**AEGIS-VIO closes that loop.** It propagates a full probabilistic state
estimate, turns the covariance into a single calibrated **risk score**,
and feeds that risk score straight back into the robot's behavior:
slow down, widen the safety margin, stop and re-localize — *before* the
drift becomes a collision, not after.

```
                 ┌─────────────┐      ┌──────────────────┐      ┌─────────────────────┐
   IMU  ───────▶ │             │      │                  │      │                     │
                 │  Error-State├─────▶│   Uncertainty     ├─────▶│  Uncertainty-Aware  │
 Camera ───────▶ │     EKF     │      │   Quantification  │      │     Navigator       │
                 │             │      │  trace · entropy  │      │  speed · margin ·   │
                 └──────▲──────┘      │  NEES/NIS · risk   │      │  halt / recovery    │
                        │             └──────────────────┘      └─────────────────────┘
                 ┌──────┴──────┐
                 │   Feature    │
                 │   Tracker    │   ← vision-quality score modulates
                 │ (KLT + age + │     measurement noise in real time
                 │   spread)    │
                 └─────────────┘
```

##  Quick start

```bash
git clone https://github.com/PanagiotaGr/aegis-vio.git
cd aegis-vio
pip install -r requirements.txt && pip install -e .

# grab a sequence and run the full pipeline
python scripts/download_euroc.py --sequence MH_01_easy
python scripts/run_euroc.py --dataset_root ./datasets/MH_01_easy/mav0 --output_dir ./results/MH_01_easy
python scripts/evaluate_results.py --results_dir ./results/MH_01_easy --dataset_root ./datasets/MH_01_easy/mav0

pytest tests/ -v
```

Want it running live on a robot? `ros2 launch aegis_vio aegis_vio.launch.py` — see [`docs/usage.md`](docs/usage.md).

## What's actually inside

| Layer | What it does | Key file(s) |
|---|---|---|
| **State estimation** | 15-dim error-state EKF (Solà 2017 quaternion formulation) propagating position, velocity, attitude, and IMU biases | `src/ekf.py`, `src/imu_integrator.py` |
| **Visual front-end** | Pyramidal KLT tracking with a composite quality score (count + track age + spatial spread) that directly modulates measurement noise | `src/feature_tracker.py`, `src/vio_estimator.py` |
| **Uncertainty quantification** | Covariance trace/determinant, differential entropy, Mahalanobis distance, 95% confidence ellipsoids, NEES/NIS consistency testing | `src/uncertainty.py` |
| **Risk-aware navigation** | Thresholded state machine — `NORMAL → CAUTIOUS → RECOVERY → HALT` — scaling speed and safety margins; greedy active-viewpoint selection | `src/navigation.py` |
| **Evaluation** | ATE/RPE via Umeyama Sim(3) alignment, tracking-failure rate, NEES/NIS chi-square bounds | `src/evaluator.py` |
| **Live operation** | Full ROS2 package — VIO node, uncertainty node, navigation node, launch file, custom messages | `ros2_ws/src/aegis_vio/` |

##  Repository layout

```
aegis-vio/
├── src/                          # core, ROS2-independent Python library
│   ├── dataset_loader.py         # EuRoC / TUM-VI dataset reader
│   ├── feature_tracker.py        # KLT optical-flow front-end + quality score
│   ├── imu_integrator.py         # strap-down IMU mechanization + Jacobians
│   ├── vio_estimator.py          # front-end + EKF orchestration
│   ├── ekf.py                    # error-state EKF (predict / update)
│   ├── uncertainty.py            # trace · det · entropy · Mahalanobis · NEES · NIS · risk
│   ├── navigation.py             # uncertainty → speed / margin / mode mapping
│   ├── uncertainty_aware_controller.py   # top-level control loop
│   ├── evaluator.py              # ATE / RPE / consistency metrics
│   └── visualization.py          # publication-quality plots
├── ros2_ws/src/aegis_vio/        # ROS2 package (live operation)
│   ├── aegis_vio/                # vio_node · navigation_node · uncertainty_node
│   ├── launch/aegis_vio.launch.py
│   ├── config/{config,dataset,ekf,navigation}.yaml
│   └── msg/{StateEstimate,UncertaintyMetrics}.msg
├── scripts/                      # run_euroc.py · evaluate_results.py · download_euroc.py
├── tests/                        # pytest unit tests for every src/ module
├── datasets/ models/ results/ plots/   # working directories
├── docs/                         # installation.md · usage.md · experiments.md
├── requirements.txt
├── setup.py
└── README.md
```

## Testing

35 unit tests cover the EKF prediction/update cycle, IMU mechanization,
every uncertainty metric, the navigation state machine, and the
evaluation pipeline:

```bash
pytest tests/ -v --cov=src
```

##  Experimental protocol

`docs/experiments.md` specifies seven challenging-environment stress
tests — **motion blur, low light, dynamic objects, fast motion,
feature-poor scenes, sensor noise, IMU drift** — each with a hypothesis,
metrics, expected outcome, and failure mode, plus the statistical
reporting protocol (N≥5 repeats, paired significance testing against a
naive constant-speed baseline).

##  Status & roadmap

The shipped measurement update in `vio_estimator.py` is an intentionally
simplified, clearly documented placeholder — a stepping stone toward a
full multi-view reprojection-error back-end (à la MSCKF / OpenVINS).
Planned extensions:

1. **Learned uncertainty prediction** — CNN/Transformer mapping image
   patches directly to predicted observation noise.
2. **Full landmark triangulation + factor-graph back-end** (GTSAM
   integration point already scaffolded).
3. **Active viewpoint selection** for next-best-view exploration.
4. **Risk-aware trajectory planning** (cost-map inflation already
   scaffolded in `navigation.inflate_obstacle_costmap`).
5. **Uncertainty-aware keyframe/marginalization selection.**

## 🙏Acknowledgements

Built in the spirit of the visual-inertial state estimation, active
perception, and risk-aware autonomy research direction pursued by
robotics and computer vision groups publishing at CVPR, ICRA, IROS, RSS,
and CoRL.

## License

MIT
