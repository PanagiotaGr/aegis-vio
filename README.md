# AegisVIO

AegisVIO is a research-oriented robotics framework that explores visual-inertial state estimation, uncertainty quantification and risk-aware navigation for autonomous robotic systems.

## Research Motivation

Autonomous robots operating in the real world must make decisions under uncertainty.

While most navigation systems estimate robot pose, they often ignore the confidence of these estimates.

AegisVIO investigates how uncertainty estimates can be integrated into perception and navigation pipelines to improve robustness and safety.





aegis-vio/
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── feature_tracker.py
│   ├── imu_integrator.py
│   ├── vio_estimator.py
│   ├── ekf.py
│   ├── uncertainty.py
│   ├── navigation.py
│   ├── uncertainty_aware_controller.py
│   ├── evaluator.py
│   └── visualization.py
├── ros2_ws/
│   └── src/
│       └── aegis_vio/
│           ├── aegis_vio/
│           │   ├── __init__.py
│           │   ├── vio_node.py
│           │   ├── navigation_node.py
│           │   └── uncertainty_node.py
│           ├── launch/
│           │   └── aegis_vio.launch.py
│           ├── config/
│           │   ├── config.yaml
│           │   ├── dataset.yaml
│           │   ├── ekf.yaml
│           │   └── navigation.yaml
│           ├── msg/
│           │   ├── StateEstimate.msg
│           │   └── UncertaintyMetrics.msg
│           ├── package.xml
│           ├── setup.py
│           └── setup.cfg
├── scripts/
│   ├── run_euroc.py
│   ├── evaluate_results.py
│   └── download_euroc.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset_loader.py
│   ├── test_feature_tracker.py
│   ├── test_ekf.py
│   ├── test_uncertainty.py
│   ├── test_navigation.py
│   └── test_evaluator.py
├── datasets/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── plots/
│   └── .gitkeep
├── docs/
│   ├── installation.md
│   ├── usage.md
│   └── experiments.md
├── requirements.txt
├── setup.py
└── README.md

## Research Areas

- Visual-Inertial Odometry
- State Estimation
- Probabilistic Robotics
- Uncertainty Quantification
- Autonomous Navigation
- Active Perception
- Risk-Aware Planning

## Planned Features

- Visual-Inertial State Estimation
- Covariance Propagation
- Uncertainty Monitoring
- Risk-Aware Navigation
- Active Re-observation
- Evaluation on EuRoC MAV

## Author

Panagiota Grosdouli
