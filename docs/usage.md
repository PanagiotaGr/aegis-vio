# Usage

## Check EuRoC loading

```bash
python scripts/run_euroc.py --sequence datasets/euroc/MH_01_easy
```

## Evaluate trajectory CSV

```bash
python scripts/evaluate_results.py --csv results/trajectory.csv
```

## ROS2 package

The ROS2 workspace is located in:

```text
ros2_ws/src/aegis_vio/
```

The ROS2 nodes are initial scaffolds and will be connected to the Python research modules as the estimator matures.
