#!/usr/bin/env python3
"""
Main script to run AEGIS-VIO on EuRoC dataset.

Usage:
    python scripts/run_euroc.py --dataset /path/to/euroc --sequence MH_01_easy
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset_loader import EuRoCDatasetLoader
from feature_tracker import FeatureTracker
from vio_estimator import VIOEstimator, VIOConfig
from ekf import EKFConfig
from evaluator import TrajectoryEvaluator
from visualization import Visualizer
from uncertainty_aware_controller import UncertaintyAwareController


def parse_args():
    parser = argparse.ArgumentParser(description='Run AEGIS-VIO on EuRoC dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to EuRoC dataset root')
    parser.add_argument('--sequence', type=str, default='MH_01_easy',
                        help='Sequence name (default: MH_01_easy)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization plots')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to output directory')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("AEGIS-VIO: Uncertainty-Aware Visual-Inertial Odometry")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output) / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    print(f"Sequence: {args.sequence}")
    
    try:
        loader = EuRoCDatasetLoader(args.dataset, load_images=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download the EuRoC dataset from:")
        print("[projects.asl.ethz.ch](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)")
        return 1
    
    # Check available sequences
    available = loader.list_available_sequences()
    if not available:
        print("No sequences found in dataset directory.")
        return 1
    
    print(f"Available sequences: {available}")
    
    if args.sequence not in available:
        print(f"Sequence '{args.sequence}' not found.")
        return 1
    
    # Load sequence
    print(f"\nLoading sequence data...")
    sequence = loader.load_sequence(args.sequence)
    
    print(f"  IMU measurements: {len(sequence.imu_measurements)}")
    print(f"  Camera frames (cam0): {len(sequence.camera_frames.get(0, []))}")
    print(f"  Ground truth poses: {len(sequence.ground_truth)}")
    
    # Get camera calibration
    cam_calib = sequence.camera_calibrations.get(0)
    if cam_calib is None:
        print("Error: Camera calibration not found")
        return 1
    
    # Configure VIO
    ekf_config = EKFConfig(
        gyro_noise=sequence.imu_calibration.gyroscope_noise_density if sequence.imu_calibration else 1.6968e-4,
        accel_noise=sequence.imu_calibration.accelerometer_noise_density if sequence.imu_calibration else 2.0e-3,
        gyro_walk=sequence.imu_calibration.gyroscope_random_walk if sequence.imu_calibration else 1.9393e-5,
        accel_walk=sequence.imu_calibration.accelerometer_random_walk if sequence.imu_calibration else 3.0e-3,
    )
    
    vio_config = VIOConfig(
        max_features=300,
        min_features=50,
        keyframe_interval=10,
        ekf_config=ekf_config,
    )
    
    # Initialize VIO
    print("\nInitializing VIO system...")
    vio = VIOEstimator(cam_calib, vio_config)
    
    if not sequence.ground_truth:
        print("Error: Ground truth required for initialization")
        return 1
    
    initial_pose = sequence.ground_truth[0]
    vio.initialize(initial_pose)
    
    # Process frames
    print("\nProcessing frames...")
    
    processing_times = []
    results = []
    frame_count = 0
    
    max_frames = args.max_frames or float('inf')
    
    for frame, imu_data, gt_pose in loader.get_synchronized_data(sequence):
        if frame_count >= max_frames:
            break
        
        start_time = time.time()
        
        # Process frame
        result = vio.process_frame(
            frame.image,
            frame.timestamp,
            imu_data,
        )
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        results.append(result)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            avg_time = np.mean(processing_times[-100:]) * 1000
            print(f"  Processed {frame_count} frames, avg time: {avg_time:.1f} ms")
    
    print(f"\nProcessed {frame_count} frames")
    print(f"Average processing time: {np.mean(processing_times)*1000:.1f} ms")
    
    # Evaluate results
    print("\nEvaluating trajectory...")
    
    # Get trajectories
    est_positions = np.array([r.state.position for r in results])
    est_timestamps = np.array([r.timestamp for r in results])
    
    gt_positions = np.array([p.position for p in sequence.ground_truth])
    gt_timestamps = np.array([p.timestamp for p in sequence.ground_truth])
    
    # Evaluate
    evaluator = TrajectoryEvaluator()
    evaluator.set_trajectories(
        estimated_positions=est_positions,
        estimated_timestamps=est_timestamps,
        ground_truth_positions=gt_positions,
        ground_truth_timestamps=gt_timestamps,
    )
    
    covariances = [r.covariance for r in results]
    
    eval_result = evaluator.compute_full_evaluation(
        sequence_name=args.sequence,
        covariances=covariances,
        processing_times=processing_times,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAbsolute Trajectory Error (ATE):")
    print(f"  RMSE:   {eval_result.ate.rmse:.4f} m")
    print(f"  Mean:   {eval_result.ate.mean:.4f} m")
    print(f"  Median: {eval_result.ate.median:.4f} m")
    print(f"  Std:    {eval_result.ate.std:.4f} m")
    
    print(f"\nRelative Pose Error (RPE, 1s):")
    print(f"  Translation RMSE: {eval_result.rpe.rmse_translation:.4f} m")
    print(f"  Rotation RMSE:    {eval_result.rpe.rmse_rotation:.2f} deg")
    
    if eval_result.consistency:
        print(f"\nCovariance Consistency:")
        print(f"  Avg NEES: {eval_result.consistency.avg_nees:.2f} (expected: {eval_result.consistency.expected_nees:.1f})")
        print(f"  Consistent: {eval_result.consistency.is_consistent}")
    
    # Save results
    result_file = output_dir / 'evaluation.json'
    eval_result.save(str(result_file))
    print(f"\nResults saved to: {result_file}")
    
    # Visualization
    if args.visualize or args.save_plots:
        print("\nGenerating visualizations...")
        
        vis = Visualizer()
        
        # Aligned trajectories from evaluator
        vis.plot_trajectory_2d(
            estimated=evaluator.aligned_est_positions,
            ground_truth=evaluator.gt_positions,
            title=f"Trajectory - {args.sequence}",
        )
        
        vis.plot_trajectory_3d(
            estimated=evaluator.aligned_est_positions,
            ground_truth=evaluator.gt_positions,
            title=f"3D Trajectory - {args.sequence}",
        )
        
        # Uncertainty visualization
        uncertainties = [r.uncertainty_metrics.position_uncertainty for r in results]
        vis.plot_trajectory_with_uncertainty(
            positions=evaluator.aligned_est_positions,
            uncertainties=uncertainties[:len(evaluator.aligned_est_positions)],
            ground_truth=evaluator.gt_positions,
            title=f"Trajectory with Uncertainty - {args.sequence}",
        )
        
        # Error over time
        errors = np.linalg.norm(evaluator.aligned_est_positions - evaluator.gt_positions, axis=1)
        vis.plot_error_over_time(
            timestamps=est_timestamps[:len(errors)],
            errors=errors,
            title=f"Position Error - {args.sequence}",
        )
        
        # Uncertainty evolution
        vis.plot_uncertainty_evolution(
            timestamps=est_timestamps[:len(results)],
            uncertainty_metrics=[r.uncertainty_metrics for r in results],
            title=f"Uncertainty Evolution - {args.sequence}",
        )
        
        # Evaluation summary
        vis.plot_evaluation_summary(
            result=eval_result,
            title=f"Evaluation Summary - {args.sequence}",
        )
        
        if args.save_plots:
            plots_dir = output_dir / 'plots'
            vis.save_all_plots(str(plots_dir))
            print(f"Plots saved to: {plots_dir}")
        
        if args.visualize:
            print("\nDisplaying plots (close windows to continue)...")
            vis.show()
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
