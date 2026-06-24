"""
Visualization Module

Provides plotting functions for:
- Trajectory visualization
- Covariance ellipses
- Uncertainty heatmaps
- Error analysis
- Real-time visualization support
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple, Dict
from pathlib import Path

from .evaluator import EvaluationResult, ATEResult, RPEResult
from .uncertainty import UncertaintyMetrics


class Visualizer:
    """
    Visualization tools for VIO results.
    
    Example:
        vis = Visualizer()
        
        # Plot trajectory comparison
        vis.plot_trajectory_comparison(
            estimated=positions_est,
            ground_truth=positions_gt,
        )
        
        # Plot uncertainty over time
        vis.plot_uncertainty_evolution(uncertainty_metrics)
        
        # Save all plots
        vis.save_all_plots("results/plots/")
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 150):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI for saving
        """
        self.figsize = figsize
        self.dpi = dpi
        self.figures: Dict[str, plt.Figure] = {}
        
        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Color schemes
        self.colors = {
            'estimated': '#1f77b4',  # Blue
            'ground_truth': '#2ca02c',  # Green
            'error': '#d62728',  # Red
            'uncertainty': '#ff7f0e',  # Orange
            'keyframe': '#9467bd',  # Purple
        }
    
    def plot_trajectory_2d(
        self,
        estimated: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        title: str = "Trajectory (Top View)",
        show_start_end: bool = True,
    ) -> plt.Figure:
        """
        Plot 2D trajectory (top-down view).
        
        Args:
            estimated: Nx3 estimated positions
            ground_truth: Nx3 ground truth positions (optional)
            title: Plot title
            show_start_end: Show start and end markers
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot ground truth
        if ground_truth is not None:
            ax.plot(
                ground_truth[:, 0], ground_truth[:, 1],
                color=self.colors['ground_truth'],
                linewidth=2, label='Ground Truth', alpha=0.8
            )
        
        # Plot estimated
        ax.plot(
            estimated[:, 0], estimated[:, 1],
            color=self.colors['estimated'],
            linewidth=2, label='Estimated', alpha=0.8
        )
        
        # Start and end markers
        if show_start_end:
            ax.scatter(
                estimated[0, 0], estimated[0, 1],
                marker='o', s=100, c='green', zorder=5, label='Start'
            )
            ax.scatter(
                estimated[-1, 0], estimated[-1, 1],
                marker='s', s=100, c='red', zorder=5, label='End'
            )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        
        self.figures['trajectory_2d'] = fig
        return fig
    
    def plot_trajectory_3d(
        self,
        estimated: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        title: str = "3D Trajectory",
        view_angles: Tuple[float, float] = (30, -60),
    ) -> plt.Figure:
        """
        Plot 3D trajectory.
        
        Args:
            estimated: Nx3 estimated positions
            ground_truth: Nx3 ground truth positions (optional)
            title: Plot title
            view_angles: (elevation, azimuth) view angles
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot ground truth
        if ground_truth is not None:
            ax.plot(
                ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                color=self.colors['ground_truth'],
                linewidth=2, label='Ground Truth', alpha=0.8
            )
        
        # Plot estimated
        ax.plot(
            estimated[:, 0], estimated[:, 1], estimated[:, 2],
            color=self.colors['estimated'],
            linewidth=2, label='Estimated', alpha=0.8
        )
        
        # Start and end markers
        ax.scatter(
            *estimated[0], marker='o', s=100, c='green', label='Start'
        )
        ax.scatter(
            *estimated[-1], marker='s', s=100, c='red', label='End'
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        
        self.figures['trajectory_3d'] = fig
        return fig
    
    def plot_trajectory_with_uncertainty(
        self,
        positions: np.ndarray,
        uncertainties: List[float],
        ground_truth: Optional[np.ndarray] = None,
        title: str = "Trajectory with Uncertainty",
    ) -> plt.Figure:
        """
        Plot trajectory colored by uncertainty level.
        
        Args:
            positions: Nx3 estimated positions
            uncertainties: N uncertainty values
            ground_truth: Nx3 ground truth positions (optional)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot ground truth
        if ground_truth is not None:
            ax.plot(
                ground_truth[:, 0], ground_truth[:, 1],
                color=self.colors['ground_truth'],
                linewidth=1, label='Ground Truth', alpha=0.5
            )
        
        # Create color map based on uncertainty
        uncertainties = np.array(uncertainties)
        norm = plt.Normalize(uncertainties.min(), uncertainties.max())
        cmap = plt.cm.viridis_r  # High uncertainty = yellow/light
        
        # Plot segments with colors
        for i in range(len(positions) - 1):
            color = cmap(norm(uncertainties[i]))
            ax.plot(
                [positions[i, 0], positions[i+1, 0]],
                [positions[i, 1], positions[i+1, 1]],
                color=color, linewidth=2
            )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Position Uncertainty (m)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.axis('equal')
        ax.grid(True)
        
        self.figures['trajectory_uncertainty'] = fig
        return fig
    
    def plot_covariance_ellipses(
        self,
        positions: np.ndarray,
        covariances: List[np.ndarray],
        ground_truth: Optional[np.ndarray] = None,
        n_ellipses: int = 20,
        confidence: float = 0.95,
        title: str = "Trajectory with Covariance Ellipses",
    ) -> plt.Figure:
        """
        Plot trajectory with covariance ellipses.
        
        Args:
            positions: Nx3 estimated positions
            covariances: N 3x3 position covariance matrices
            ground_truth: Nx3 ground truth positions (optional)
            n_ellipses: Number of ellipses to plot
            confidence: Confidence level for ellipses
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from scipy.stats import chi2
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Chi-squared value for 2 DOF (2D ellipse)
        chi2_val = chi2.ppf(confidence, df=2)
        
        # Plot ground truth
        if ground_truth is not None:
            ax.plot(
                ground_truth[:, 0], ground_truth[:, 1],
                color=self.colors['ground_truth'],
                linewidth=2, label='Ground Truth', alpha=0.8
            )
        
        # Plot trajectory
        ax.plot(
            positions[:, 0], positions[:, 1],
            color=self.colors['estimated'],
            linewidth=2, label='Estimated', alpha=0.8
        )
        
        # Sample indices for ellipses
        n = len(positions)
        indices = np.linspace(0, n - 1, n_ellipses, dtype=int)
        
        # Plot ellipses
        for idx in indices:
            pos = positions[idx]
            cov = covariances[idx][:2, :2]  # 2D covariance
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Ellipse parameters
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width = 2 * np.sqrt(eigenvalues[0] * chi2_val)
            height = 2 * np.sqrt(eigenvalues[1] * chi2_val)
            
            ellipse = Ellipse(
                xy=(pos[0], pos[1]),
                width=width,
                height=height,
                angle=angle,
                facecolor='none',
                edgecolor=self.colors['uncertainty'],
                linewidth=1,
                alpha=0.6,
            )
            ax.add_patch(ellipse)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        
        self.figures['covariance_ellipses'] = fig
        return fig
    
    def plot_error_over_time(
        self,
        timestamps: np.ndarray,
        errors: np.ndarray,
        title: str = "Position Error Over Time",
        ylabel: str = "Error (m)",
    ) -> plt.Figure:
        """
        Plot error evolution over time.
        
        Args:
            timestamps: N timestamps
            errors: N error values
            title: Plot title
            ylabel: Y-axis label
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert timestamps to seconds from start
        t = timestamps - timestamps[0]
        if t[0] > 1e9:  # nanoseconds
            t = t * 1e-9
        
        ax.plot(t, errors, color=self.colors['error'], linewidth=1.5)
        ax.fill_between(t, 0, errors, color=self.colors['error'], alpha=0.3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        
        # Add statistics
        stats_text = f"Mean: {np.mean(errors):.3f}\nStd: {np.std(errors):.3f}\nMax: {np.max(errors):.3f}"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.8),
        )
        
        self.figures['error_time'] = fig
        return fig
    
    def plot_uncertainty_evolution(
        self,
        timestamps: np.ndarray,
        uncertainty_metrics: List[UncertaintyMetrics],
        title: str = "Uncertainty Evolution",
    ) -> plt.Figure:
        """
        Plot uncertainty metrics over time.
        
        Args:
            timestamps: N timestamps
            uncertainty_metrics: N UncertaintyMetrics objects
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), sharex=True)
        
        # Convert timestamps
        t = timestamps - timestamps[0]
        if t[0] > 1e9:
            t = t * 1e-9
        
        # Position uncertainty
        pos_unc = [m.position_uncertainty for m in uncertainty_metrics]
        axes[0].plot(t, pos_unc, color=self.colors['uncertainty'], linewidth=1.5)
        axes[0].fill_between(t, 0, pos_unc, color=self.colors['uncertainty'], alpha=0.3)
        axes[0].set_ylabel('Position (m)')
        axes[0].set_title('Position Uncertainty')
        axes[0].grid(True)
        
        # Rotation uncertainty
        rot_unc = [np.degrees(m.rotation_uncertainty) for m in uncertainty_metrics]
        axes[1].plot(t, rot_unc, color=self.colors['uncertainty'], linewidth=1.5)
        axes[1].fill_between(t, 0, rot_unc, color=self.colors['uncertainty'], alpha=0.3)
        axes[1].set_ylabel('Rotation (deg)')
        axes[1].set_title('Rotation Uncertainty')
        axes[1].grid(True)
        
        # Tracking quality
        quality = [m.tracking_quality for m in uncertainty_metrics]
        axes[2].plot(t, quality, color='green', linewidth=1.5)
        axes[2].fill_between(t, 0, quality, color='green', alpha=0.3)
        axes[2].set_ylabel('Quality')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Tracking Quality')
        axes[2].set_ylim([0, 1.1])
        axes[2].grid(True)
        
        fig.suptitle(title)
        fig.tight_layout()
        
        self.figures['uncertainty_evolution'] = fig
        return fig
    
    def plot_xyz_errors(
        self,
        timestamps: np.ndarray,
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        title: str = "Position Error per Axis",
    ) -> plt.Figure:
        """
        Plot X, Y, Z errors separately.
        
        Args:
            timestamps: N timestamps
            estimated: Nx3 estimated positions
            ground_truth: Nx3 ground truth positions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2), sharex=True)
        
        errors = estimated - ground_truth
        
        t = timestamps - timestamps[0]
        if t[0] > 1e9:
            t = t * 1e-9
        
        labels = ['X Error', 'Y Error', 'Z Error']
        
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(t, errors[:, i], linewidth=1)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_ylabel(f'{label} (m)')
            ax.grid(True)
            
            # Add RMSE
            rmse = np.sqrt(np.mean(errors[:, i] ** 2))
            ax.text(0.02, 0.98, f'RMSE: {rmse:.3f} m',
                   transform=ax.transAxes, verticalalignment='top')
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title)
        fig.tight_layout()
        
        self.figures['xyz_errors'] = fig
        return fig
    
    def plot_nees_nis(
        self,
        timestamps: np.ndarray,
        nees_values: List[float],
        nis_values: Optional[List[float]] = None,
        expected_nees: float = 3.0,
        title: str = "Estimation Consistency",
    ) -> plt.Figure:
        """
        Plot NEES and NIS for consistency analysis.
        
        Args:
            timestamps: N timestamps
            nees_values: N NEES values
            nis_values: N NIS values (optional)
            expected_nees: Expected NEES value (degrees of freedom)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from scipy.stats import chi2
        
        n_plots = 2 if nis_values else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], self.figsize[1] * 0.6 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        t = timestamps[:len(nees_values)] - timestamps[0]
        if t[0] > 1e9:
            t = t * 1e-9
        
        # NEES plot
        axes[0].plot(t, nees_values, linewidth=1, label='NEES')
        axes[0].axhline(y=expected_nees, color='green', linestyle='--', label=f'Expected ({expected_nees})')
        
        # Confidence bounds
        n = len(nees_values)
        lower = chi2.ppf(0.025, df=int(expected_nees) * n) / n
        upper = chi2.ppf(0.975, df=int(expected_nees) * n) / n
        axes[0].axhline(y=lower, color='red', linestyle=':', label='95% bounds')
        axes[0].axhline(y=upper, color='red', linestyle=':')
        
        axes[0].set_ylabel('NEES')
        axes[0].set_title('Normalized Estimation Error Squared')
        axes[0].legend()
        axes[0].grid(True)
        
        # NIS plot
        if nis_values:
            t_nis = timestamps[:len(nis_values)] - timestamps[0]
            if t_nis[0] > 1e9:
                t_nis = t_nis * 1e-9
            
            axes[1].plot(t_nis, nis_values, linewidth=1, label='NIS')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('NIS')
            axes[1].set_title('Normalized Innovation Squared')
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[0].set_xlabel('Time (s)')
        
        fig.suptitle(title)
        fig.tight_layout()
        
        self.figures['nees_nis'] = fig
        return fig
    
    def plot_evaluation_summary(
        self,
        result: EvaluationResult,
        title: str = "Evaluation Summary",
    ) -> plt.Figure:
        """
        Create summary visualization of evaluation results.
        
        Args:
            result: EvaluationResult object
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        
        # ATE histogram
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(result.ate.errors, bins=30, color=self.colors['error'], alpha=0.7, edgecolor='black')
        ax1.axvline(result.ate.mean, color='red', linestyle='--', label=f'Mean: {result.ate.mean:.3f}')
        ax1.axvline(result.ate.rmse, color='blue', linestyle='--', label=f'RMSE: {result.ate.rmse:.3f}')
        ax1.set_xlabel('ATE (m)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Absolute Trajectory Error Distribution')
        ax1.legend()
        ax1.grid(True)
        
        # RPE translation histogram
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(result.rpe.translation_errors, bins=30, color=self.colors['uncertainty'], alpha=0.7, edgecolor='black')
        ax2.axvline(result.rpe.mean_translation, color='red', linestyle='--', 
                   label=f'Mean: {result.rpe.mean_translation:.3f}')
        ax2.set_xlabel('RPE Translation (m)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Relative Pose Error Distribution')
        ax2.legend()
        ax2.grid(True)
        
        # Per-axis RMSE bar chart
        ax3 = fig.add_subplot(2, 2, 3)
        axes = ['X', 'Y', 'Z']
        rmse_values = [result.ate.rmse_x, result.ate.rmse_y, result.ate.rmse_z]
        bars = ax3.bar(axes, rmse_values, color=[self.colors['error'], self.colors['uncertainty'], self.colors['estimated']])
        ax3.set_ylabel('RMSE (m)')
        ax3.set_title('Per-Axis Position RMSE')
        ax3.grid(True, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars, rmse_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}',
                    ha='center', va='bottom')
        
        # Summary text
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        summary_text = f"""
        Evaluation Summary
        ==================
        
        Sequence: {result.sequence_name}
        Duration: {result.duration:.1f} s
        Length: {result.sequence_length:.1f} m
        Poses: {result.num_poses}
        
        ATE
        ---
        RMSE: {result.ate.rmse:.4f} m
        Mean: {result.ate.mean:.4f} m
        Median: {result.ate.median:.4f} m
        Std: {result.ate.std:.4f} m
        
        RPE (1s)
        --------
        Translation RMSE: {result.rpe.rmse_translation:.4f} m
        Rotation RMSE: {result.rpe.rmse_rotation:.2f} deg
        
        Processing
        ----------
        Avg Time: {result.avg_processing_time*1000:.1f} ms
        """
        
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        self.figures['evaluation_summary'] = fig
        return fig
    
    def save_all_plots(self, output_dir: str):
        """
        Save all generated plots to directory.
        
        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = output_path / f"{name}.png"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
    
    def show(self):
        """Display all figures."""
        plt.show()
    
    def close_all(self):
        """Close all figures."""
        plt.close('all')
        self.figures.clear()
