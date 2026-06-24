import numpy as np

from src.simulation.synthetic_vio import compute_position_error, run_synthetic_vio


def test_synthetic_vio_shapes():
    result = run_synthetic_vio(num_steps=100)
    assert result.timestamps.shape == (100,)
    assert result.ground_truth.shape == (100, 2)
    assert result.estimate.shape == (100, 2)
    assert result.covariance_trace.shape == (100,)
    assert result.uncertainty_score.shape == (100,)


def test_synthetic_vio_error_is_finite():
    result = run_synthetic_vio(num_steps=100)
    error = compute_position_error(result)
    assert error.shape == (100,)
    assert np.all(np.isfinite(error))


def test_uncertainty_increases_during_degradation():
    result = run_synthetic_vio(num_steps=600)
    normal = result.uncertainty_score[result.visual_quality > 0.9]
    degraded = result.uncertainty_score[result.visual_quality < 0.2]
    assert degraded.mean() > normal.mean()
