import numpy as np

from src.uncertainty import covariance_trace, mahalanobis_distance_squared, uncertainty_score


def test_covariance_trace():
    assert covariance_trace(np.eye(3)) == 3.0


def test_mahalanobis_distance_squared():
    assert mahalanobis_distance_squared(np.array([1.0, 2.0]), np.eye(2)) == 5.0


def test_uncertainty_score_decreases_with_quality():
    cov = np.eye(2) * 0.1
    assert uncertainty_score(cov, quality=0.5) > uncertainty_score(cov, quality=1.0)
