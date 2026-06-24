import numpy as np

from src.ekf import ExtendedKalmanFilter


def test_ekf_predict_update_shapes():
    ekf = ExtendedKalmanFilter(state_dim=2)
    ekf.predict(np.eye(2), np.eye(2) * 0.01)
    ekf.update(np.array([1.0]), np.array([[1.0, 0.0]]), np.array([[0.1]]))
    assert ekf.x.shape == (2,)
    assert ekf.P.shape == (2, 2)
