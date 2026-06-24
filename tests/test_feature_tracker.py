import numpy as np

from src.feature_tracker import ORBFeatureTracker


def test_feature_tracker_handles_blank_images():
    tracker = ORBFeatureTracker()
    image = np.zeros((100, 100), dtype=np.uint8)
    result = tracker.match(image, image)
    assert result.inlier_ratio == 0.0
