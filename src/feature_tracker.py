"""Feature tracking front-end for AegisVIO."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureMatchResult:
    keypoints_1: list
    keypoints_2: list
    matches: list
    inlier_mask: np.ndarray | None

    @property
    def inlier_ratio(self) -> float:
        if self.inlier_mask is None or len(self.matches) == 0:
            return 0.0
        return float(np.mean(self.inlier_mask.ravel() > 0))


class ORBFeatureTracker:
    """ORB detector, descriptor matcher, and RANSAC outlier rejection."""

    def __init__(self, n_features: int = 1500, ratio_test: float = 0.75) -> None:
        self.ratio_test = ratio_test
        self.detector = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, image: np.ndarray):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(image, None)

    def match(self, image_1: np.ndarray, image_2: np.ndarray) -> FeatureMatchResult:
        kp1, des1 = self.detect(image_1)
        kp2, des2 = self.detect(image_2)
        if des1 is None or des2 is None:
            return FeatureMatchResult(kp1, kp2, [], None)

        raw = self.matcher.knnMatch(des1, des2, k=2)
        matches = []
        for pair in raw:
            if len(pair) == 2 and pair[0].distance < self.ratio_test * pair[1].distance:
                matches.append(pair[0])

        if len(matches) < 8:
            return FeatureMatchResult(kp1, kp2, matches, None)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
        return FeatureMatchResult(kp1, kp2, matches, mask)
