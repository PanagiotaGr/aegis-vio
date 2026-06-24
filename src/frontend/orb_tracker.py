"""ORB feature detection and matching front-end.

This is the first visual front-end milestone for AegisVIO. It does not yet
perform full VIO; it gives us reliable feature detection, matching, and
RANSAC-based outlier rejection for EuRoC image sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    keypoints_1: list[cv2.KeyPoint]
    keypoints_2: list[cv2.KeyPoint]
    matches: list[cv2.DMatch]
    inlier_mask: np.ndarray | None
    essential_matrix: np.ndarray | None

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    @property
    def num_inliers(self) -> int:
        if self.inlier_mask is None:
            return 0
        return int(np.sum(self.inlier_mask.ravel() > 0))


class ORBTracker:
    """ORB detector + brute-force matcher + geometric filtering."""

    def __init__(
        self,
        n_features: int = 1500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        ratio_test: float = 0.75,
    ) -> None:
        self.ratio_test = ratio_test
        self.detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match(self, image_1: np.ndarray, image_2: np.ndarray, camera_matrix: np.ndarray | None = None) -> MatchResult:
        kp1, des1 = self.detect_and_compute(image_1)
        kp2, des2 = self.detect_and_compute(image_2)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return MatchResult(kp1, kp2, [], None, None)

        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches: list[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            return MatchResult(kp1, kp2, good_matches, None, None)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        if camera_matrix is None:
            essential, mask = cv2.findFundamentalMat(
                pts1,
                pts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=1.0,
                confidence=0.999,
            )
        else:
            essential, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )

        return MatchResult(kp1, kp2, good_matches, mask, essential)

    @staticmethod
    def draw_matches(image_1: np.ndarray, image_2: np.ndarray, result: MatchResult, inliers_only: bool = True) -> np.ndarray:
        matches = result.matches
        if inliers_only and result.inlier_mask is not None:
            mask = result.inlier_mask.ravel() > 0
            matches = [m for m, keep in zip(result.matches, mask) if keep]

        return cv2.drawMatches(
            image_1,
            result.keypoints_1,
            image_2,
            result.keypoints_2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
