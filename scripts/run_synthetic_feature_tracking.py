import cv2
import numpy as np
from pathlib import Path

from src.feature_tracker import ORBFeatureTracker

out = Path("plots/synthetic_feature_tracking")
out.mkdir(parents=True, exist_ok=True)

img1 = np.zeros((400, 400), dtype=np.uint8)
img2 = np.zeros((400, 400), dtype=np.uint8)

cv2.circle(img1, (120, 120), 30, 255, -1)
cv2.rectangle(img1, (220, 220), (280, 280), 255, -1)

cv2.circle(img2, (130, 125), 30, 255, -1)
cv2.rectangle(img2, (230, 225), (290, 285), 255, -1)

tracker = ORBFeatureTracker()
result = tracker.match(img1, img2)

vis = cv2.drawMatches(
    img1, result.keypoints_1,
    img2, result.keypoints_2,
    result.matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imwrite(str(out / "synthetic_matches.png"), vis)

print("Synthetic feature tracking completed.")
print("matches:", len(result.matches))
print("inlier ratio:", result.inlier_ratio)
print("plot:", out / "synthetic_matches.png")
