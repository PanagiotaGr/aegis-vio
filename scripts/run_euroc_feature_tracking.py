from pathlib import Path
import cv2

from src.dataset_loader import EuRoCDatasetLoader
from src.feature_tracker import ORBFeatureTracker

sequence = "datasets/euroc/machine_hall/machine_hall/MH_05_difficult"

loader = EuRoCDatasetLoader(sequence)

img_paths = sorted(
    (Path(sequence) / "mav0" / "cam0" / "data").glob("*.png")
)

img1 = cv2.imread(str(img_paths[0]), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(str(img_paths[1]), cv2.IMREAD_GRAYSCALE)

tracker = ORBFeatureTracker()

result = tracker.match(img1, img2)

print(f"Matches: {len(result.matches)}")
print(f"Inlier ratio: {result.inlier_ratio:.3f}")

out_dir = Path("plots/euroc_feature_tracking")
out_dir.mkdir(parents=True, exist_ok=True)

vis = cv2.drawMatches(
    img1,
    result.keypoints_1,
    img2,
    result.keypoints_2,
    result.matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

cv2.imwrite(
    str(out_dir / "euroc_matches.png"),
    vis,
)

print("Saved:", out_dir / "euroc_matches.png")
