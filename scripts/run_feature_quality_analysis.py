from pathlib import Path
import cv2
import pandas as pd

from src.feature_tracker import ORBFeatureTracker

sequence = "datasets/euroc/machine_hall/machine_hall/MH_05_difficult"

image_paths = sorted(
    (Path(sequence) / "mav0" / "cam0" / "data").glob("*.png")
)

tracker = ORBFeatureTracker()

results = []

max_pairs = 300

for i in range(min(max_pairs, len(image_paths) - 1)):

    img1 = cv2.imread(str(image_paths[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image_paths[i + 1]), cv2.IMREAD_GRAYSCALE)

    result = tracker.match(img1, img2)

    matches = len(result.matches)
    inlier_ratio = result.inlier_ratio

    quality = matches * inlier_ratio

    results.append(
        {
            "frame": i,
            "matches": matches,
            "inlier_ratio": inlier_ratio,
            "quality": quality,
        }
    )

df = pd.DataFrame(results)

out_dir = Path("results/feature_quality")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "feature_quality.csv"

df.to_csv(csv_path, index=False)

print(df.describe())

print()
print("Saved:", csv_path)
