from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/feature_quality/feature_quality.csv")

out_dir = Path("plots/feature_quality")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 4))
plt.plot(df["frame"], df["quality"])
plt.xlabel("Frame pair")
plt.ylabel("Feature quality score")
plt.title("EuRoC MH_01 Feature Quality Over Time")
plt.grid(True)
plt.savefig(out_dir / "feature_quality_over_time.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(df["frame"], df["inlier_ratio"])
plt.xlabel("Frame pair")
plt.ylabel("Inlier ratio")
plt.title("EuRoC MH_01 Inlier Ratio Over Time")
plt.grid(True)
plt.savefig(out_dir / "inlier_ratio_over_time.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved plots to:", out_dir)
