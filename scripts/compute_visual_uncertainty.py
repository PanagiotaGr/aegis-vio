from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/feature_quality/feature_quality.csv")

q = df["quality"]

q_min = q.min()
q_max = q.max()

df["quality_norm"] = (q - q_min) / (q_max - q_min)

epsilon = 1e-6
df["visual_uncertainty"] = 1.0 / (df["quality_norm"] + epsilon)

# clip για να μην εκτοξεύεται υπερβολικά
df["visual_uncertainty"] = df["visual_uncertainty"].clip(upper=20.0)

out_dir = Path("results/visual_uncertainty")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "visual_uncertainty.csv"
df.to_csv(csv_path, index=False)

plot_dir = Path("plots/visual_uncertainty")
plot_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 4))
plt.plot(df["frame"], df["visual_uncertainty"])
plt.xlabel("Frame pair")
plt.ylabel("Visual uncertainty")
plt.title("EuRoC MH_01 Visual Uncertainty Over Time")
plt.grid(True)
plt.savefig(plot_dir / "visual_uncertainty_over_time.png", dpi=200, bbox_inches="tight")
plt.close()

print(df[["frame", "quality", "quality_norm", "visual_uncertainty"]].describe())
print("Saved:", csv_path)
print("Saved plot:", plot_dir / "visual_uncertainty_over_time.png")
