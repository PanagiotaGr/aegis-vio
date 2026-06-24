from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/visual_uncertainty/visual_uncertainty.csv")

def mode_from_uncertainty(u):
    if u < 2.0:
        return "nominal"
    elif u < 6.0:
        return "cautious"
    else:
        return "recovery"

df["navigation_mode"] = df["visual_uncertainty"].apply(mode_from_uncertainty)

out_dir = Path("results/navigation_modes")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "navigation_modes.csv"
df.to_csv(csv_path, index=False)

print(df["navigation_mode"].value_counts())
print("Saved:", csv_path)

plot_dir = Path("plots/navigation_modes")
plot_dir.mkdir(parents=True, exist_ok=True)

mode_map = {
    "nominal": 0,
    "cautious": 1,
    "recovery": 2,
}

df["mode_numeric"] = df["navigation_mode"].map(mode_map)

plt.figure(figsize=(10, 4))
plt.plot(df["frame"], df["mode_numeric"])
plt.yticks([0, 1, 2], ["nominal", "cautious", "recovery"])
plt.xlabel("Frame pair")
plt.ylabel("Navigation mode")
plt.title("Uncertainty-Aware Navigation Mode Over Time")
plt.grid(True)
plt.savefig(plot_dir / "navigation_modes_over_time.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved plot:", plot_dir / "navigation_modes_over_time.png")
