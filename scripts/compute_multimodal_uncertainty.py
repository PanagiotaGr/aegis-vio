from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset_loader import EuRoCDatasetLoader

sequence = "datasets/euroc/machine_hall/MH_01_easy"

visual_df = pd.read_csv("results/visual_uncertainty/visual_uncertainty.csv")

loader = EuRoCDatasetLoader(sequence)
imu = loader.load_imu()

# Υπολογίζουμε IMU magnitude
imu["gyro_norm"] = np.sqrt(imu["w_x"]**2 + imu["w_y"]**2 + imu["w_z"]**2)
imu["accel_norm"] = np.sqrt(imu["a_x"]**2 + imu["a_y"]**2 + imu["a_z"]**2)

# Κάνουμε downsample το IMU ώστε να ταιριάξει περίπου με τα frame pairs
n = len(visual_df)
imu_chunks = np.array_split(imu, n)

imu_instability = []
for chunk in imu_chunks:
    gyro_std = chunk["gyro_norm"].std()
    accel_std = chunk["accel_norm"].std()
    imu_instability.append(gyro_std + 0.1 * accel_std)

visual_df["imu_instability_raw"] = imu_instability

# Normalize IMU instability στο [0, 1]
x = visual_df["imu_instability_raw"]
visual_df["imu_instability"] = (x - x.min()) / (x.max() - x.min() + 1e-9)

# Normalize visual uncertainty στο [0, 1]
v = visual_df["visual_uncertainty"]
visual_df["visual_uncertainty_norm"] = (v - v.min()) / (v.max() - v.min() + 1e-9)

# Multi-modal uncertainty
alpha = 0.7
beta = 0.3

visual_df["multimodal_uncertainty"] = (
    alpha * visual_df["visual_uncertainty_norm"]
    + beta * visual_df["imu_instability"]
)

out_dir = Path("results/multimodal_uncertainty")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "multimodal_uncertainty.csv"
visual_df.to_csv(csv_path, index=False)

plot_dir = Path("plots/multimodal_uncertainty")
plot_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 4))
plt.plot(visual_df["frame"], visual_df["visual_uncertainty_norm"], label="Visual uncertainty")
plt.plot(visual_df["frame"], visual_df["imu_instability"], label="IMU instability")
plt.plot(visual_df["frame"], visual_df["multimodal_uncertainty"], label="Multi-modal uncertainty")
plt.xlabel("Frame pair")
plt.ylabel("Normalized score")
plt.title("Multi-Modal Uncertainty Over Time")
plt.legend()
plt.grid(True)
plt.savefig(plot_dir / "multimodal_uncertainty_over_time.png", dpi=200, bbox_inches="tight")
plt.close()

print(visual_df[[
    "visual_uncertainty_norm",
    "imu_instability",
    "multimodal_uncertainty",
]].describe())

print("Saved:", csv_path)
print("Saved plot:", plot_dir / "multimodal_uncertainty_over_time.png")
