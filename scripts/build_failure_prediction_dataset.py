from pathlib import Path

import numpy as np
import pandas as pd

# multimodal uncertainty
df = pd.read_csv(
    "results/multimodal_uncertainty/multimodal_uncertainty.csv"
)

# --------------------------------------------------
# προσωρινό proxy target
# αργότερα θα αντικατασταθεί με πραγματικό pose error
# --------------------------------------------------

future_error = []

window = 10

for i in range(len(df)):

    start = i
    end = min(i + window, len(df))

    future_u = df["multimodal_uncertainty"].iloc[start:end]

    error = future_u.mean()

    future_error.append(error)

df["future_localization_error"] = future_error

# binary failure label

threshold = df["future_localization_error"].quantile(0.80)

df["failure"] = (
    df["future_localization_error"] > threshold
).astype(int)

out_dir = Path("results/failure_prediction")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "failure_prediction_dataset.csv"

df.to_csv(csv_path, index=False)

print()
print(df["failure"].value_counts())
print()
print(df.head())
print()
print("saved:", csv_path)
