from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("results/failure_prediction/failure_prediction_dataset.csv")

features = [
    "matches",
    "inlier_ratio",
    "quality",
    "visual_uncertainty",
    "imu_instability",
    "multimodal_uncertainty",
]

X = df[features]
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

out_dir = Path("results/failure_prediction_model")
out_dir.mkdir(parents=True, exist_ok=True)

report = classification_report(y_test, pred)
cm = confusion_matrix(y_test, pred)
auc = roc_auc_score(y_test, proba)

with open(out_dir / "random_forest_report.txt", "w") as f:
    f.write("Random Forest Failure Prediction\n\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write(f"\n\nROC-AUC: {auc:.4f}\n")

print(report)
print("Confusion matrix:")
print(cm)
print(f"ROC-AUC: {auc:.4f}")
print("Saved:", out_dir / "random_forest_report.txt")
