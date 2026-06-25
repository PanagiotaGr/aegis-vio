# AEGIS-VIO — Εκτέλεση και αποτελέσματα

## Τι έγινε

Το project εγκαταστάθηκε και εκτελέστηκε σε Ubuntu/WSL με Python virtual environment.

## Έλεγχος εγκατάστασης

Εκτελέστηκαν τα unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests
Αποτέλεσμα:

12 passed
Synthetic demos

Εκτελέστηκαν:

python scripts/run_uncertainty_demo.py
python scripts/run_synthetic_feature_tracking.py

Παρήχθησαν αποτελέσματα στα:

results/synthetic_uncertainty_demo/
plots/synthetic_uncertainty_demo/
plots/synthetic_feature_tracking/
EuRoC Machine Hall sequence

Χρησιμοποιήθηκε το EuRoC sequence:

datasets/euroc/machine_hall/machine_hall/MH_05_difficult

Το dataset δεν ανεβαίνει στο GitHub.

Επιβεβαίωση φόρτωσης:

cam0 frames: 2273
cam1 frames: 2273
IMU packets: 22721
ground truth: True
Feature tracking

Εκτελέστηκε:

python scripts/run_euroc_feature_tracking.py

Αποτέλεσμα:

Matches: 1319
Inlier ratio: 0.987

Plot:

plots/euroc_feature_tracking/euroc_matches.png
Feature quality και uncertainty pipeline

Εκτελέστηκαν:

python scripts/run_feature_quality_analysis.py
python scripts/compute_visual_uncertainty.py
python scripts/compute_navigation_modes.py
python scripts/compute_multimodal_uncertainty.py
python scripts/build_failure_prediction_dataset.py
python scripts/train_failure_predictor.py

Παρήχθησαν:

results/feature_quality/feature_quality.csv
results/visual_uncertainty/visual_uncertainty.csv
results/navigation_modes/navigation_modes.csv
results/multimodal_uncertainty/multimodal_uncertainty.csv
results/failure_prediction/failure_prediction_dataset.csv
results/failure_prediction_model/random_forest_report.txt

και plots:

plots/feature_quality/
plots/visual_uncertainty/
plots/navigation_modes/
plots/multimodal_uncertainty/
Συμπέρασμα

Το πλήρες pipeline του AEGIS-VIO εκτελέστηκε επιτυχώς: synthetic demos, EuRoC feature tracking, visual uncertainty, navigation mode selection, multimodal uncertainty και failure prediction model.
