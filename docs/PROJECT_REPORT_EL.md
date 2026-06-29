# Αναλυτική Τεκμηρίωση Project

**Project:** AEGIS-VIO

## Σκοπός

Το AEGIS-VIO είναι ερευνητικό project για Visual-Inertial Odometry με εκτίμηση αβεβαιότητας. Ο στόχος είναι ένα ρομποτικό σύστημα να μην παράγει μόνο εκτίμηση θέσης, αλλά και ένδειξη αξιοπιστίας της εκτίμησης.

## Κεντρική ιδέα

Το σύστημα υπολογίζει pose estimate, covariance και risk score. Το risk score περνάει στο navigation layer, ώστε το robot να προσαρμόζει τη συμπεριφορά του όταν η αβεβαιότητα μεγαλώνει.

## Αρχιτεκτονική

| Layer | Ρόλος | Αρχεία |
| --- | --- | --- |
| State estimation | Error-State EKF και IMU propagation | `src/ekf.py`, `src/imu_integrator.py` |
| Visual front-end | KLT feature tracking και visual quality score | `src/feature_tracker.py`, `src/vio_estimator.py` |
| Uncertainty | Covariance, entropy, NEES, NIS, risk | `src/uncertainty.py` |
| Navigation | Risk-aware state machine | `src/navigation.py` |
| Evaluation | ATE, RPE και consistency metrics | `src/evaluator.py` |
| ROS2 | Nodes, config, launch, messages | `ros2_ws/src/aegis_vio/` |

## Error-State EKF

Ο estimator χρησιμοποιεί 15-dimensional error-state EKF. Το state περιλαμβάνει θέση, ταχύτητα, attitude και IMU biases. Η IMU χρησιμοποιείται για πρόβλεψη, ενώ οι οπτικές μετρήσεις διορθώνουν την κατάσταση.

## Visual front-end

Το visual front-end βασίζεται σε pyramidal KLT optical flow. Εκτός από feature tracks, παράγει quality score με βάση πλήθος features, ηλικία tracks και χωρική διασπορά. Όταν η οπτική ποιότητα μειώνεται, αυξάνεται το measurement noise.

## Uncertainty quantification

Το project υπολογίζει covariance trace, determinant, differential entropy, Mahalanobis distance, confidence ellipsoids, NEES, NIS και συνολικό risk score. Έτσι αξιολογείται όχι μόνο η ακρίβεια αλλά και η αξιοπιστία της εκτίμησης.

## Risk-aware navigation

Η πλοήγηση χρησιμοποιεί καταστάσεις NORMAL, CAUTIOUS, RECOVERY και HALT. Όσο αυξάνεται το risk, μειώνεται η ταχύτητα και αυξάνονται τα safety margins.

## Evaluation

Η αξιολόγηση περιλαμβάνει Absolute Trajectory Error, Relative Pose Error, Umeyama Sim(3) alignment, tracking failure rate και NEES/NIS consistency bounds.

## Πειράματα

Το repository περιγράφει stress tests για motion blur, low light, dynamic objects, fast motion, feature-poor scenes, sensor noise και IMU drift. Στόχος είναι να φανεί πότε το VIO γίνεται λιγότερο αξιόπιστο και πώς αντιδρά το risk-aware navigation.

## Εκτέλεση

```bash
git clone https://github.com/PanagiotaGr/aegis-vio.git
cd aegis-vio
pip install -r requirements.txt
pip install -e .
python scripts/run_simulation.py --scenario mixed --output_dir results/simulation
pytest tests/ -v
```

## Συμπέρασμα

Το AEGIS-VIO δείχνει πώς η αβεβαιότητα μπορεί να μετατραπεί σε πρακτική απόφαση πλοήγησης. Η βασική συνεισφορά είναι η σύνδεση ανάμεσα σε probabilistic VIO estimation και risk-aware robotic behavior.
