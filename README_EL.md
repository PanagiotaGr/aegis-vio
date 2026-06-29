# AEGIS-VIO

## Ελληνική αναλυτική τεκμηρίωση

Το AEGIS-VIO είναι ένα ερευνητικό prototype για uncertainty-aware Visual-Inertial Odometry και risk-aware robotic navigation. Ο στόχος του project είναι ένα ρομποτικό σύστημα να μην εκτιμά μόνο τη θέση του, αλλά να γνωρίζει και πόσο αξιόπιστη είναι αυτή η εκτίμηση.

Σε πολλά VIO και SLAM συστήματα, το σύστημα συνεχίζει να παράγει pose estimates ακόμη και όταν οι συνθήκες γίνονται δύσκολες. Παραδείγματα τέτοιων συνθηκών είναι χαμηλός φωτισμός, motion blur, feature-poor διάδρομοι, δυναμικά αντικείμενα και απότομες κινήσεις. Το AEGIS-VIO προσπαθεί να κλείσει αυτό το κενό συνδέοντας την αβεβαιότητα με τη συμπεριφορά του robot.

## Κεντρική ιδέα

Το project υπολογίζει μία πιθανοτική εκτίμηση κατάστασης και από αυτήν εξάγει risk score. Το risk score χρησιμοποιείται από το navigation layer για να αποφασίσει αν το robot πρέπει να κινηθεί κανονικά, να γίνει πιο προσεκτικό, να προσπαθήσει recovery ή να σταματήσει.

Η βασική αρχή είναι:

Όταν η αβεβαιότητα αυξάνεται, η συμπεριφορά του robot πρέπει να γίνεται πιο συντηρητική.

## Αρχιτεκτονική συστήματος

| Layer | Τι κάνει | Βασικά αρχεία |
| --- | --- | --- |
| State estimation | Error-State EKF με θέση, ταχύτητα, attitude και IMU biases | `src/ekf.py`, `src/imu_integrator.py` |
| Visual front-end | KLT optical-flow tracking και visual quality score | `src/feature_tracker.py`, `src/vio_estimator.py` |
| Uncertainty quantification | covariance metrics, entropy, NEES, NIS, risk score | `src/uncertainty.py` |
| Risk-aware navigation | state machine και προσαρμογή ταχύτητας/safety margins | `src/navigation.py` |
| Control loop | σύνδεση estimator, uncertainty και navigation | `src/uncertainty_aware_controller.py` |
| Evaluation | ATE, RPE, consistency metrics | `src/evaluator.py` |
| ROS2 package | nodes, messages, launch files και YAML config | `ros2_ws/src/aegis_vio/` |

## Error-State EKF

Ο πυρήνας του estimator είναι ένας 15-dimensional error-state EKF. Η κατάσταση περιλαμβάνει:

- θέση,
- ταχύτητα,
- attitude,
- IMU accelerometer bias,
- IMU gyroscope bias.

Η IMU χρησιμοποιείται για propagation, ενώ οι οπτικές μετρήσεις χρησιμοποιούνται για correction. Η error-state μορφή είναι κατάλληλη για visual-inertial navigation, επειδή εκτιμά σφάλματα γύρω από την ονομαστική κατάσταση.

## Visual front-end

Το visual front-end βασίζεται σε pyramidal KLT optical flow. Εκτός από feature tracks, υπολογίζεται και visual quality score. Το score αυτό εξαρτάται από:

- πόσα features παρακολουθούνται,
- πόσο σταθερά είναι τα tracks,
- πόσο καλά απλώνονται τα features μέσα στην εικόνα.

Αν η εικόνα έχει λίγα ή κακής ποιότητας features, το measurement noise αυξάνεται. Έτσι το σύστημα δεν εμπιστεύεται υπερβολικά αδύναμες οπτικές μετρήσεις.

## Uncertainty quantification

Το project δεν περιορίζεται στο pose estimate. Υπολογίζει και μετρικές αβεβαιότητας, όπως:

- covariance trace,
- covariance determinant,
- differential entropy,
- Mahalanobis distance,
- confidence ellipsoids,
- NEES,
- NIS,
- συνολικό risk score.

Αυτές οι μετρικές βοηθούν να αξιολογηθεί αν το σύστημα είναι ακριβές αλλά και αν είναι συνεπές ως προς την εκτιμώμενη αβεβαιότητά του.

## Risk-aware navigation

Η πλοήγηση οργανώνεται ως state machine:

| Κατάσταση | Περιγραφή |
| --- | --- |
| NORMAL | χαμηλή αβεβαιότητα και κανονική πλοήγηση |
| CAUTIOUS | αυξημένη αβεβαιότητα, χαμηλότερη ταχύτητα |
| RECOVERY | υψηλό risk, προσπάθεια ανάκτησης αξιόπιστης κατάστασης |
| HALT | κρίσιμο risk, ασφαλής παύση |

Το navigation layer χρησιμοποιεί την αβεβαιότητα για να ρυθμίσει ταχύτητα, margins και συμπεριφορά ασφαλείας.

## Evaluation

Η αξιολόγηση περιλαμβάνει:

- Absolute Trajectory Error,
- Relative Pose Error,
- Umeyama Sim(3) alignment,
- tracking-failure rate,
- NEES/NIS consistency bounds.

Έτσι το project αξιολογεί τόσο την ακρίβεια της τροχιάς όσο και την αξιοπιστία της πιθανοτικής εκτίμησης.

## Πειραματικό πρωτόκολλο

Το repository περιγράφει stress tests για challenging environments:

- motion blur,
- low light,
- dynamic objects,
- fast motion,
- feature-poor scenes,
- sensor noise,
- IMU drift.

Τα πειράματα στοχεύουν να δείξουν πότε αυξάνεται η αβεβαιότητα και πώς πρέπει να αντιδρά η πλοήγηση.

## Εκτέλεση

```bash
git clone https://github.com/PanagiotaGr/aegis-vio.git
cd aegis-vio
pip install -r requirements.txt
pip install -e .
python scripts/run_simulation.py --scenario mixed --output_dir results/simulation
pytest tests/ -v
```

Για EuRoC:

```bash
python scripts/download_euroc.py --sequence MH_01_easy
python scripts/run_euroc.py --dataset_root ./datasets/MH_01_easy/mav0 --output_dir ./results/MH_01_easy
python scripts/evaluate_results.py --results_dir ./results/MH_01_easy --dataset_root ./datasets/MH_01_easy/mav0
```

Για ROS2:

```bash
ros2 launch aegis_vio aegis_vio.launch.py
```

## Περιορισμοί

Το project είναι research prototype. Το measurement update στο `vio_estimator.py` αναφέρεται ως απλοποιημένο placeholder και λειτουργεί ως βάση για μελλοντική επέκταση σε πιο πλήρες VIO backend.

## Μελλοντικές βελτιώσεις

- learned uncertainty prediction,
- πλήρης landmark triangulation,
- factor-graph back-end,
- active viewpoint selection,
- risk-aware trajectory planning,
- uncertainty-aware keyframe selection.

## Τι κάναμε αναλυτικά

Στο repository προστέθηκε ελληνική τεκμηρίωση ώστε το project να μπορεί να παρουσιαστεί καθαρά ως ερευνητική και πανεπιστημιακή εργασία.

Συγκεκριμένα:

- αναλύθηκε η βασική ιδέα του uncertainty-aware VIO,
- εξηγήθηκε γιατί η εκτίμηση θέσης πρέπει να συνοδεύεται από εκτίμηση αξιοπιστίας,
- περιγράφηκε η αρχιτεκτονική του project ανά layer,
- τεκμηριώθηκε ο ρόλος του Error-State EKF,
- εξηγήθηκε το visual front-end και το visual quality score,
- παρουσιάστηκαν οι uncertainty metrics,
- περιγράφηκε το risk-aware navigation state machine,
- καταγράφηκαν τα evaluation metrics,
- προστέθηκε ελληνικό project report στο `docs/PROJECT_REPORT_EL.md`,
- προστέθηκε ελληνικός οδηγός εγκατάστασης στο `docs/INSTALLATION_EL.md`,
- δημιουργήθηκε PDF report για παράδοση ή παρουσίαση.

## Συμπέρασμα

Το AEGIS-VIO δείχνει ότι η αβεβαιότητα δεν πρέπει να μένει απλώς ως αριθμός στο estimator. Πρέπει να επηρεάζει τη συμπεριφορά του robot. Η σύνδεση pose estimation, uncertainty quantification και risk-aware navigation είναι η βασική συνεισφορά του project.
