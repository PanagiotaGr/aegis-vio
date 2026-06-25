# Experiment 008 - Δημιουργία Dataset Πρόβλεψης Αποτυχίας Εντοπισμού

## Στόχος

Στόχος του πειράματος είναι η δημιουργία ενός dataset που θα επιτρέψει την εκπαίδευση μοντέλων μηχανικής μάθησης για την πρόβλεψη μελλοντικών σφαλμάτων εντοπισμού (localization failures).

Η βασική ερευνητική υπόθεση είναι ότι τα σήματα αβεβαιότητας από το οπτικό και αδρανειακό υποσύστημα μπορούν να προβλέψουν μελλοντική υποβάθμιση της ακρίβειας του συστήματος.

---

## Είσοδοι

Για κάθε frame pair αποθηκεύονται:

- matches
- inlier_ratio
- visual_quality
- visual_uncertainty
- imu_instability
- multimodal_uncertainty

---

## Έξοδος

Για κάθε χρονική στιγμή θα υπολογίζεται:

future_localization_error

το οποίο προκύπτει από το ground truth trajectory του EuRoC dataset.

---

## Ερευνητικό Ερώτημα

Μπορούμε να προβλέψουμε μελλοντική αποτυχία εντοπισμού χρησιμοποιώντας μόνο δείκτες αβεβαιότητας;

---

## Αναμενόμενα Μοντέλα

- Random Forest
- XGBoost
- MLP Neural Network

---

## Αναμενόμενη Συνεισφορά

Ανάπτυξη ενός uncertainty-aware failure prediction module για Visual-Inertial Odometry συστήματα.
