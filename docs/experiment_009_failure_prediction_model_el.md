# Experiment 009 – Πρόβλεψη Αποτυχίας Εντοπισμού με Random Forest

## Στόχος

Στόχος του πειράματος είναι η εκπαίδευση ενός μοντέλου μηχανικής μάθησης για την πρόβλεψη πιθανής μελλοντικής αποτυχίας εντοπισμού.

## Είσοδοι

Χρησιμοποιήθηκαν χαρακτηριστικά από το οπτικό και αδρανειακό υποσύστημα:

- matches
- inlier_ratio
- quality
- visual_uncertainty
- imu_instability
- multimodal_uncertainty

## Μοντέλο

Random Forest Classifier με class balancing.

## Αποτελέσματα

- Accuracy: 0.85
- Failure Precision: 0.59
- Failure Recall: 0.87
- Failure F1-score: 0.70
- ROC-AUC: 0.8611

Confusion Matrix:

```text
[[51  9]
 [ 2 13]]
