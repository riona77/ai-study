"""
AI Study Lab - Master Retraining Script
Retrains all 5 ML models (Logistic Regression, Decision Tree, Naive Bayes, SVM, K-Means),
updates serialized .pkl artifacts, and saves comprehensive accuracy metrics to model_metrics.json.
"""

from pathlib import Path
import json
import time

from logistic_regression import train_model as train_logistic
from decision_tree import train_model as train_decision
from naive_bayes import train_model as train_naive
from svm_question_classifier import train_model as train_svm
from kmeans import train_model as train_kmeans

BASE_DIR = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("[+] STARTING AI STUDY LAB MODEL RETRAINING & BENCHMARKING")
    print("=" * 60)
    start_time = time.time()

    print("\n[1/5] Retraining Logistic Regression (Course Recommender)...")
    _, acc_log = train_logistic()

    print("\n[2/5] Retraining Decision Tree (Pathway Level Predictor)...")
    _, acc_dt = train_decision()

    print("\n[3/5] Retraining Naive Bayes (Chapter Classifier)...")
    _, acc_nb = train_naive()

    print("\n[4/5] Retraining SVM (Question Type Classifier)...")
    _, acc_svm = train_svm()

    print("\n[5/5] Retraining K-Means (Course Cluster Explorer)...")
    _, sil_km = train_kmeans()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[+] RETRAINING COMPLETE IN {elapsed:.2f} SECONDS")
    print("=" * 60)
    print(f"[*] Logistic Regression Accuracy: {acc_log * 100:.2f}%")
    print(f"[*] Decision Tree Accuracy:        {acc_dt * 100:.2f}%")
    print(f"[*] Naive Bayes Accuracy:         {acc_nb * 100:.2f}%")
    print(f"[*] SVM Question Classifier:       {acc_svm * 100:.2f}%")
    print(f"[*] K-Means Silhouette Score:     {sil_km:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
