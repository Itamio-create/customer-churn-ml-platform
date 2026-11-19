"""
Model Training and Comparison Module
Trains, validates, and compares multiple ML classifiers:
- Logistic Regression
- Random Forest
- HistGradientBoosting
Selects the champion model, evaluates on held-out test data, and persists pipeline.
"""
import time
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    SAVED_MODEL_PATH,
    MODEL_METRICS_PATH,
    RANDOM_STATE,
    CV_FOLDS,
    FIGURES_DIR
)
from src.data_loader import get_train_test_data
from src.pipeline import create_model_pipeline
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    save_metrics_report
)

def get_candidate_models() -> Dict[str, Any]:
    """
    Returns candidate classifiers with tuned hyperparameters for tabular churn data.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

def train_and_evaluate_all():
    """
    Main training workflow:
    1. Loads stratified train/test split.
    2. Runs 5-fold cross-validation on candidates.
    3. Fits candidates, selects champion model based on CV ROC-AUC.
    4. Evaluates champion on test set.
    5. Saves model artifact, visual plots, and metrics summary.
    """
    print("=" * 60)
    print("Starting Customer Churn Model Training & Evaluation")
    print("=" * 60)

    X_train, X_test, y_train, y_test = get_train_test_data()
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"Positive class (Churn=1) prevalence in train: {(y_train == 1).mean() * 100:.2f}%\n")

    candidates = get_candidate_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    model_results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        print(f"Training and validating: {name}...")
        start_time = time.time()
        pipeline = create_model_pipeline(model)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        mean_auc = cv_scores.mean()
        std_auc = cv_scores.std()

        # Fit full training set
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start_time

        fitted_pipelines[name] = pipeline
        model_results[name] = {
            "cv_mean_roc_auc": float(mean_auc),
            "cv_std_roc_auc": float(std_auc),
            "training_time_sec": round(elapsed, 2)
        }
        print(f"  -> 5-Fold CV ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f}) | Time: {elapsed:.2f}s")

    # Select champion
    best_name = max(model_results, key=lambda k: model_results[k]["cv_mean_roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nChampion Model Selected: '{best_name}' with CV ROC-AUC = {model_results[best_name]['cv_mean_roc_auc']:.4f}")

    # Evaluate Champion on Test Split
    print("\nEvaluating champion on held-out test dataset...")
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    print(f"  -> Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  -> Precision: {test_metrics['precision']:.4f}")
    print(f"  -> Recall   : {test_metrics['recall']:.4f}")
    print(f"  -> F1-Score : {test_metrics['f1_score']:.4f}")
    print(f"  -> ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"  -> PR-AUC   : {test_metrics['pr_auc']:.4f}")

    # Generate Evaluation Figures
    print("\nGenerating diagnostic performance plots...")
    cm_path = plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / "confusion_matrix.png")
    roc_pr_path = plot_roc_pr_curves(y_test, y_proba, FIGURES_DIR / "roc_pr_curves.png")
    print(f"  -> Saved {cm_path}")
    print(f"  -> Saved {roc_pr_path}")

    # Save Model Artifact
    SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, SAVED_MODEL_PATH)
    print(f"Saved champion pipeline artifact to: {SAVED_MODEL_PATH}")

    # Combine Summary
    summary = {
        "champion_model": best_name,
        "cross_validation_comparison": model_results,
        "test_metrics": test_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_metrics_report(summary, MODEL_METRICS_PATH)
    print("\nTraining workflow completed successfully!")
    return summary

if __name__ == "__main__":
    train_and_evaluate_all()
