"""
Evaluation and Metrics Visualization Module
Computes classification metrics, confusion matrices, ROC/PR curves,
and saves evaluation reports and plots.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any

# Ensure matplotlib uses writable cache
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from src.config import FIGURES_DIR, MODEL_METRICS_PATH

def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    """
    Computes all standard binary classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "support_positive": int((y_true == 1).sum()),
        "support_negative": int((y_true == 0).sum()),
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, output_path: Path = None):
    """
    Plots and saves styled confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        ax=ax,
        cbar=False
    )
    ax.set_title("Customer Churn Confusion Matrix", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    if output_path is None:
        output_path = FIGURES_DIR / "confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def plot_roc_pr_curves(y_true, y_proba, output_path: Path = None):
    """
    Plots both ROC and Precision-Recall curves side by side.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Chance")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PR Curve
    no_skill = (y_true == 1).mean()
    ax2.plot(recall, precision, color="#10B981", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    ax2.plot([0, 1], [no_skill, no_skill], color="gray", linestyle="--", lw=1.5, label=f"Baseline ({no_skill:.2f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path is None:
        output_path = FIGURES_DIR / "roc_pr_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def save_metrics_report(metrics_dict: Dict[str, Any], filepath: Path = MODEL_METRICS_PATH):
    """
    Saves metrics summary as formatted JSON.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {filepath}")
