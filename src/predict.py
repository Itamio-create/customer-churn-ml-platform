"""
Inference Engine Module
Provides clean inference functions for single customer profiles and batch predictions.
Assigns risk levels and computes calibrated risk scores.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Union

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import numpy as np
import pandas as pd
import joblib
from src.config import SAVED_MODEL_PATH, ALL_FEATURE_COLUMNS

class ChurnPredictor:
    def __init__(self, model_path: Path = SAVED_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Run 'python src/train.py' first."
            )
        self.pipeline = joblib.load(self.model_path)

    @staticmethod
    def classify_risk(probability: float) -> str:
        if probability >= 0.65:
            return "High Risk"
        elif probability >= 0.35:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on a single customer dictionary.
        """
        df = pd.DataFrame([customer_data])
        # Ensure all required columns exist
        for col in ALL_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[ALL_FEATURE_COLUMNS]
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = int(self.pipeline.predict(df)[0])
        risk_tier = self.classify_risk(prob)

        # Key risk indicators
        risk_flags = []
        if customer_data.get("Contract") == "Month-to-month":
            risk_flags.append("Month-to-month contract (High churn rate)")
        if customer_data.get("tenure", 50) < 12:
            risk_flags.append("New customer account (< 12 months tenure)")
        if customer_data.get("InternetService") == "Fiber optic" and customer_data.get("TechSupport") != "Yes":
            risk_flags.append("Fiber optic service without dedicated Tech Support")
        if customer_data.get("PaymentMethod") == "Electronic check":
            risk_flags.append("Payment via Electronic check")
        if customer_data.get("MonthlyCharges", 0) > 85.0:
            risk_flags.append("High monthly subscription tier (> $85/month)")

        return {
            "churn_prediction": "Churn" if prediction == 1 else "Retain",
            "churn_binary": prediction,
            "churn_probability": round(prob, 4),
            "churn_percentage": f"{prob * 100:.1f}%",
            "risk_tier": risk_tier,
            "key_risk_factors": risk_flags if risk_flags else ["Standard account metrics, low risk profile"]
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs inference on a pandas DataFrame of customer profiles.
        """
        features_df = df[ALL_FEATURE_COLUMNS].copy()
        probabilities = self.pipeline.predict_proba(features_df)[:, 1]
        predictions = self.pipeline.predict(features_df)

        result_df = df.copy()
        result_df["Churn_Probability"] = probabilities.round(4)
        result_df["Churn_Prediction"] = ["Churn" if p == 1 else "Retain" for p in predictions]
        result_df["Risk_Tier"] = [self.classify_risk(p) for p in probabilities]
        return result_df

# Convenience function
_predictor_instance = None

def get_predictor() -> ChurnPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ChurnPredictor()
    return _predictor_instance
