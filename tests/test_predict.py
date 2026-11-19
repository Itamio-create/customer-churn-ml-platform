"""
Unit tests for inference and prediction engine.
"""
import pytest
import pandas as pd
from src.predict import ChurnPredictor
from data.raw.generate_dataset import generate_telco_churn_dataset
from src.config import SAVED_MODEL_PATH

def test_risk_classification():
    assert ChurnPredictor.classify_risk(0.85) == "High Risk"
    assert ChurnPredictor.classify_risk(0.45) == "Moderate Risk"
    assert ChurnPredictor.classify_risk(0.15) == "Low Risk"

def test_inference_execution():
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet generated.")

    predictor = ChurnPredictor(SAVED_MODEL_PATH)
    sample_profile = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0
    }

    res = predictor.predict_single(sample_profile)
    assert "churn_prediction" in res
    assert res["churn_prediction"] in ["Churn", "Retain"]
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_tier"] in ["High Risk", "Moderate Risk", "Low Risk"]
    assert isinstance(res["key_risk_factors"], list)

    # Test batch
    raw_df = generate_telco_churn_dataset(num_records=15)
    batch_res = predictor.predict_batch(raw_df)
    assert "Churn_Probability" in batch_res.columns
    assert "Churn_Prediction" in batch_res.columns
    assert len(batch_res) == 15
