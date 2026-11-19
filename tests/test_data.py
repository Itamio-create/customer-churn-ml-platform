"""
Unit tests for data generation and preprocessing routines.
"""
import pytest
import pandas as pd
from data.raw.generate_dataset import generate_telco_churn_dataset
from src.data_loader import clean_and_prepare_data, get_train_test_data
from src.config import ALL_FEATURE_COLUMNS, TARGET_COLUMN

def test_generate_telco_churn_dataset():
    df = generate_telco_churn_dataset(num_records=100)
    assert len(df) == 100
    assert TARGET_COLUMN in df.columns
    for col in ALL_FEATURE_COLUMNS:
        assert col in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({"Yes", "No"})

def test_clean_and_prepare_data():
    raw_df = generate_telco_churn_dataset(num_records=50)
    clean_df = clean_and_prepare_data(raw_df)
    
    assert set(clean_df[TARGET_COLUMN].unique()).issubset({0, 1})
    assert clean_df["TotalCharges"].dtype in [float, int, "float64"]
    assert clean_df["MonthlyCharges"].dtype in [float, int, "float64"]
    assert not clean_df[TARGET_COLUMN].isna().any()

def test_train_test_split():
    X_train, X_test, y_train, y_test = get_train_test_data()
    assert len(X_train) > len(X_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert set(y_train.unique()) == {0, 1}
