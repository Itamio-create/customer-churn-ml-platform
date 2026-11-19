"""
Unit tests for preprocessing transformers and pipelines.
"""
import pytest
import numpy as np
from data.raw.generate_dataset import generate_telco_churn_dataset
from src.data_loader import clean_and_prepare_data
from src.pipeline import build_preprocessor, create_model_pipeline
from sklearn.linear_model import LogisticRegression
from src.config import ALL_FEATURE_COLUMNS

def test_preprocessor_transformation():
    raw_df = generate_telco_churn_dataset(num_records=40)
    clean_df = clean_and_prepare_data(raw_df)
    X = clean_df[ALL_FEATURE_COLUMNS]

    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    # Check numpy array output
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 40
    # Must have transformed numeric and one-hot encoded categorical columns (> 19)
    assert X_trans.shape[1] > len(ALL_FEATURE_COLUMNS)
    # No NaNs after preprocessing
    assert not np.isnan(X_trans).any()

def test_end_to_end_pipeline():
    raw_df = generate_telco_churn_dataset(num_records=50)
    clean_df = clean_and_prepare_data(raw_df)
    X = clean_df[ALL_FEATURE_COLUMNS]
    y = clean_df["Churn"]

    pipe = create_model_pipeline(LogisticRegression(max_iter=200))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    probas = pipe.predict_proba(X)

    assert len(preds) == 50
    assert probas.shape == (50, 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
