"""
Data Loader & Validation Module
Loads raw customer churn data, performs schema validation, type casting,
and creates stratified train-test splits.
"""
import sys
from pathlib import Path
from typing import Tuple

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE
)

def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw CSV data. If not found, generates synthetic benchmark data.
    """
    if not filepath.exists():
        from data.raw.generate_dataset import generate_telco_churn_dataset
        print(f"Data not found at {filepath}. Generating synthetic benchmark dataset...")
        df = generate_telco_churn_dataset()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath)
    return df

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Casts numeric columns appropriately.
    - Strips whitespace.
    - Encodes binary target 'Churn' -> 1 (Yes) / 0 (No).
    """
    data = df.copy()

    # Drop customerID if present
    if ID_COLUMN in data.columns:
        data = data.drop(columns=[ID_COLUMN])

    # Convert TotalCharges to numeric if read as string/object
    if "TotalCharges" in data.columns and data["TotalCharges"].dtype == object:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].str.strip(), errors="coerce")
        # Impute missing with MonthlyCharges * tenure or median
        data["TotalCharges"] = data["TotalCharges"].fillna(data["MonthlyCharges"] * data["tenure"])

    # Ensure numerical columns are numeric
    for col in NUMERICAL_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Map Churn to binary integer if present
    if TARGET_COLUMN in data.columns:
        if data[TARGET_COLUMN].dtype not in [int, "int64", "int32"]:
            data[TARGET_COLUMN] = (
                data[TARGET_COLUMN]
                .astype(str)
                .str.strip()
                .str.capitalize()
                .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
            )
            # Default any unmapped to 0 and cast to int
            data[TARGET_COLUMN] = data[TARGET_COLUMN].fillna(0).astype(int)

    return data

def get_train_test_data(
    filepath=RAW_DATA_PATH,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns stratified train/test features and target.
    """
    df = load_raw_data(filepath)
    df_clean = clean_and_prepare_data(df)

    X = df_clean[ALL_FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    print(f"X_train shape: {X_train.shape}, y_train class distribution: \n{y_train.value_counts(normalize=True)}")
    print(f"X_test shape: {X_test.shape}, y_test class distribution: \n{y_test.value_counts(normalize=True)}")
