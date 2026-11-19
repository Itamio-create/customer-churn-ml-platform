"""
Preprocessing Pipeline Module
Builds clean, leakage-free Scikit-learn pipelines with ColumnTransformer.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def build_preprocessor() -> ColumnTransformer:
    """
    Constructs the ColumnTransformer for numerical and categorical preprocessing:
    - Numerical: median imputation followed by standard scaling.
    - Categorical: most frequent imputation followed by one-hot encoding (ignores unknown levels).
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor

def create_model_pipeline(classifier) -> Pipeline:
    """
    Combines feature preprocessor with any estimator/classifier into an end-to-end Pipeline.
    """
    preprocessor = build_preprocessor()
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return model_pipeline
