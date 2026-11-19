from setuptools import setup, find_packages

setup(
    name="customer_churn_ml_platform",
    version="1.0.0",
    author="Itamio-create",
    description="End-to-end Machine Learning System for Customer Churn Prediction and Retention Analytics",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.30.0",
        "pytest>=7.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
