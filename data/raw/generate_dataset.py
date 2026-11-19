"""
Synthetic Telco Churn Dataset Generator
Generates a realistic, benchmark-grade customer churn dataset matching the schema
and statistical distribution of the classic IBM Telco Customer Churn benchmark.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_DATA_PATH, RANDOM_STATE

def generate_telco_churn_dataset(num_records: int = 7043, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates realistic customer churn dataset with known structural relationships:
    - Month-to-month contracts have substantially higher churn.
    - Higher tenure reduces churn risk.
    - Fiber optic customers without Tech Support / Online Security have higher churn.
    - Payment by Electronic check is correlated with higher churn.
    """
    np.random.seed(random_state)
    
    # 1. Customer Identifiers & Demographics
    customer_ids = [f"{np.random.randint(1000, 9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=5))}" for _ in range(num_records)]
    gender = np.random.choice(["Male", "Female"], size=num_records, p=[0.505, 0.495])
    senior_citizen = np.random.choice([0, 1], size=num_records, p=[0.838, 0.162])
    partner = np.random.choice(["Yes", "No"], size=num_records, p=[0.483, 0.517])
    
    # Dependents correlated with partner
    dependents = []
    for p in partner:
        if p == "Yes":
            dependents.append(np.random.choice(["Yes", "No"], p=[0.55, 0.45]))
        else:
            dependents.append(np.random.choice(["Yes", "No"], p=[0.10, 0.90]))
            
    # 2. Tenure & Contract Type
    # Bimodal tenure distribution (many new customers, many loyal long-term)
    size1 = int(num_records * 0.45)
    size2 = num_records - size1
    tenure_choices = np.concatenate([
        np.random.exponential(scale=12, size=size1),
        np.random.normal(loc=55, scale=12, size=size2)
    ])
    tenure = np.clip(np.round(tenure_choices).astype(int), 1, 72)
    
    contract = []
    for t in tenure:
        if t < 12:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.85, 0.12, 0.03]))
        elif t < 36:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.45, 0.35, 0.20]))
        else:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.15, 0.30, 0.55]))
            
    # 3. Phone & Internet Services
    phone_service = np.random.choice(["Yes", "No"], size=num_records, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.46, 0.54]))
            
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=num_records, p=[0.44, 0.34, 0.22])
    
    # Sub-services depend on internet service
    def get_addon_service(internet_series, yes_prob=0.35):
        services = []
        for isrv in internet_series:
            if isrv == "No":
                services.append("No internet service")
            else:
                services.append(np.random.choice(["Yes", "No"], p=[yes_prob, 1 - yes_prob]))
        return services

    online_security = get_addon_service(internet_service, 0.36)
    online_backup = get_addon_service(internet_service, 0.44)
    device_protection = get_addon_service(internet_service, 0.43)
    tech_support = get_addon_service(internet_service, 0.37)
    streaming_tv = get_addon_service(internet_service, 0.49)
    streaming_movies = get_addon_service(internet_service, 0.50)
    
    paperless_billing = np.random.choice(["Yes", "No"], size=num_records, p=[0.592, 0.408])
    payment_method = np.random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ], size=num_records, p=[0.336, 0.229, 0.219, 0.216])
    
    # 4. Charges Calculation
    monthly_charges = []
    for i in range(num_records):
        base = 20.0
        if internet_service[i] == "DSL":
            base += 25.0
        elif internet_service[i] == "Fiber optic":
            base += 50.0
            
        if multiple_lines[i] == "Yes":
            base += 5.0
        if online_security[i] == "Yes":
            base += 7.0
        if online_backup[i] == "Yes":
            base += 6.0
        if device_protection[i] == "Yes":
            base += 6.0
        if tech_support[i] == "Yes":
            base += 7.0
        if streaming_tv[i] == "Yes":
            base += 10.0
        if streaming_movies[i] == "Yes":
            base += 10.0
            
        noise = np.random.normal(0, 2.5)
        charge = max(18.25, round(base + noise, 2))
        monthly_charges.append(charge)
        
    monthly_charges = np.array(monthly_charges)
    # Total charges with some slight variation
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, size=num_records), 2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # 5. Realistic Churn Probability Formula (Log-Odds)
    log_odds = (
        -1.50
        + 1.35 * (np.array(contract) == "Month-to-month")
        - 0.75 * (np.array(contract) == "Two year")
        - 0.045 * (tenure)
        + 0.015 * (monthly_charges - 65)
        + 0.50 * (internet_service == "Fiber optic")
        - 0.45 * (np.array(tech_support) == "Yes")
        - 0.40 * (np.array(online_security) == "Yes")
        + 0.35 * (senior_citizen == 1)
        + 0.45 * (payment_method == "Electronic check")
        + 0.20 * (paperless_billing == "Yes")
        - 0.25 * (np.array(dependents) == "Yes")
    )
    
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn_labels = np.where(np.random.binomial(1, churn_prob) == 1, "Yes", "No")
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels
    })
    
    return df

if __name__ == "__main__":
    df = generate_telco_churn_dataset()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"Generated {len(df)} records. Saved to: {RAW_DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(f"Overall Churn rate: {churn_rate:.2f}%")
