import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load yourz training data
def extract_features(df):
    df = pd.read_csv("C:/Users/nisha/Desktop/credit_score_system/data/raw/cs-training.csv")

# Define the exact features used during training
    features = ['age', 'monthly_income', 'open_credit_lines', 'credit_history', 'loan_amount']
    X = df[features]
# Fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Save the fitted scaler
    joblib.dump(scaler, "C:/Users/nisha/Desktop/credit_score_system/models/scaler.pkl")
    print("Scaler saved successfully.")
    return pd.DataFrame(X_scaled, columns=features), scaler
