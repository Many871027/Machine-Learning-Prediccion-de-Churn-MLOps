import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def load_and_preprocess_data(filepath="WA_Fn-UseC_-Telco-Customer-Churn.csv", save_encoders=True):
    """
    Loads Telco Churn CSV and preprocesses it.
    If save_encoders=True, saves LabelEncoders to artifacts/label_encoders.pkl
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # 1. Eliminar customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # 2. Convertir TotalCharges a numérico
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 3. Eliminar nulos
    df = df.dropna()
    
    # 4. Codificar variable objetivo
    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    elif 'Churn' in df.columns:
        # already encoded
        pass
    
    # 5. Codificar variables categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    if save_encoders:
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(encoders, "artifacts/label_encoders.pkl")
        
    return df, encoders

def preprocess_new_data(new_data_dict, encoders_path="artifacts/label_encoders.pkl"):
    """
    Preprocess new inference data using saved encoders.
    """
    df = pd.DataFrame([new_data_dict])
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        for col in df.columns:
            if col in encoders:
                # To handle unknown classes during inference, simple mapping:
                known_classes = list(encoders[col].classes_)
                # default to first known label if unseen
                df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
                df[col] = encoders[col].transform(df[col])
                
    return df
