import os
import pytest
import pandas as pd
from src.data_pipeline import load_and_preprocess_data

def test_data_pipeline_loads_properly():
    # Using a dummy file to ensure pipeline works regardless of the real data present in CI
    dummy_csv = "dummy_data.csv"
    pd.DataFrame({
        "customerID": ["A1"],
        "TotalCharges": ["20.5"],
        "Churn": ["Yes"],
        "gender": ["Male"]
    }).to_csv(dummy_csv, index=False)
    
    try:
        df, encoders = load_and_preprocess_data(dummy_csv, save_encoders=False)
        assert len(df) == 1
        assert 'customerID' not in df.columns
        assert df['TotalCharges'].iloc[0] == 20.5
        assert df['Churn'].iloc[0] == 1
    finally:
        if os.path.exists(dummy_csv):
            os.remove(dummy_csv)
