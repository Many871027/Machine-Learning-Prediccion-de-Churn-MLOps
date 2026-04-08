from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import json
import os
import joblib

app = FastAPI(
    title="Churn Prediction API",
    description="Inference API to calculate probability of a customer churning.",
    version="1.0.0"
)

from src.data_pipeline import preprocess_new_data

class CustomerFeatures(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    if not os.path.exists("artifacts/model"):
        raise HTTPException(status_code=500, detail="Model artifact missing. Run training pipeline first (`python -m src.model_pipeline`).")
    if not os.path.exists("artifacts/scaler.pkl"):
        raise HTTPException(status_code=500, detail="Scaler artifact missing.")
        
    try:
        # Preprocess
        input_df = preprocess_new_data(customer.dict(), "artifacts/label_encoders.pkl")
        
        # Scale
        scaler = joblib.load("artifacts/scaler.pkl")
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
        # Predict with locally registered model (could also fetch from MLflow tracking by URI)
        model = mlflow.pyfunc.load_model("artifacts/model")
        
        # Attempt to get probability
        if hasattr(model._model_impl, "predict_proba"):
            proba = model._model_impl.predict_proba(input_scaled)[0, 1]
        else:
            proba = model.predict(input_scaled)[0]
            
        churn_pred = 1 if proba >= 0.5 else 0
        return {"churn": churn_pred, "probability": float(proba), "warning": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Reads all experimental runs from MLflow tracking server and returns a comparative table."""
    try:
        from mlflow.tracking import MlflowClient
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tracking_uri = "file:///" + os.path.join(root_dir, "mlruns").replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)
        
        client = MlflowClient(tracking_uri=tracking_uri)
        
        experiment = client.get_experiment_by_name("Churn_Prediction_Experiments")
        if not experiment:
            return {"message": "No experiments found. Ensure you ran the notebook `notebooks/churn-prediction.ipynb` first!"}
            
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        table = []
        for r in runs:
            data = r.data.metrics
            table.append({
                "Model": r.data.params.get("model_type", "Unknown Model"),
                "Accuracy": round(data.get("accuracy", 0), 4),
                "AUC_ROC": round(data.get("auc_roc", 0), 4),
                "F1_Score": round(data.get("f1_score", 0), 4),
                "Precision": round(data.get("precision", 0), 4),
                "Recall": round(data.get("recall", 0), 4)
            })
            
        # Sort by AUC
        table = sorted(table, key=lambda x: x["AUC_ROC"], reverse=True)
        return {"comparative_table": table}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
