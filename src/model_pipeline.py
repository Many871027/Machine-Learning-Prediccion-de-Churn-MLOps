import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.metrics import (confusion_matrix, precision_score, accuracy_score, recall_score,
                             f1_score, roc_auc_score, log_loss, roc_curve)

def evaluar_y_registrar(model, model_name, X_tr, X_te, y_tr, y_te, cv_strategy=None, best_params=None, color='#3498db'):
    """
    Evalúa un modelo, calcula métricas, genera artefactos (matriz confusión, ROC, Feature Importance) y registra en MLflow.
    """
    os.makedirs("artifacts", exist_ok=True)
    
    if best_params is None:
        best_params = {}

    y_pred = model.predict(X_te)
    
    # Extraer modelo base si es un Pipeline
    base_model = model.steps[-1][1] if hasattr(model, 'steps') else model
    
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(base_model, 'predict_proba') else model.predict(X_te)
    
    # Metricas de rendimiento
    matriz = confusion_matrix(y_te, y_pred)
    precision = precision_score(y_te, y_pred, zero_division=0)
    exactitud = accuracy_score(y_te, y_pred)
    sensibilidad = recall_score(y_te, y_pred, zero_division=0)
    puntaje_f1 = f1_score(y_te, y_pred, zero_division=0)
    
    try:
        auc_val = roc_auc_score(y_te, y_proba)
        logloss = log_loss(y_te, y_proba)
    except:
        auc_val = 0.5
        logloss = 9.9
        
    tn, fp, fn, tp = matriz.ravel()
    
    # MLflow Start
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracking_uri = "file:///" + os.path.join(root_dir, "mlruns").replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("Churn_Prediction_Experiments")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", X_tr.shape[1])
        
        for k, v in best_params.items():
            mlflow.log_param(f"model_{k}", v)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("accuracy", exactitud)
        mlflow.log_metric("recall", sensibilidad)
        mlflow.log_metric("f1_score", puntaje_f1)
        mlflow.log_metric("auc_roc", auc_val)
        mlflow.log_metric("log_loss", logloss)
        
        mlflow.log_metric("TN", int(tn))
        mlflow.log_metric("FP", int(fp))
        mlflow.log_metric("FN", int(fn))
        mlflow.log_metric("TP", int(tp))

        # 1. Matriz Confusion
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f"Matriz de Confusión - {model_name}")
        cm_path = f"artifacts/confusion_matrix_{model_name}.png"
        fig_cm.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close(fig_cm)
        
        # 2. Curva ROC
        if hasattr(base_model, 'predict_proba'):
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
            ax_roc.plot([0,1], [0,1], 'r--')
            ax_roc.set_title(f"ROC - {model_name}")
            ax_roc.legend()
            roc_path = f"artifacts/roc_curve_{model_name}.png"
            fig_roc.savefig(roc_path)
            mlflow.log_artifact(roc_path)
            plt.close(fig_roc)

        # 3. Feature Importance (si el modelo lo soporta)
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            
            cols = [f"Feature {i}" for i in range(len(importances))]
            if hasattr(X_tr, 'columns'):
                cols = X_tr.columns
                
            sorted_idx = np.argsort(importances)[::-1]
            sns.barplot(x=importances[sorted_idx][:10], y=np.array(cols)[sorted_idx][:10], ax=ax_fi)
            ax_fi.set_title(f"Top 10 Feature Importance - {model_name}")
            fi_path = f"artifacts/feature_importance_{model_name}.png"
            fig_fi.tight_layout()
            fig_fi.savefig(fi_path)
            mlflow.log_artifact(fi_path)
            plt.close(fig_fi)

        # Log Model
        signature = infer_signature(X_tr, model.predict(X_tr))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

        mlflow.set_tag("model_type", model_name)
    
    return {
        "model_name": model_name,
        "precision": precision,
        "accuracy": exactitud,
        "recall": sensibilidad,
        "f1": puntaje_f1,
        "auc": auc_val
    }

def train_production_model(X, y):
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if hasattr(X_train, 'columns'):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Drop Model for FastAPI Native Loading
    import shutil
    if os.path.exists("artifacts/model"):
        shutil.rmtree("artifacts/model")
    mlflow.sklearn.save_model(model, "artifacts/model")
    
    res = evaluar_y_registrar(model, "XGBoost_Production", X_train_scaled, X_test_scaled, y_train, y_test)
    print(f"Production Model Trained. Acc: {res['accuracy']:.4f}, AUC: {res['auc']:.4f}")
    
    return model, scaler

if __name__ == "__main__":
    from src.data_pipeline import load_and_preprocess_data
    print("Loading data...")
    df, _ = load_and_preprocess_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print("Training production model...")
    train_production_model(X, y)
