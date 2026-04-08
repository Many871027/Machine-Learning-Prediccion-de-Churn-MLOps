import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from src.data_pipeline import load_and_preprocess_data
from src.model_pipeline import evaluar_y_registrar
from src.config import get_model_configs

df, encoders = load_and_preprocess_data('WA_Fn-UseC_-Telco-Customer-Churn.csv', save_encoders=False)
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

ratio_neg_pos = sum(y_train==0)/sum(y_train==1)
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
modelos_cfg = get_model_configs(ratio_neg_pos=ratio_neg_pos)

for nombre_modelo, cfg in modelos_cfg.items():
    print(f"-> Optimizando y evaluando: {nombre_modelo}")
    grid = GridSearchCV(
        estimator=cfg['estimator'],
        param_grid=cfg['param_grid'],
        cv=cv_strategy,
        scoring='f1',
        n_jobs=-1, verbose=0, refit=True
    )
    grid.fit(X_train_scaled, y_train)
    mejores_params = grid.best_params_
    res = evaluar_y_registrar(grid.best_estimator_, nombre_modelo, X_train_scaled, X_test_scaled, y_train, y_test, best_params=mejores_params)
    print(f"[{nombre_modelo}] Finalizado. F1-Score: {res['f1']:.4f}")
