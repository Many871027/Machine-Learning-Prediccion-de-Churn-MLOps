#  MLOps Telco Churn Prediction System

Plataforma MLOps *End-to-End* diseñada para predecir la fuga de clientes (Churn) utilizando modelos experimentales paramétricos (XGBoost, Random Forest, SVM, etc.) optimizados hiper-paramétricamente con estrategias robustas contra el desbalanceo estructural de clases (SMOTE, Pipelines Imblearn). El ecosistema está nativamente acoplado a un servidor de observabilidad de hiperparámetros (MLflow) y a un microservicio expuesto por Pydantic y FastAPI.

##  Estructura del Nodo del Proyecto

- `app/`: Aplicación backend FastAPI para inferencia corporativa asíncrona en producción.
- `src/`: Tuberías ETL de datos (Data Pipelines) modulares y rutinas lógicas de iteración.
- `notebooks/`: Sandbox de experimentación científica inicial y desarrollo.
- `artifacts/`: Compilados matemáticos serializados y encodificados pre-operacionales (`.pkl`).
- `mlruns/`: Registro histórico local y automatizado de MLflow para telemetría profunda.
- `tests/`: Batería de pruebas estructurales unitarias para Integración Continua sobre Github Actions.

---

## 🛠️ 1. Instalación y Configuración Inicial

### Clonar el repositorio y acceder a la terminal:
```bash
git clone <URL_DEL_REPOSITORIO>
cd Final-SL
```

### Instaurar el Entorno Virtual Aislado:
Aislar el entorno de trabajo asegura reproducibilidad entre distintos computadores (Windows powershell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Sincronización de Dependencias:
```powershell
pip install -r requirements.txt
```
*(Asegúrate de que el dataset original `WA_Fn-UseC_-Telco-Customer-Churn.csv` se encuentre anclado en la raíz del proyecto para uso general).*

---

## 🔬 2. Entrenamiento y Sistema de Experimentación

Nuestra arquitectura de MLOps compite automáticamente docenas de arquitecturas algorítmicas utilizando validación cruzada estratificada sobre `GridSearchCV`.

**Opción A: Entrenamiento Remoto Silencioso (Recomendado)**
```powershell
.\venv\Scripts\python run_experiments.py
```
**Opción B: Entrenamiento Clásico Interactivo**
Abre e interactúa experimentalmente tu flujo desde el entorno de `notebooks/churn-prediction.ipynb`.

### Observabilidad (MLflow Tracking Server)
Una vez desencadenada la optimización, todas las métricas puras, el código, y gráficas biológicas como la Matriz de Confusión y Feature Importance habrán sido interceptadas. Revisa analíticamente tu interfaz ejecutando:
```powershell
.\venv\Scripts\mlflow ui
```
🌐 Navega a **[http://localhost:5000](http://localhost:5000)** y selecciona tu experimento vital `Churn_Prediction_Experiments`.

---

## ⚙️ 3. Integración a Producción

Cuando hayas validado el modelo asiduamente (observado objetivamente a través de MLflow), actualiza el perfil base embebido en `src/model_pipeline.py`. Después compila tu contenedor nativo con:

```powershell
.\venv\Scripts\python -m src.model_pipeline
```
*Este comando instanciará, congelará los estandarizadores numéricos, transformadores, el propio modelo XGBoost_Production predictivo, y los enviará automáticamente a tu directorio `/artifacts` esperando el salto a Inferencia.*

---

## 🌐 4. Despliegue del Sistema de Inferencia Corporativo (FastAPI)

Abre un canal seguro levantando a través del servidor especializado Uvicorn:
```powershell
.\venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Swagger UI y Pruebas Unitarias
Navega inmediatamente a tu módulo de validación visual e interceptación RESTful:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

Tendrás 2 micro-sistemas listos a detonación directa:
- 📊 **`GET /metrics`**: Extraerá de forma paralela en la web la tabla oficial telemetríca incrustada en el Tracking URI de MLflow para auditar ordenamientos AUC_ROC.
- 🎯 **`POST /predict`**: Un endpoint vivo transaccional en JSON. Alimenta sus `CustomerFeatures` default, apóyate en su procesador embebido internamente, y obtén un logeo vivo en formato binario del comportamiento financiero esperado, resultando en clasificaciones atípicas (ej. `{"churn": 1, "probability": 0.776}`).
