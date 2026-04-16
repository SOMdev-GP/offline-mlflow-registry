import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================
# 📊 LOAD DATA
# =========================
df = pd.read_csv("data/sample.csv")

X = df[["feature"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🚨 THRESHOLD (PLACE HERE)
# =========================
THRESHOLD = df["target"].var() * 0.1
print("MSE Threshold:", THRESHOLD)

# =========================
# 🚀 MLflow SETUP
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("offline_mlflow_registry")

# =========================
# 🧠 TRAINING
# =========================
with mlflow.start_run() as run:

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_metric("mse", mse)

    # 🚨 QUALITY GATE
    if mse > THRESHOLD:
        raise Exception(f"Model rejected! MSE={mse} > THRESHOLD={THRESHOLD}")

    mlflow.sklearn.log_model(model, "model")

    print("Run ID:", run.info.run_id)
    print("MSE:", mse)