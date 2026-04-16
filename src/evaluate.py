import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("file:./mlruns")

# READ RUN ID FROM FILE
with open("run_id.txt", "r") as f:
    run_id = f.read().strip()

print("Using Run ID:", run_id)

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

df = pd.read_csv("data/sample.csv")
X = df[["feature"]]
y = df["target"]

preds = model.predict(X)

mse = mean_squared_error(y, preds)

print("MSE:", mse)