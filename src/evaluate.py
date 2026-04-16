import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = input("Enter Run ID: ")

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

df = pd.read_csv("data/sample.csv")
X = df[["feature"]]
y = df["target"]

preds = model.predict(X)

mse = mean_squared_error(y, preds)

print("Evaluation MSE:", mse)