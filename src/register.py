import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = input("Enter Run ID to register: ")

model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name="LinearRegressionOfflineModel"
)

print("Model registered!")
print("Version:", result.version)