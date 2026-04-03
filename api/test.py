import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://13.49.223.143:8000/")

client = MlflowClient()
model = client.get_model_version("yt_chrome_plugin_model", "1")

print(model.run_id)