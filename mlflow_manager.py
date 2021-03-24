import mlflow
import os
import onnx


class MlflowManager:
    def __init__(self):
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    def registered_model_version(self, onnx_model_url, model_name):
        onnx_model = onnx.load_model(onnx_model_url)
        mlflow.onnx.log_model(onnx_model, artifact_path="model", registered_model_name=model_name)