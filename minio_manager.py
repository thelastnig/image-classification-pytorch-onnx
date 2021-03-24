from minio import Minio
import os


class MinioManager:
    def __init__(self, run_uuid):
        self.client = Minio(os.environ["MLFLOW_S3_ENDPOINT_URL"].split("//")[1],
                            os.environ["AWS_ACCESS_KEY_ID"],
                            os.environ["AWS_SECRET_ACCESS_KEY"],
                            secure=False)
        self.run_uuid = run_uuid

    def load_model_weights(self, weights_file_name):
        weights_path = f"0/{self.run_uuid}/artifacts/model/data/{weights_file_name}"
        self.client.fget_object("mlflow-storage", weights_path, './' + weights_file_name)
