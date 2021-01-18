import argparse
from datetime import datetime

from kfp import dsl
from kfp.compiler import compiler
import kfp.components
from kubernetes.client.models import V1EnvVar, V1Volume

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Create a kubeflow pipeline')
  parser.add_argument('--docker-image', metavar="docker_img_name", type=str)
  args = parser.parse_args()

  pipeline_name = 'trainer_semantic_segmentation'
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


  @dsl.pipeline(
    name=pipeline_name,
    description='test for pipeline')
  def pipeline(aws_access_key_id,
               aws_secret_access_key,
               mlflow_s3_endpoint_url="http://175.197.4.122:9000",
               mlflow_tracking_uri="http://175.197.4.214:5000",
               aws_default_region="SEOUL",
               model_name="DeepV3PlusW38",
               split_type="T"):
    op = dsl.ContainerOp(
      name=pipeline_name, image=args.docker_image,
      arguments=["--model_name", model_name, "--split_type", split_type]
    ).add_pvolumes(
      {"/dev/shm": V1Volume(name="dshm", empty_dir={"medium": "Memory"})}
    ).add_env_variable(
      V1EnvVar(name="AWS_ACCESS_KEY_ID", value=aws_access_key_id)
    ).add_env_variable(
      V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=aws_secret_access_key)
    ).add_env_variable(
      V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value=mlflow_s3_endpoint_url)
    ).add_env_variable(
      V1EnvVar(name="MLFLOW_TRACKING_URI", value=mlflow_tracking_uri)
    ).add_env_variable(
      V1EnvVar(name="AWS_DEFAULT_REGION", value=aws_default_region)
    ).set_gpu_limit(1)


  pipeline_file_nm = f"{pipeline_name}.tar.gz"
  compiler.Compiler().compile(pipeline, pipeline_file_nm)
  client = kfp.Client("http://175.197.5.142:31380/pipeline")
  try:
    client.upload_pipeline(pipeline_file_nm, f"{pipeline_name}_{timestamp}")
  except TypeError:
    pass  # https://github.com/kubeflow/pipelines/issues/2764
    # This can be removed once KF proper uses the latest KFP
