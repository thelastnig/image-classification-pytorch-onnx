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

  pipeline_name = 'image_classification_pytorch_onnx_conversion'
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


  @dsl.pipeline(
    name=pipeline_name,
    description='test for pipeline')
  def pipeline(aws_access_key_id="smr",
               aws_secret_access_key="smr0701!",
               mlflow_s3_endpoint_url="http://175.197.4.122:9000",
               mlflow_tracking_uri="http://175.197.4.214:5000",
               aws_default_region="SEOUL",
               pachyderm_host_uri="14.36.0.193",
               sm_aip_db_name="sm_aip",
               ml_flow_db_name="mlflow_db",
               sm_aip_db_user="sm_dev",
               sm_aip_db_password="txvq39zss4uc",
               sm_aip_db_host="175.197.4.150",
               sm_aip_db_port="3306",
               algorithm_name="resnet18",
               model_name="format_conversion_test",
               model_version="5",
               dataset_name="56_dev_test1_comscars2015_030910331615253607",
               image_depth=3,
               image_height=224,
               image_width=224,
               run_uuid="acb95354cd0142e88a7b45c439c0dc4d",
               user_id="dev_test1",
               project_id=52
               ):
    op = dsl.ContainerOp(
      name=pipeline_name, image="thelastnig/pt-image-classification-onnx:4.0",
      arguments=["--model_name", model_name,
                 "--model_version", model_version,
                 "--dataset_name", dataset_name,
                 "--algorithm_name", algorithm_name,
                 "--image_depth", image_depth,
                 "--image_height", image_height,
                 "--image_width", image_width,
                 "--run_uuid", run_uuid,
                 "--user_id", user_id,
                 "--project_id", project_id,
                 ]
    ).add_env_variable(V1EnvVar(name="AWS_ACCESS_KEY_ID", value=aws_access_key_id))\
      .add_env_variable(V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=aws_secret_access_key))\
      .add_env_variable(V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value=mlflow_s3_endpoint_url))\
      .add_env_variable(V1EnvVar(name="PACHYDERM_HOST_URI", value=pachyderm_host_uri))\
      .add_env_variable(V1EnvVar(name="MLFLOW_TRACKING_URI", value=mlflow_tracking_uri))\
      .add_env_variable(V1EnvVar(name="AWS_DEFAULT_REGION", value=aws_default_region))\
      .add_env_variable(V1EnvVar(name="SM_AIP_DB_NAME", value=sm_aip_db_name))\
      .add_env_variable(V1EnvVar(name="ML_FLOW_DB_NAME", value=ml_flow_db_name))\
      .add_env_variable(V1EnvVar(name="SM_AIP_DB_USER", value=sm_aip_db_user))\
      .add_env_variable(V1EnvVar(name="SM_AIP_DB_PASSWORD", value=sm_aip_db_password))\
      .add_env_variable(V1EnvVar(name="SM_AIP_DB_HOST", value=sm_aip_db_host))\
      .add_env_variable(V1EnvVar(name="SM_AIP_DB_PORT", value=sm_aip_db_port))\
      .set_gpu_limit(1)


  pipeline_file_nm = f"{pipeline_name}.tar.gz"
  compiler.Compiler().compile(pipeline, pipeline_file_nm)
  client = kfp.Client("http://14.36.0.193:31380/pipeline")
  try:
    client.upload_pipeline(pipeline_file_nm, f"{pipeline_name}_{timestamp}")
  except TypeError:
    pass
