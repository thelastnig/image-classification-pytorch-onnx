import argparse

from kfp import dsl
from kfp.compiler import compiler
import kfp.components

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Create a kubeflow pipeline')
  parser.add_argument('--docker-image', metavar="docker_img_name", type=str)
  args = parser.parse_args()

  pipeline_name = 'trainer_classification'


  @dsl.pipeline(
    name=pipeline_name,
    description='test for pipeline')
  def pipeline():
    op = dsl.ContainerOp(name=pipeline_name,
                         image=args.docker_image,
                         )


  pipeline_file_nm = f"{pipeline_name}.tar.gz"
  compiler.Compiler().compile(pipeline, pipeline_file_nm)
  client = kfp.Client("http://175.197.4.150:31380/pipeline")
  try:
    client.upload_pipeline(pipeline_file_nm, pipeline_name)
  except TypeError:
    pass  # https://github.com/kubeflow/pipelines/issues/2764
    # This can be removed once KF proper uses the latest KFP
