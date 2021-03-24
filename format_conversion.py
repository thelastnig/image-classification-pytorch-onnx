import argparse
import os
from src.models import *
from src.pachy_meta_dataset import PachyClassificationMetaDataset
from minio_manager import MinioManager
from mlflow_manager import MlflowManager
import torch


def main(args):
    dataset = PachyClassificationMetaDataset(f'{args.dataset_name}/master', '/')
    model_cls = eval(args.algorithm_name)
    torch_model = model_cls(num_classes=dataset.num_classes)

    model_url = 'model.pth'
    onnx_model_url = "model.onnx"
    batch_size = 1

    minio_manager = MinioManager(args.run_uuid)
    minio_manager.load_model_weights(model_url)

    # ONNX 변환 부분 시작 -------------------------------------------------
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(torch.load(model_url, map_location=map_location), strict=False)

    torch_model.eval()

    x = torch.randn(batch_size, args.image_depth, args.image_height, args.image_width, requires_grad=True)

    torch.onnx.export(torch_model,
                      x,
                      onnx_model_url,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}}
                      )
    # ONNX 변환 부분 끝 -------------------------------------------------

    mlflow_manager = MlflowManager()
    mlflow_manager.registered_model_version(onnx_model_url, args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default="cocoval", type=str, help="dataset_name")
    parser.add_argument('--algorithm_name', default="resnet18", type=str, help="algorithm_name")
    parser.add_argument('--image_depth', default=3, type=int, help="image_depth")
    parser.add_argument('--image_height', default=256, type=int, help="image_height")
    parser.add_argument('--image_width', default=256, type=int, help="image_width")
    parser.add_argument('--model_name', type=str, help="model_name")
    parser.add_argument('--run_uuid', type=str, help="run_uuid")
    args = parser.parse_args()
    main(args)