import argparse
import copy
import os

import mlflow
import torch

from src.models import *
from src.pachy_dataset import PachySemanticDataset
from src.split import SplitDataset, CrossValidationDataset
from src.trainer import SemanticSegmentationTrainer, SemanticSegmentationCVWrapper
import config


def main(args):
  os.environ["GIT_PYTHON_REFRESH"] = "quiet"
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Running on {device}")

  if args.model_name in globals():
    model_cls = eval(args.model_name)
  else:
    raise ValueError(f"{args.model_name} is not a registered model name")

  dataset = PachySemanticDataset(f'{args.dataset_name}/master', '/')
  hyper_dict = {
    'epochs': args.epoch,
    'batch_size': args.batch_size,
    'num_workers': 2,
    'optimizer': 'adam',
    'learning_rate': args.lr,
    'weight_decay': args.weight_decay,
  }
  config.assert_and_infer_cfg(train_mode=True)

  if args.split_type == "T":
    split_total = args.split_training + args.split_validation + args.split_test
    train_val_barrier = args.split_training / split_total
    val_test_barrier = (args.split_training + args.split_validation) / split_total

    train_dataset = SplitDataset(dataset, (0., train_val_barrier), args.split_seed)
    val_dataset = SplitDataset(dataset, (train_val_barrier, val_test_barrier), args.split_seed)
    test_dataset = SplitDataset(dataset, (val_test_barrier, 1.), args.split_seed)

    model = model_cls(num_classes=dataset.num_classes)
    trainer = SemanticSegmentationTrainer(
      train_dataset, val_dataset, test_dataset, model,
      hyper_dict, args.experiment_name, device)
    trainer.train()
  elif args.split_type == "C":
    split_total = args.num_cv_folds + 1
    val_test_barrier = args.num_cv_folds / split_total

    trainval_dataset = SplitDataset(dataset, (0., val_test_barrier), args.split_seed)
    test_dataset = SplitDataset(dataset, (val_test_barrier, 1.), args.split_seed)
    cv_dataset = CrossValidationDataset(trainval_dataset, args.num_cv_folds, args.split_seed)
    trainer = SemanticSegmentationCVWrapper(
      cv_dataset, test_dataset, lambda: model_cls(num_classes=dataset.num_classes),
      hyper_dict, args.experiment_name, device)
    trainer.train()
  else:
    raise ValueError(f"Unknown split_type {args.split_type}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Learning parameters
  parser.add_argument('--dataset_name', default="voctrainval", type=str, help="dataset_name")
  parser.add_argument('--batch_size', default=32, type=int, help="batch_size")
  parser.add_argument('--epoch', default=10, type=int, help="epoch")
  parser.add_argument('--lr', default=1e-3, type=float, help="lr")
  parser.add_argument('--model_name', default="DeepV3PlusW38", type=str, help="model_name")
  parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
  parser.add_argument('--weight_decay', default=5e-4, type=float, help="weight_decay")
  parser.add_argument('--experiment_name', default="any_exp", type=str, help="exp_name")
  parser.add_argument('--split_type', default="C", type=str, help="split_type")
  parser.add_argument('--num_cv_folds', default=5, type=int, help="num_cv_folds")
  parser.add_argument('--split_training', default=80, type=int, help="split_training")
  parser.add_argument('--split_validation', default=10, type=int, help="split_validation")
  parser.add_argument('--split_test', default=10, type=int, help="split_test")
  parser.add_argument('--split_seed', default=42, type=int, help="split_seed")
  args = parser.parse_args()
  main(args)
