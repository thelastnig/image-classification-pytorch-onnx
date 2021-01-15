import argparse
import os

import torch

from src.models import *
from src.pachy_dataset import PachyClassificationDataset
from src.split import SplitDataset, CrossValidation
from src.trainer import ImageClassificationTrainer


def main(args):
  os.environ["GIT_PYTHON_REFRESH"] = "quiet"
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Running on {device}")

  if args.model_name in globals():
    model_cls = eval(args.model_name)
  else:
    raise ValueError(f"{args.model_name} is not a registered model name")

  dataset = PachyClassificationDataset(f'{args.dataset_name}/master', '/data/images/test/')
  model = model_cls(num_classes=dataset.num_classes)
  hyper_dict = {
    'epochs': args.epoch,
    'batch_size': args.batch_size,
    'num_workers': 2,
    'optimizer': 'adam',
    'learning_rate': args.lr,
    'weight_decay': args.weight_decay,
  }

  split_total = args.split_training + args.split_validation + args.split_test
  train_val_barrier = args.split_training / split_total
  val_test_barrier = (args.split_training + args.split_validation) / split_total
  test_dataset = SplitDataset(dataset, (val_test_barrier, 1.), args.split_seed)
  if args.split_type == "T":
    train_dataset = SplitDataset(dataset, (0., train_val_barrier), args.split_seed)
    val_dataset = SplitDataset(dataset, (train_val_barrier, val_test_barrier), args.split_seed)

    trainer = ImageClassificationTrainer(
      train_dataset, val_dataset, test_dataset, model,
      hyper_dict, args.experiment_name, device)
    trainer.train()
  elif args.split_type == "C":
    print(f"Running {args.num_cv_folds}-fold cross validation")
    trainval_dataset = SplitDataset(dataset, (0., val_test_barrier), args.split_seed)
    cv = CrossValidation(trainval_dataset, args.num_cv_folds, args.split_seed)
    for fold_idx, (train_dataset, val_dataset) in enumerate(cv):
      trainer = ImageClassificationTrainer(
        train_dataset, val_dataset, test_dataset, model, hyper_dict,
        f"{args.experiment_name}_{fold_idx + 1}/{args.num_cv_folds}", device)
      trainer.train()
  else:
    raise ValueError(f"Unknown split_type {args.split_type}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Learning parameters
  parser.add_argument('--dataset_name', default="cifar10-raw", type=str, help="dataset_name")
  parser.add_argument('--batch_size', default=32, type=int, help="batch_size")
  parser.add_argument('--epoch', default=2, type=int, help="epoch")
  parser.add_argument('--lr', default=1e-3, type=float, help="lr")
  parser.add_argument('--model_name', default="resnet18", type=str, help="model_name")
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
