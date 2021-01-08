import argparse

import torch

from src.resnet import *
from src.pachy_dataset import PachyClassificationDataset
from src.trainer import TrainerBase


def main(args):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Running on {device}")
  train_dataset = PachyClassificationDataset('cifar10-raw/master', '/data/images/train/')
  test_dataset = PachyClassificationDataset('cifar10-raw/master', '/data/images/test/')
  model = resnet18(num_classes=train_dataset.num_classes)

  trainer = TrainerBase(train_dataset, test_dataset, None,
                        model, {
                          'epochs': 20,
                          'batch_size': 32,
                          'num_workers': 2,
                          'optimizer': 'adam',
                          'learning_rate': 1e-3,
                          'weight_decay': 2e-5,
                        }, args.experiment_name)
  trainer.train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Learning parameters
  parser.add_argument('--dataset_name', default="jybtest", type=str, help="dataset_name")
  parser.add_argument('--batch_size', default=512, type=int, help="batch_size")
  parser.add_argument('--epoch', default=90, type=int, help="epoch")
  parser.add_argument('--lr', default=1e-3, type=float, help="lr")
  parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
  parser.add_argument('--weight_decay', default=5e-4, type=float, help="weight_decay")
  parser.add_argument('--experiment_name', default="any_exp", type=str, help="exp_name")
  parser.add_argument('--split_type', default="T", type=str, help="split_type")
  parser.add_argument('--split_training', default=80, type=int, help="split_training")
  parser.add_argument('--split_validation', default=10, type=int, help="split_validation")
  parser.add_argument('--split_test', default=10, type=int, help="split_test")
  parser.add_argument('--split_seed', default=42, type=int, help="split_seed")
  args = parser.parse_args()
  main(args)
