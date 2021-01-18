import math
from typing import Tuple, List

import numpy as np
import torch


class SplitDataset(torch.utils.data.Dataset):
  """
  Dynamically split given Dataset
  Designed to share Dataset across SplitDataset instances
  """

  def __init__(self,
               dataset: torch.utils.data.Dataset,
               split_interval: Tuple[float, float],
               seed: int = 0):
    """
    :param dataset: Reference Dataset
    :param split_interval: Sample range [x, y) where 0 <= x < y <= 1
    :param seed: Random seed for shuffling the indices
    """
    self.full_dataset = dataset
    cnt = len(dataset)

    np.random.seed(seed)

    lower_bound = math.ceil(split_interval[0] * cnt)
    upper_bound = math.ceil(split_interval[1] * cnt)

    idx = np.arange(0, cnt)
    np.random.shuffle(idx)
    self.idx_in_full_dataset = idx[lower_bound:upper_bound].tolist()

  def __len__(self):
    return len(self.idx_in_full_dataset)

  def __getitem__(self, idx: int):
    return self.full_dataset[self.idx_in_full_dataset[idx]]


class MergedDataset(torch.utils.data.Dataset):
  """
  Merge given list of Datasets
  """

  def __init__(self,
               dataset_lst: List[torch.utils.data.Dataset]):
    self.dataset_lst = dataset_lst
    self.dataset_cnt = len(dataset_lst)
    self.cnt_lst = [len(dataset) for dataset in dataset_lst]
    self.sum_lst = [0]
    for cnt in self.cnt_lst:
      self.sum_lst.append(self.sum_lst[-1] + cnt)

  def __len__(self):
    return self.sum_lst[-1]

  def __getitem__(self, idx: int):
    for dataset_idx in range(self.dataset_cnt):
      if self.sum_lst[dataset_idx + 1] > idx:
        return self.dataset_lst[dataset_idx][idx - self.sum_lst[dataset_idx]]


class CrossValidationDataset(object):
  """
  Dynamically split given Dataset into folds
  and iterate over folds
  """

  def __init__(self,
               dataset: torch.utils.data.Dataset,
               num_folds: int,
               seed: int = 0):
    """
    :param dataset: Reference Dataset
    :param num_folds: Number of folds. Usually known as 'k'
    :param seed: Random seed for shuffling the indices
    """
    self.full_dataset = dataset
    self.num_folds = num_folds
    self.iter = 0

    self.folds = []
    for idx in range(num_folds):
      self.folds.append(SplitDataset(dataset, (idx / num_folds, (idx + 1) / num_folds), seed))

  def __len__(self):
    return self.num_folds

  def __iter__(self):
    self.iter = 0
    return self

  def __next__(self):
    if self.iter == self.num_folds:
      raise StopIteration()
    val_idx = (self.iter - 1) % self.num_folds
    train_dataset = MergedDataset(self.folds[:val_idx] + self.folds[val_idx + 1:])
    val_dataset = self.folds[val_idx]
    self.iter += 1
    return train_dataset, val_dataset
