import os
import shutil

import python_pachyderm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .utils import *


class PachyClassificationDataset(Dataset):
  """
  Image Classification Dataset for Pachyderm
  Assumes class-wise folder structure as flow_from_directory
  """

  def __init__(self, commit, path_prefix="/",
               pachy_host="175.197.4.150", pachy_port="30650",
               local_root='/data', transform=ToTensor()):
    self.commit = commit
    self.path_prefix = path_prefix
    self.client = python_pachyderm.Client(host=pachy_host, port=pachy_port)
    self.path_lst = [res.file.path for res in self.client.glob_file(commit, path_prefix + "*/*")]
    self.class_labels = list(set(get_class_label_from_path(path, path_prefix)
                                 for path in self.path_lst))
    self.num_classes = len(self.class_labels)
    self.transform = transform

    self.local_root = local_root
    self._download_data_from_pachyderm()

  def _download_data_from_pachyderm(self):
    for path in self.path_lst:
      local_path = join_pachy_path(self.local_root, path)
      os.makedirs(os.path.dirname(local_path), exist_ok=True)
      pfs_file = self.client.get_file(self.commit, path)
      with open(local_path, 'w') as local_file:
        shutil.copyfileobj(pfs_file, local_file)

  def __len__(self):
    return len(self.path_lst)

  def __getitem__(self, idx):
    if isinstance(idx, torch.Tensor):
      idx = idx.tolist()
    if isinstance(idx, list):
      return [self[i] for i in idx]

    path = self.path_lst[idx]
    return (
      self.transform(Image.open(join_pachy_path(self.local_root, path))),
      index_to_one_hot(
        self.class_labels.index(
          get_class_label_from_path(path, self.path_prefix)),
        self.num_classes
      )
    )
