import python_pachyderm
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import *


class PachyClassificationDataset(Dataset):
  """
  Image Classification Dataset for Pachyderm
  Assumes class-wise folder structure as flow_from_directory
  """
  def __init__(self, commit, path_prefix="/", pachy_host="175.197.4.150", pachy_port="30650"):
    self.commit = commit
    self.path_prefix = path_prefix
    self.client = python_pachyderm.Client(host=pachy_host, port=pachy_port)
    self.path_lst = [res.file.path for res in self.client.glob_file(commit, path_prefix + "*/*")]
    self.class_labels = list(set(get_class_label_from_path(path, path_prefix)
                                 for path in self.path_lst))
    self.num_classes = len(self.class_labels)

  def __len__(self):
    return len(self.file_lst)

  def __getitem__(self, idx):
    if isinstance(idx, torch.Tensor):
      idx = idx.tolist()
    if isinstance(idx, list):
      return [self[i] for i in idx]

    path = self.file_lst[idx]
    pfs_file = self.client.get_file(self.commit, path)
    return (
      Image.open(pfs_file),
      index_to_one_hot(
        self.class_labels.index(
          get_class_label_from_path(path, self.path_prefix)),
        self.num_classes
      )
    )
