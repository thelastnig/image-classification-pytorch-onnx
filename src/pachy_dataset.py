import json

import python_pachyderm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as tf
from tqdm import tqdm
import os

from src.utils import *


class ToLongTensor:
  def __init__(self):
    pass

  def __call__(self, img):
    y = np.array(img)
    return torch.as_tensor(y, dtype=torch.long).squeeze(0)


class PachySemanticDataset(Dataset):
  """
  Semantic Segmentation Dataset for Pachyderm
  """

  def __init__(self, commit, path_prefix="/",
               pachy_host=os.environ['PACHYDERM_HOST_URI'], 
               pachy_port="30650",
               local_root='/data',
               transform=T.Compose([T.Resize((512, 512)), T.ToTensor()]),
               target_transform=T.Compose([T.Resize((512, 512)), ToLongTensor()])):
    self.commit = commit
    self.path_prefix = path_prefix
    self.client = python_pachyderm.Client(host=pachy_host, port=pachy_port)
    self.image_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                           for res in self.client.glob_file(commit, path_prefix + "images/*")]
    self.label_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                           for res in self.client.glob_file(commit, path_prefix + "labels/*")]
    self.anno_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                          for res in self.client.glob_file(commit, path_prefix + "annotations/*")]
    self.meta_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                          for res in self.client.glob_file(commit, path_prefix + "meta.json")]
    self.transform = transform
    self.target_transform = target_transform

    self.local_root = local_root
    self._download_data_from_pachyderm(self.image_path_lst, self.path_prefix + "images/*")
    self._download_data_from_pachyderm(self.label_path_lst, self.path_prefix + "labels/*")
    self._download_data_from_pachyderm(self.anno_path_lst, self.path_prefix + "annotations/*")
    self._download_data_from_pachyderm(self.meta_path_lst, self.path_prefix + "meta.json")

    with open(join_pachy_path(self.local_root, "meta.json")) as meta_f:
      meta = json.load(meta_f)
    self.class_names = meta["class_names"]
    self.num_classes = len(self.class_names)

  def _download_data_from_pachyderm(self, path_lst, glob):
    print("Downloading data into worker")
    idx = 0
    continued = False
    current_size = 0
    for chunk in self.client.get_file(self.commit, glob):
      local_path = join_pachy_path(self.local_root, path_lst[idx]['path'][len(self.path_prefix):])
      if not continued:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
      with open(local_path, "ab" if continued else "wb") as local_file:
        local_file.write(chunk)
        current_size += len(chunk)
      if current_size == path_lst[idx]["size"]:
        idx += 1
        continued = False
        current_size = 0
      elif current_size < path_lst[idx]["size"]:
        continued = True
      else:
        raise IOError("Wrong chunk size")
    print(f"Downloaded {idx} files")

  def __len__(self):
    return len(self.anno_path_lst)

  def __getitem__(self, idx):
    if isinstance(idx, torch.Tensor):
      idx = idx.tolist()
    if isinstance(idx, list):
      return [self[i] for i in idx]

    with open(join_pachy_path(self.local_root, self.anno_path_lst[idx]['path'])) as anno_f:
      anno = json.load(anno_f)
    image_file_name = anno["image"]["file_name"]
    label_file_name = anno["semantic"]

    return (
      self.transform(Image.open(os.path.join(self.local_root, "images", image_file_name))),
      self.target_transform(Image.open(os.path.join(self.local_root, "labels", label_file_name)))
    )
