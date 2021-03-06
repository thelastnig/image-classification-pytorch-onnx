import json
import os

import python_pachyderm
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import os

from src.utils import *


class PachyClassificationDataset(Dataset):
  """
  Image Classification Dataset for Pachyderm
  Assumes class-wise folder structure as flow_from_directory
  """

  def __init__(self, commit, path_prefix="/",
               pachy_host=os.environ['PACHYDERM_HOST_URI'],
               pachy_port="30650",
               local_root='/data', 
               transform=T.Compose([T.Resize((256, 256)), 
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
                ):
    self.commit = commit
    self.path_prefix = path_prefix
    self.local_root = local_root

    self.client = python_pachyderm.Client(host=pachy_host, port=pachy_port)

    self.image_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                           for res in self.client.glob_file(commit, path_prefix + "images/*")]
    self.anno_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                          for res in self.client.glob_file(commit, path_prefix + "annotations/*")]
    self.meta_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                          for res in self.client.glob_file(commit, path_prefix + "meta.json")]

    self._download_data_from_pachyderm(self.image_path_lst, self.path_prefix + "images/*")
    self._download_data_from_pachyderm(self.anno_path_lst, self.path_prefix + "annotations/*")
    self._download_data_from_pachyderm(self.meta_path_lst, self.path_prefix + "meta.json")

    with open(os.path.join(self.local_root, "meta.json")) as meta_f:
      meta = json.load(meta_f)
    self.meta = meta
    self.class_names = self.meta["class_names"]
    self.num_classes = len(self.class_names)
    self.transform = transform

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

    with open(join_pachy_path(self.local_root, self.anno_path_lst[idx]["path"]), 'r') as anno_f:
      ann_json = json.load(anno_f)
      try:
        anno = json.loads(ann_json)
      except:
        anno = ann_json
    target = [0] * self.num_classes
    for ist in anno["instances"]:
      target[int(ist['category_id'])] = 1

    img = Image.open(os.path.join(self.local_root, "images", anno["image"]["file_name"])).convert('RGB')

    return (
      self.transform(img),
      torch.tensor(target, dtype=torch.float32)
    )
