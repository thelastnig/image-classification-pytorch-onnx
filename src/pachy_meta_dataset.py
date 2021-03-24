import json
import python_pachyderm
import os
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.utils import *


class PachyClassificationMetaDataset(Dataset):
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

        self.meta_path_lst = [{'path': res.file.path, 'size': res.size_bytes}
                              for res in self.client.glob_file(commit, path_prefix + "meta.json")]

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
