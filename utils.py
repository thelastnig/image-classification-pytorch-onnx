import torch


def get_class_label_from_path(path, path_prefix):
  split_path = path.replace(path_prefix, "").split("/")
  if split_path[0]:  # leading / is removed
    return split_path[0]
  return split_path[1]


def index_to_one_hot(index, num_classes, dtype=torch.float32):
  ret = torch.zeros(num_classes, dtype=dtype)
  ret[index] = 1
  return ret
