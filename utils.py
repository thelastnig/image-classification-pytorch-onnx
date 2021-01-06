import os

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


def get_optimizer(parameters, hyper_dict):
  opt_str_to_cls = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    # TODO Add all available optimizers
  }
  try:
    opt_cls = opt_str_to_cls[hyper_dict['optimizer']]
  except KeyError:
    raise ValueError(f"Optimizer {hyper_dict['optimizer']} is not available")

  return opt_cls(parameters, lr=hyper_dict['learning_rate'],
                 weight_decay=hyper_dict['weight_decay'])


def join_pachy_path(local_root, pachy_path):
  pachy_path = pachy_path[1:] if pachy_path[0] == "/" else pachy_path
  return os.path.join(local_root, pachy_path)
