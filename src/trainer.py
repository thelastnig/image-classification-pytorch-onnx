import os
import time
import copy

import mlflow
import torch
from torch.utils.data import DataLoader

from src.utils import get_optimizer, AverageMeter, accuracy


class ImageClassificationTrainer(object):
  def __init__(self,
               train_dataset, val_dataset, test_dataset,
               model, hyper_dict, experiment_name,
               device, cross_validation=False):
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.test_dataset = test_dataset

    self.model = model
    self.best_model = copy.deepcopy(model)
    self.best_val_loss = None

    self.epochs = hyper_dict['epochs']
    self.batch_size = hyper_dict['batch_size']
    self.num_workers = hyper_dict['num_workers']
    self.hyper_dict = hyper_dict

    self.experiment_name = experiment_name
    self.device = device
    self.cross_validation = cross_validation

    key_lst = []
    for split in ('train', 'val', 'test'):
      for metric in ('loss', 'acc', 'time'):
        key_lst.append(f"{split}_{metric}")

    self.avg_meter = {key: AverageMeter() for key in key_lst}
    self.tag_str = {key: "" for key in key_lst}

    self.train_ldr = DataLoader(train_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=True)
    self.val_ldr = DataLoader(val_dataset, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False)
    self.test_ldr = DataLoader(test_dataset, batch_size=self.batch_size,
                               num_workers=self.num_workers, shuffle=False)
    self.optimizer = get_optimizer(model.parameters(), hyper_dict)

    # state variables
    self.current_iter = 0

  def train_epoch(self):
    for x, gt in self.train_ldr:
      start_time = time.time()
      self.model.zero_grad()
      gt = gt.to(self.device)
      pred, loss = self.model(x.to(self.device), gt)
      acc = accuracy(pred, gt.argmax(dim=1), 1)
      self.avg_meter['train_loss'].update(loss.item(), x.shape[0])
      self.avg_meter['train_acc'].update(acc, x.shape[0])
      loss.backward()
      self.optimizer.step()
      end_time = time.time()
      self.avg_meter['train_time'].update(end_time - start_time)
      self.current_iter += 1

  def test(self, loader, split_str="test"):
    with torch.no_grad():
      for x, gt in loader:
        start_time = time.time()
        self.model.zero_grad()
        gt = gt.to(self.device)
        pred, loss = self.model(x.to(self.device), gt)
        acc = accuracy(pred, gt.argmax(dim=1), 1)
        self.avg_meter[f'{split_str}_loss'].update(loss.item())
        self.avg_meter[f'{split_str}_acc'].update(acc)
        end_time = time.time()
        self.avg_meter[f'{split_str}_time'].update(end_time - start_time)

  def save_checkpoint(self, epoch):
    filename = f"checkpoint.{epoch}.pth"
    torch.save({
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict(),
    }, filename)
    return filename

  def before_train(self):
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    filename = self.save_checkpoint(0)
    model_size = round(os.path.getsize(filename) / 1e6)
    mlflow.set_tag("model_size", model_size)

    for hyper_key, hyper_value in self.hyper_dict.items():
      mlflow.log_param(hyper_key, hyper_value)

    self.model = self.model.to(self.device)

  def before_epoch(self, epoch):
    for avg_meter in self.avg_meter.values():
      avg_meter.reset()

  def after_epoch(self, epoch):
    for key in self.avg_meter:
      self.tag_str[key] += f"{round(self.avg_meter[key].avg, 3)} "
      mlflow.set_tag(f"{self.experiment_name}_{key}", self.tag_str[key])

    for split in ('train', 'val', 'test'):
      for metric in ('loss', 'acc'):
        key = f"{split}_{metric}"
        mlflow.log_metric(key=key, value=self.avg_meter[key].avg, step=epoch)

    if self.best_val_loss is None or self.best_val_loss > self.avg_meter['val_loss'].avg:
      self.best_val_loss = self.avg_meter['val_loss'].avg
      self.best_model = copy.deepcopy(self.model)
      self.test_loss_at_best_val = self.avg_meter['test_loss'].avg

  def after_train(self):
    if not self.cross_validation:
      mlflow.log_artifacts('runs', artifact_path="tensorboard")
      mlflow.pytorch.log_model(self.best_model, artifact_path="pytorch-model")
    print('Finished Training')

  def train(self):
    self.before_train()
    self.current_iter = 0

    for epoch in range(self.epochs):
      self.before_epoch(epoch)
      self.train_epoch()
      self.test(self.val_ldr, "val")
      self.test(self.test_ldr, "test")
      self.after_epoch(epoch)

    self.after_train()
