import os
import time
import copy
from tqdm import tqdm

import mlflow
import torch
from torch.utils.data import DataLoader
from datetime import datetime, timezone

from src.utils import get_optimizer, AverageMeter, accuracy

import signal
import sys


class LockableHandler(object):
  def __init__(self, trainer):
    self.locked = False
    self.received_signal = False
    self.trainer = trainer

    signal.signal(signal.SIGTERM, self.on_sigterm)

  def handle(self):
    pass

  def lock(self):
    if self.locked:
        raise ValueError("lock() called on locked object")
    self.locked = True

  def unlock(self):
    if not self.locked:
        raise ValueError("unlock() called on unlocked object")
    self.locked = False
    if self.received_signal:
        self.handle()

  def on_sigterm(self, signum, frame):
    self.received_signal = True
    if not self.locked:
        self.handle()


class LockableModelSaveHandler(LockableHandler):
  def handle(self):
    os.makedirs('data', exist_ok=True)
    try:
      torch.save({"model": self.trainer.best_model}, "data/model.pth")
    except:
      torch.save({"weights": self.trainer.best_model.state_dict()}, "data/model.pth")
      mlflow.log_artifact("src/models/", artifact_path="architecture")
    mlflow.log_artifact("data/", artifact_path="model")
    sys.exit(0)


class ImageClassificationTrainer(object):
  def __init__(self,
               train_dataset, val_dataset, test_dataset,
               model, hyper_dict, experiment_name,
               device, cross_validation=False):
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.test_dataset = test_dataset

    self.handler = LockableModelSaveHandler(self)

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

    key_lst = ['time']
    for split in ('train', 'val', 'test'):
      for metric in ('loss', 'acc'):
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
    for x, gt in tqdm(self.train_ldr, desc="train"):
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
      self.avg_meter['time'].update(end_time - start_time)
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
        # self.avg_meter[f'{split_str}_time'].update(end_time - start_time)


  def save_checkpoint(self, epoch):
    filename = f"checkpoint.{epoch}.pth"
    torch.save({
      "model": self.model.state_dict()
    }, filename)
    return filename

  def before_train(self):
    print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    filename = self.save_checkpoint(0)
    model_size = round(os.path.getsize(filename) / 1e6)
    mlflow.set_tag("model_size", model_size)

    mlflow.set_tag("experiment_name", f"{self.experiment_name}")
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

    local_time = datetime.now(timezone.utc).astimezone().isoformat()
    for metric in ('loss', 'acc'):
      for split in ('train', 'val', 'test'):
        key = f"{split}_{metric}"
        mlflow.log_metric(key=key, value=self.avg_meter[key].avg, step=epoch)
        hpo_metric = 'accuracy' if metric == 'acc' else metric
        if split == 'train':
          print(f"{local_time} {hpo_metric}={round(self.avg_meter[key].avg, 4)}")
        elif split == 'val':
          print(f"{local_time} validation_{hpo_metric}={round(self.avg_meter[key].avg, 4)}")

    if self.best_val_loss is None or self.best_val_loss > self.avg_meter['val_loss'].avg:
      self.best_val_loss = self.avg_meter['val_loss'].avg
      self.best_model = copy.deepcopy(self.model)
      self.test_loss_at_best_val = self.avg_meter['test_loss'].avg

  def after_train(self):
    self.handler.lock()
    if not self.cross_validation:
      # mlflow.pytorch.log_model(self.best_model, artifact_path="model")
      os.makedirs('data', exist_ok=True)
      try:
        torch.save({"model": self.best_model}, "data/model.pth")
      except:
        torch.save({"weights": self.best_model.state_dict()}, "data/model.pth")
      mlflow.log_artifact("src/models/", artifact_path="architecture")
      mlflow.log_artifact("data/", artifact_path="model")
    print('Finished Training')
    self.handler.unlock()

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


class ImageClassificationCVWrapper(object):
  def __init__(self, cv_dataset, test_dataset, build_model,
               hyper_dict, experiment_name, device):
    self.cv_dataset = cv_dataset
    self.num_folds = len(cv_dataset)
    self.fold_idx = 0
    self.train_dataset = self.val_dataset = None
    self.test_dataset = test_dataset
    self.build_model = build_model
    self.trainer = None
    self.hyper_dict = hyper_dict
    self.experiment_name = experiment_name
    self.device = device

    self.best_model = None
    self.test_loss_at_best_val = -1

  def before_train(self):
    print(f"Running {self.num_folds}-fold cross validation")

  def before_fold(self):
    model = self.build_model()
    self.trainer = ImageClassificationTrainer(
      self.train_dataset, self.val_dataset, self.test_dataset, model, self.hyper_dict,
      f"{self.experiment_name}_{self.fold_idx + 1}/{self.num_folds}", self.device, True)

  def train_fold(self):
    self.trainer.train()

  def after_fold(self):
    if self.best_model is None or self.test_loss_at_best_val > self.trainer.test_loss_at_best_val:
      self.test_loss_at_best_val = self.trainer.test_loss_at_best_val
      self.best_model = copy.deepcopy(self.trainer.best_model)

  def after_train(self):
    mlflow.pytorch.log_model(self.best_model, artifact_path="model")

  def train(self):
    self.before_train()
    for fold_idx, (train_dataset, val_dataset) in enumerate(self.cv_dataset):
      self.fold_idx = fold_idx
      self.train_dataset = train_dataset
      self.val_dataset = val_dataset

      self.before_fold()
      self.train_fold()
      self.after_fold()

    self.after_train()
