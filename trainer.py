import torch
from torch.utils.data import DataLoader

from .utils import get_optimizer


class TrainerBase(object):
  def __init__(self,
               train_dataset, test_dataset, preprocessor,
               model, tracker, uploader,
               hyper_dict):
    self.train_dataset = train_dataset
    self.test_dataset = test_dataset
    self.preprocessor = preprocessor

    self.model = model
    self.tracker = tracker
    self.uploader = uploader

    self.epochs = hyper_dict['epochs']
    self.batch_size = hyper_dict['batch_size']
    self.num_workers = hyper_dict['num_workers']

    self.test_len = len(test_dataset)
    self.train_ldr = DataLoader(train_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=True)
    self.test_ldr = DataLoader(test_dataset, batch_size=self.batch_size,
                               num_workers=self.num_workers, shuffle=False)
    self.optimizer = get_optimizer(model.parameters(), hyper_dict)

    # state variables
    self.current_iter = 0

  def train_epoch(self):
    for x, gt in self.train_ldr:
      self.model.zero_grad()
      _, loss = self.model(x, gt)
      self.tracker.set_train_loss(self.current_iter, loss.item())
      loss.backward()
      self.optimizer.step()
      self.current_iter += 1

  def test(self):
    loss_weighted_sum = 0.
    with torch.no_grad():
      for x, gt in self.test_ldr:
        self.model.zero_grad()
        _, loss = self.model(x, gt)
        loss_weighted_sum += loss.item() * x.shape[0]
    self.tracker.set_test_loss(self.current_iter, loss_weighted_sum / self.test_len)

  def predict(self, x):
    x = self.preprocessor(x)
    with torch.no_grad():
      pred, _ = self.model(x)
    return pred

  def train(self):
    self.current_iter = 0
    for _ in self.epochs:
      self.train_epoch()
      self.test()
