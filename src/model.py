from torch import nn


class ClassificationModel(nn.Module):
  def __init__(self, num_classes=10):
    super(ClassificationModel, self).__init__()

    # network definition
    self.conv = nn.Sequential(
      nn.Conv2d(3, 16, 3, 1, 1),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.Conv2d(16, 32, 3, 2, 1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, 3, 2, 1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
    )
    self.fc = nn.Linear(64, num_classes)

    # criterion for classification
    self.crit = nn.CrossEntropyLoss()

  def forward(self, x, y):
    """
    :param x: image input. Expected to have NCHW shape
    :param y: ground truth. Expected to have N x num_classes shape
    :return: tuple of (prediction, loss)
    """

    out = self.conv(x)
    out = out.flatten(2).mean(dim=2)
    pred = self.fc(out)

    loss = self.crit(pred, y)

    return pred, loss
