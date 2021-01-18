import numpy as np
from sklearn.metrics import jaccard_similarity_score


def calculate_iou(pred, gt):
  return jaccard_similarity_score(pred.detach().cpu().numpy().reshape(-1),
                                  gt.detach().cpu().numpy().reshape(-1))
