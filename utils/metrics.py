import numpy as np

def accuracy(y_pred, y_true):
    return np.mean(np.round(y_pred) == y_true)

def precision(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-7)  