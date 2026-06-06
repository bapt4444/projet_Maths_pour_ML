import numpy as np


def one_hot_encode(labels, num_classes=10):
    """
    Encodage one-hot vectorise.
    labels : array de forme (n,) ou (n, 1)
    """
    labels = np.array(labels, dtype=int).reshape(-1)
    n = labels.shape[0]
    Y = np.zeros((n, num_classes), dtype=np.float32)
    Y[np.arange(n), labels] = 1.0
    return Y
