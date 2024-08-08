import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from torch.utils.data import Dataset


def get_U(K=2, d=2, n=16):
    """ K: num of Gaussian components
        d: rank of U
        n: dim of x
    """
    A = np.random.rand(n, n)
    Q, R = np.linalg.qr(A)

    U_dt = {}
    for i in range(K):
        U_dt[i] = Q[:, d*i: d*(i+1)]
    return U_dt


class MoGData(Dataset):
    def __init__(self, K=2, d=2, n=16, N=20, alpha=1, U_dt=None):
        """alpha: std for the noise added to x"""
        if U_dt is None:
            self.U_dt = get_U(K=K, d=d, n=n)
        else:
            self.U_dt = U_dt
        self.inputs, self.labels = self.generate_data(K=K, d=d, n=n, N=N, alpha=alpha)
        self.labels = self.labels.reshape(-1, 1)
        self.alpha = alpha

    def generate_data(self, K=2, d=2, n=16, N=20, alpha=1):
        y = np.tile(np.arange(K), N//K)
        inputs = []
        for i, label in enumerate(y):
            a, e = np.random.randn(d), np.random.randn(n)
            input = self.U_dt[label] @ a + e * alpha
            inputs.append(input)
        inputs = np.concatenate([x.reshape(1, -1) for x in inputs], axis=0)
        return inputs, y

    def __len__(self):
        return len(self.inputs)  # Return the total number of samples

    def __getitem__(self, idx):
        input = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input, label