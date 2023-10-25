import numpy as np
import torch
from sklearn.datasets import make_moons
from common import torch_utils
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 2
OUTPUT_DIM = 1
DATASET_NAME = "Two Moons"

_N = 400
_seed = 0
noise=0.1

def get_train_and_test_data(m):
    torch_utils.set_seed(_seed)

    m_moon_sets = [make_moons(n_samples=_N, noise=noise) for _ in range(m)]
    X,Y = list(zip(*m_moon_sets))
    Y = [y.reshape(-1,1) for y in Y]
    x_train = torch.Tensor(np.concatenate(X, axis=1))
    y_train = torch.Tensor(np.concatenate(Y, axis=1))

    m_moon_sets = [make_moons(n_samples=_N, noise=noise) for _ in range(m)]
    X,Y = list(zip(*m_moon_sets))
    Y = [y.reshape(-1,1) for y in Y]
    x_test = torch.Tensor(np.concatenate(X, axis=1))
    y_test = torch.Tensor(np.concatenate(Y, axis=1))

    return x_train, y_train, x_test, y_test

def build_loaders(m=1, batch_size=None):
    """
        Return Dataloaders with specific Batch size. 
        if ommited, batch size is the complete dataset.
    """
    if batch_size is None: batch_size = _N
    x,y, xx,yy = get_train_and_test_data(m)

    train_dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(TensorDataset(xx, yy), batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader
