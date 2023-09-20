import sys
sys.path.append('../')

import numpy as np
import torch

from common import torch_utils

from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 4
OUTPUT_DIM = 2

_N = 100
_seed = 0

def _f(data):
    """
    The function that transforms input to output.
    from BIMT code
    """
    x1 = data[:,[0]]
    x2 = data[:,[1]]
    x3 = data[:,[2]]
    x4 = data[:,[3]]
    out = np.transpose(np.array([(x1+x3)**3, x2**2+np.sin(np.pi*x4)]))
    return out.squeeze()

def get_train_and_test_data():
    """from BIMT code"""
    torch_utils.set_seed(_seed)
    inputs = np.random.rand(_N, INPUT_DIM)*2-1
    labels = _f(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float, requires_grad=True)
    labels = torch.tensor(labels, dtype=torch.float, requires_grad=True)

    inputs_test = np.random.rand(_N, INPUT_DIM)*2-1
    labels_test = _f(inputs_test)
    inputs_test = torch.tensor(inputs_test, dtype=torch.float, requires_grad=True)
    labels_test = torch.tensor(labels_test, dtype=torch.float, requires_grad=True)

    return inputs, labels, inputs_test, labels_test

def get_dataloaders(batch_size=None):
    """Return Dataloaders with specific Batch size. if ommited, batch size is the complete dataset."""
    if batch_size is None: batch_size = _N
    x,y,xx,yy = get_train_and_test_data()

    train_dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(TensorDataset(xx, yy), batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader
