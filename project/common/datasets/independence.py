import sys
sys.path.append('../')

import numpy as np
import torch

from common import torch_utils

from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 4
OUTPUT_DIM = 2
DATASET_NAME = "Independence_Symbolic"

_N = 100
_seed = 0



def f1(x1, x3):
    return (x1+x3)**3

def f2(x2, x4):
    return x2**2+np.sin(np.pi*x4)

def f(data):
    """
    The function that transforms input to output.
    from BIMT code
    """
    x1 = data[:,[0]]
    x2 = data[:,[1]]
    x3 = data[:,[2]]
    x4 = data[:,[3]]

    y1 = f1(x1,x3)
    y2 = f2(x2, x4)

    out = np.array([y1, y2]).T
    return out.squeeze()

def get_train_and_test_data():
    """from BIMT code"""
    torch_utils.set_seed(_seed)
    inputs = np.random.rand(_N, INPUT_DIM)*2-1
    labels = f(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float, requires_grad=True)
    labels = torch.tensor(labels, dtype=torch.float, requires_grad=True)

    inputs_test = np.random.rand(_N, INPUT_DIM)*2-1
    labels_test = f(inputs_test)
    inputs_test = torch.tensor(inputs_test, dtype=torch.float, requires_grad=True)
    labels_test = torch.tensor(labels_test, dtype=torch.float, requires_grad=True)

    return inputs, labels, inputs_test, labels_test

def build_loaders(batch_size=None):
    """
        Return Dataloaders with specific Batch size. 
        if ommited, batch size is the complete dataset.
    """
    if batch_size is None: batch_size = _N
    x,y,xx,yy = get_train_and_test_data()

    train_dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(TensorDataset(xx, yy), batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader
