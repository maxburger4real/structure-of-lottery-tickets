"""This file contains everything to create datasets."""
import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from common.config import Config
from common.constants import *

circles_inputs = moons_inputs = 2
circles_outputs = moons_outputs = 1

# visible
def build_dataloaders_from_config(config: Config):
    
    n_samples = config.n_samples
    noise = config.noise
    batch_size = config.batch_size

    if config.dataset == Datasets.MOONS_AND_CIRCLES.name:
        config.update({
            'task_description' : (
                ('moons' , (moons_inputs, moons_outputs)),
                ('circles', (circles_inputs, circles_outputs))
            )
        }, allow_val_change=True)
        return __build_moons_and_circles_dl(n_samples=n_samples, noise=noise, batch_size=batch_size)

    if config.dataset == MULTI_MOONS:
        config.update({
            'task_description' : (
                ('moons-1', (moons_inputs, moons_outputs)),
                ('moons-2', (moons_inputs, moons_outputs)),
            )
        }, allow_val_change=True)
        return __build_moons_and_moons_dl(n_samples=n_samples, noise=noise, batch_size=batch_size)
    
    if config.dataset == Datasets.MOONS.name:
        return __build_moons_dl(n_samples=n_samples, noise=noise, batch_size=config.batch_size)
    
    if config.dataset == Datasets.CIRCLES.name:
        return __build_circles_dl(n_samples=n_samples, noise=noise, batch_size=config.batch_size)

# base dataset makers
def __make_moons(n_samples, noise, random_state, shuffle=True, scale=True) -> tuple[np.ndarray, np.ndarray]:
    x, y = datasets.make_moons(n_samples, noise=noise, random_state=random_state, shuffle=shuffle)
    if scale: x = MinMaxScaler().fit_transform(x)
    return x,y

def __make_circles(n_samples, noise, random_state, shuffle=True, scale=True) -> tuple[np.ndarray, np.ndarray]:
    x, y = datasets.make_circles(n_samples, noise=noise, random_state=random_state, shuffle=shuffle, factor=0.35)
    if scale: x = MinMaxScaler().fit_transform(x)
    return x,y

# different datasets
def __build_moons_and_circles_dl(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    n = int(n_samples/2)

    # sample the data
    train_dataset = __concat_datasets(
        __make_moons(n, noise=noise, random_state=0),
        __make_circles(n, noise=noise, random_state=1),
    )
    test_dataset = __concat_datasets(
        __make_moons(50, noise=noise, random_state=2),
        __make_circles(50, noise=noise, random_state=3),
    )

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset)

    return train_loader, test_loader

def __build_moons_and_moons_dl(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    n = int(n_samples/2)

    train_dataset = __concat_datasets(
        __make_moons(n, noise=noise, random_state=0),
        __make_moons(n, noise=noise, random_state=1),
    )    
    test_dataset = __concat_datasets(
        __make_moons(50, noise=noise, random_state=2),
        __make_moons(50, noise=noise, random_state=3),
    )

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset)

    return train_loader, test_loader

def __build_moons_dl(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # TODO: 
    train_dataset = __make_moons(n_samples, noise=noise, random_state=1)
    test_dataset = __make_moons(100, noise=noise, random_state=2)

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset)

    return train_loader, test_loader

def __build_circles_dl(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    train_dataset = __make_circles(n_samples, noise=noise, random_state=1)
    test_dataset = __make_circles(100, noise=noise, random_state=2)

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset)

    return train_loader, test_loader

# helpers
def __concat_datasets(*datasets):
    """
    concatenate datasets with shapes
    X : (N, d)
    Y : (N, )

    returns X : (N, #d) Y : (N, #)
    """
    x,y = list(zip(*datasets))
    X = np.hstack(x)
    Y = np.vstack(y).T

    return X, Y

def __build_dataloader(x: np.ndarray, y: np.ndarray, batch_size=None):

    if batch_size is None: batch_size = x.shape[0]

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    
    if len(y.shape) == 1:
        y = y.reshape(-1,1)

    return DataLoader(
        TensorDataset(x, y), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
