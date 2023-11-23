"""This file contains everything to create datasets."""
import numpy as np
from sklearn import datasets
from torch.utils.data import DataLoader, TensorDataset
from common.config import Config
from common.constants import *


# visible
def build_dataloaders_from_config(config: Config):

    if config.dataset == MOONS_AND_CIRCLES:
        # TODO: remove magic numbers
        return __build_moons_and_circles(n_samples=500, noise=0.05, batch_size=config.batch_size)

    if config.dataset == CONCAT_MOONS:
        return __build_moons_and_moons(n_samples=500, noise=0.05, batch_size=config.batch_size)


# different datasets
def __build_moons_and_circles(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    train_dataset = __concat_datasets(
        datasets.make_circles(n_samples, noise=noise, random_state=0, shuffle=True),
        datasets.make_moons(n_samples, noise=noise, random_state=1, shuffle=True),
    )    
    test_dataset = __concat_datasets(
        datasets.make_circles(n_samples, noise=noise, random_state=2, shuffle=True),
        datasets.make_moons(n_samples, noise=noise, random_state=3, shuffle=True),
    )

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def __build_moons_and_moons(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    train_dataset = __concat_datasets(
        datasets.make_moons(n_samples, noise=noise, random_state=0, shuffle=True),
        datasets.make_moons(n_samples, noise=noise, random_state=1, shuffle=True),
    )    
    test_dataset = __concat_datasets(
        datasets.make_moons(n_samples, noise=noise, random_state=2, shuffle=True),
        datasets.make_moons(n_samples, noise=noise, random_state=3, shuffle=True),
    )

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset, batch_size=batch_size)

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

    return DataLoader(
        TensorDataset(x, y), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
