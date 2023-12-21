"""This file contains everything to create datasets."""
import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from common.config import Config
from common.constants import *
from common import torch_utils

circles_inputs = moons_inputs = 2
circles_outputs = moons_outputs = 1

# visible
def build_dataloaders_from_config(config: Config):
    
    n_samples = config.n_samples
    noise = config.noise

    if config.dataset == OLD_MOONS:
        
        config.update({
            'task_description' : (
                ('moons-1', (moons_inputs, moons_outputs)),
                ('moons-2', (moons_inputs, moons_outputs)),
            )
        }, allow_val_change=True)

        torch_utils.set_seed(config.data_seed)

        m_moon_sets = [datasets.make_moons(n_samples=n_samples, noise=noise) for _ in range(2)]
        X,Y = list(zip(*m_moon_sets))
        Y = [y.reshape(-1,1) for y in Y]
        x_train = torch.Tensor(np.concatenate(X, axis=1))
        y_train = torch.Tensor(np.concatenate(Y, axis=1))

        m_moon_sets = [datasets.make_moons(n_samples=n_samples, noise=noise) for _ in range(2)]

        X,Y = list(zip(*m_moon_sets))
        x_test = torch.Tensor(np.concatenate(X, axis=1))

        Y = [y.reshape(-1,1) for y in Y]
        y_test = torch.Tensor(np.concatenate(Y, axis=1))

        if config.batch_size is None: batch_size = n_samples
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True, num_workers=0)

        return train_dataloader, test_dataloader
    
    if config.dataset == MOONS_AND_CIRCLES:
        config.update({
            'task_description' : (
                ('moons' , (moons_inputs, moons_outputs)),
                ('circles', (circles_inputs, circles_outputs))
            )
        }, allow_val_change=True)
        return __build_moons_and_circles(n_samples=n_samples, noise=noise, batch_size=config.batch_size)

    if config.dataset == MULTI_MOONS:
        config.update({
            'task_description' : (
                ('moons-1', (moons_inputs, moons_outputs)),
                ('moons-2', (moons_inputs, moons_outputs)),
            )
        }, allow_val_change=True)
        torch_utils.set_seed(config.data_seed)
        return __build_moons_and_moons(n_samples=n_samples, noise=noise, batch_size=config.batch_size)
    
    if config.dataset == MULTI_CIRCLES:
        raise NotImplementedError('didnt do it yet')
        return __build_moons_and_moons(n_samples=n_samples, noise=noise, batch_size=config.batch_size)
    
    if config.dataset == MOONS:
        return __build_moons(n_samples=n_samples, noise=noise, batch_size=config.batch_size)
    
    if config.dataset == CIRCLES:
        return __build_circles(n_samples=n_samples, noise=noise, batch_size=config.batch_size)


# different datasets

def __build_moons_and_circles(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    train_dataset = __concat_datasets(
        datasets.make_moons(n_samples, noise=noise, random_state=1, shuffle=True),
        datasets.make_circles(n_samples, noise=noise, random_state=0, shuffle=True, factor=0.5),
    )
    test_dataset = __concat_datasets(
        datasets.make_moons(n_samples, noise=noise, random_state=3, shuffle=True),
        datasets.make_circles(n_samples, noise=noise, random_state=2, shuffle=True, factor=0.5),
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

def __build_moons(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    train_dataset = datasets.make_moons(n_samples, noise=noise, random_state=1, shuffle=True)
    test_dataset = datasets.make_moons(n_samples, noise=noise, random_state=2, shuffle=True)

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def __build_circles(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    train_dataset = datasets.make_circles(n_samples, noise=noise, random_state=1, shuffle=True, factor=0.5)
    test_dataset = datasets.make_circles(n_samples, noise=noise, random_state=2, shuffle=True, factor=0.5)

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
