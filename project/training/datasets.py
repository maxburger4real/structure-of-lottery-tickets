"""This file contains everything to create datasets."""

import torch
import torchvision
import numpy as np
from enum import Enum
from sklearn import datasets
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import torch_utils

circles_inputs = moons_inputs = 2
circles_outputs = moons_outputs = 1
mnist_inputs, mnist_outputs = 784, 10  # 28x28

# visible
class Datasets(Enum):
    '''Datasets.Enum
    .name : used to match with the configuration. 
    .value : used as a task description for multi task settings.
    '''
    CIRCLES_MOONS = OrderedDict(circles=(2, 1), moons=(2,1))
    MNIST = OrderedDict(mnist=(784, 10))
    # TODO:
    MINI_MNIST = OrderedDict(mnist=(196, 10))  
    MINI_FASHION = OrderedDict(mnist=(196, 10))
    MINI_MNIST_FASHION = OrderedDict(mnist=(392, 20))
    # TODO FINAL:
    FASHION = OrderedDict(mnist=(784, 10))
    MNIST_FASHION = OrderedDict(mnist=(1568, 20))

class Scalers(Enum):
    MinMaxZeroMean = 'min-max-zero-mean'
    MinMaxZeroOne = 'min-max-zero-one'
    StandardUnitVariance = 'std-unit-variance'

def make_dataloaders(x_train, y_train, x_test, y_test, batch_size):

    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    return train_dataloader, test_dataloader

def make_dataset(name, n_samples, noise, seed, factor, scaler):
    '''create the correct dataset, based on name.'''
    match Datasets[name]:
        case Datasets.CIRCLES_MOONS:
            return __make_circles_and_moons(n_samples, noise, seed, factor, scaler)
        case Datasets.MNIST:
            return __make_mnist(scaler)
        
        case _:
            raise ValueError(f'Unknown dataset {name}')
    
def __make_circles_and_moons(n_samples, noise, seed, factor, scaler):
    '''The concatenated circles and moons dataset.'''

    torch_utils.set_seed(seed)

    circles = datasets.make_circles(int(n_samples*2), noise=noise, factor=factor)
    moons = datasets.make_moons(int(n_samples*2), noise=noise)
    x, y = __concat_datasets([circles, moons])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    x_train, x_test = __scale_dataset(x_train, x_test, scaler)

    return x_train, y_train, x_test, y_test

def __make_mnist(scaler):
    '''The classic MNIST Dataset from torchvision.'''

    train = torchvision.datasets.MNIST(
        root="mnist_data",
        train=True,
        download=True,
    )

    x_train = torch.flatten(train.data, start_dim=1)

    test = torchvision.datasets.MNIST(
        root="mnist_data",
        train=False,
        download=True,
    )

    x_test = torch.flatten(test.data, start_dim=1)
    x_train, x_test = __scale_dataset(x_train, x_test, scaler)

    return x_train, train.targets, x_test, test.targets

# helpers
def __concat_datasets(list_of_datasets):
    list_of_x, list_of_y = list(zip(*list_of_datasets))
    x = torch.Tensor(np.concatenate(list_of_x, axis=1))

    list_of_y_unsqueezed = [y.reshape(-1,1) for y in list_of_y]
    y = torch.Tensor(np.concatenate(list_of_y_unsqueezed, axis=1))

    return x, y

def __scale_dataset(x_train, x_test, scaler):
    """Scale the dataset featurewise."""
    if scaler is None:
        return x_train, x_test
    
    match Scalers[scaler]:
        case Scalers.MinMaxZeroMean:
            scaler = MinMaxScaler(feature_range=(-1,1))
        case Scalers.MinMaxZeroOne:
            scaler = MinMaxScaler(feature_range=(0,1))
        case Scalers.StandardUnitVariance:
            scaler = StandardScaler()
        case _:
            raise ValueError('Unknown Scaler')

    scaler = scaler.fit(x_train)
    x_train = torch.from_numpy(scaler.transform(x_train)).float()
    x_test = torch.from_numpy(scaler.transform(x_test)).float()

    return x_train, x_test
