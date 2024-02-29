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
    """Datasets.Enum
    .name : used to match with the configuration.
    .value : used as a task description for multi task settings.
    """

    CIRCLES_MOONS = OrderedDict(circles=(2, 1), moons=(2, 1))
    MNIST = OrderedDict(mnist=(784, 10))
    FASHION = OrderedDict(mnist=(784, 10))

    # mini datasets
    MINI_MNIST = OrderedDict(mnist=(144, 3))
    MINI_FASHION = OrderedDict(fashion=(144, 3))
    TINY_FASHION_AND_MNIST = OrderedDict(mnist=(144, 2), fashion=(144, 2))
    TINY_FASHION_AND_MNIST_2 = OrderedDict(mnist=(196, 2), fashion=(196, 2))
    MINI_FASHION_AND_MNIST = OrderedDict(mnist=(196, 3), fashion=(196, 3))
    HALF_FASHION_AND_MNIST = OrderedDict(mnist=(196, 5), fashion=(196, 5))
    RESIZE_FULL_FASHION_AND_MNIST = OrderedDict(mnist=(196, 10), fashion=(196, 10))
    FULL_FASHION_AND_MNIST = OrderedDict(mnist=(784, 10), fashion=(784, 10))
    LARGEST_FASHION_AND_MNIST = OrderedDict(mnist=(784, 10), fashion=(784, 10))

class Scalers(Enum):
    MinMaxZeroMean = "min-max-zero-mean"
    MinMaxZeroOne = "min-max-zero-one"
    StandardUnitVariance = "std-unit-variance"


def make_dataloaders(x_train, y_train, x_test, y_test, batch_size):
    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)

    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(test_set, batch_size=len(test_set))

    return train_dataloader, test_dataloader


def make_dataset(name, n_samples, noise, seed, factor, scaler):
    """create the correct dataset, based on name."""
    match Datasets[name]:
        case Datasets.CIRCLES_MOONS:
            return __make_circles_and_moons(n_samples, noise, seed, factor, scaler)
        case Datasets.MNIST:
            return __make_mnist(scaler)
        case Datasets.MINI_MNIST:
            return __make_mnist(
                scaler, 
                transforms=torchvision.transforms.Resize((12, 12), antialias=True), 
                subset=[0,1,2]
            )
        case Datasets.FASHION:
            return __make_fashion_mnist(scaler)
        case Datasets.MINI_FASHION:
            return __make_fashion_mnist(
                scaler, 
                transforms=torchvision.transforms.Resize((12, 12), antialias=True), 
                subset=(1, 6, 7)
            )
        case Datasets.TINY_FASHION_AND_MNIST:
            return __make_minst_and_fashion_mnist(
                scaler, 
                transforms=torchvision.transforms.Resize((12, 12), antialias=True), 
                mnist_subset=(0,1),
                fashion_subset=(1,6),
                train_size=1000, # per_class
                test_size=200, # per class
            )
        case Datasets.TINY_FASHION_AND_MNIST_2:
            return __make_minst_and_fashion_mnist(
                scaler, 
                transforms=torchvision.transforms.Resize((14, 14), antialias=True), 
                mnist_subset=(0,1),
                fashion_subset=(1,6),
                train_size=2000, # per_class
                test_size=200, # per class
            )
        case Datasets.MINI_FASHION_AND_MNIST:
            return __make_minst_and_fashion_mnist(
                scaler, 
                transforms=torchvision.transforms.Resize((14, 14), antialias=True), 
                mnist_subset=(0,1,2),
                fashion_subset=(1,6,7),
                train_size=3000, # per_class
                test_size=300, # per class
            )
        case Datasets.HALF_FASHION_AND_MNIST:
            return __make_minst_and_fashion_mnist(
                scaler, 
                transforms=torchvision.transforms.Resize((14, 14), antialias=True), 
                mnist_subset=(0,1,2,3,4),
                fashion_subset=(0,1,2,3,4),
                train_size=3000, # per_class
                test_size=300, # per class
            )
        case Datasets.RESIZE_FULL_FASHION_AND_MNIST:
            return __make_minst_and_fashion_mnist(
                scaler, 
                transforms=torchvision.transforms.Resize((14, 14), antialias=True), 
                train_size=2000, # per_class
                test_size=200, # per class
            )        
        case Datasets.FULL_FASHION_AND_MNIST:
            return __make_minst_and_fashion_mnist(
                scaler, 
                train_size=2000, # per_class
                test_size=200, # per class
            )
        case Datasets.LARGEST_FASHION_AND_MNIST:
            return __make_minst_and_fashion_mnist(
                scaler, 
                train_size=5421, # min number of least common mnist digit
                test_size=892,  # same
            )
        case _:
            raise ValueError(f"Unknown dataset {name}")


def __make_circles_and_moons(n_samples, noise, seed, factor, scaler):
    """The concatenated circles and moons dataset."""

    torch_utils.set_seed(seed)

    circles = datasets.make_circles(int(n_samples * 2), noise=noise, factor=factor)
    moons = datasets.make_moons(int(n_samples * 2), noise=noise)
    x, y = __concat_datasets([circles, moons])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=42
    )
    x_train, x_test = __scale_dataset(x_train, x_test, scaler)

    return x_train, y_train, x_test, y_test


def __make_mnist(scaler, transforms=None, subset=None):
    """The classic MNIST Dataset from torchvision."""

    train = torchvision.datasets.MNIST(
        root="mnist_data",
        train=True,
        download=True,
    )
    test = torchvision.datasets.MNIST(
        root="mnist_data",
        train=False,
        download=True,
    )
    if transforms is not None:
        train.data = transforms(train.data)
        test.data = transforms(test.data)

    if subset is not None:
        train_mask = torch.isin(train.targets, torch.tensor(subset))
        train.data = train.data[train_mask]
        train.targets = train.targets[train_mask]

        test_mask = torch.isin(test.targets, torch.tensor(subset))
        test.data = test.data[test_mask]
        test.targets = test.targets[test_mask]

    x_train = torch.flatten(train.data, start_dim=1)
    x_test = torch.flatten(test.data, start_dim=1)
    x_train, x_test = __scale_dataset(x_train, x_test, scaler)

    return x_train, train.targets.to(torch.long), x_test, test.targets.to(torch.long)


def __make_fashion_mnist(scaler, transforms=None, subset=None):
    """The fashion MNIST Dataset from torchvision."""

    train = torchvision.datasets.FashionMNIST(
        root="fashion_mnist_data",
        train=True,
        download=True,
    )
    test = torchvision.datasets.FashionMNIST(
        root="fashion_mnist_data",
        train=False,
        download=True,
    )
    if transforms is not None:
        train.data = transforms(train.data)
        test.data = transforms(test.data)

    if subset is not None:
        train_mask = torch.isin(train.targets, torch.tensor(subset))
        train.data = train.data[train_mask]
        train.targets = train.targets[train_mask]

        test_mask = torch.isin(test.targets, torch.tensor(subset))
        test.data = test.data[test_mask]
        test.targets = test.targets[test_mask]

        # remap targets to [0-n]
        unique_numbers = sorted(set(train.targets.tolist()))
        mapping = {num: i for i, num in enumerate(unique_numbers)}
        train.targets = torch.tensor([mapping[num.item()] for num in train.targets])
        test.targets = torch.tensor([mapping[num.item()] for num in test.targets])

    x_train = torch.flatten(train.data, start_dim=1)
    x_test = torch.flatten(test.data, start_dim=1)
    x_train, x_test = __scale_dataset(x_train, x_test, scaler)

    return x_train, train.targets.to(torch.long), x_test, test.targets.to(torch.long)


def __make_minst_and_fashion_mnist(scaler, train_size: int, test_size: int, transforms=None, fashion_subset=None, mnist_subset=None):
    fx_train, fy_train, fx_test, fy_test = __make_fashion_mnist(
        scaler=None, transforms=transforms, subset=fashion_subset
    )
    fx_train, fy_train = __balanced_subset(fx_train,fy_train,train_size)
    fx_test, fy_test = __balanced_subset(fx_test,fy_test,test_size)

    mx_train, my_train, mx_test, my_test = __make_mnist(
        scaler=None, transforms=transforms, subset=mnist_subset
    )
    mx_train, my_train = __balanced_subset(mx_train,my_train,train_size)
    mx_test, my_test = __balanced_subset(mx_test,my_test,test_size)

    x_train, y_train = __concat_datasets([(mx_train, my_train), (fx_train, fy_train)])
    x_test, y_test = __concat_datasets([(mx_test, my_test), (fx_test, fy_test)])
    x_train, x_test = __scale_dataset(x_train, x_test, scaler)

    return x_train, y_train, x_test, y_test


# helpers
def __balanced_subset(x,y,n):
    if n is None: return x, y

    class_indices = [torch.where(y == i)[0] for i in y.unique()]
    subset_indices = torch.cat([
        indices[torch.randperm(len(indices))[:n]] 
        for indices in class_indices
    ])
    
    # shuffle the indices to mix up the dataset
    subset_indices = subset_indices[torch.randperm(len(subset_indices))]
    x, y = x[subset_indices], y[subset_indices]
    return x,y


def __concat_datasets(list_of_datasets):
    list_of_x, list_of_y = list(zip(*list_of_datasets))
    x = torch.Tensor(np.concatenate(list_of_x, axis=1))

    list_of_y_unsqueezed = [y.reshape(-1, 1) for y in list_of_y]
    y = torch.Tensor(np.concatenate(list_of_y_unsqueezed, axis=1))

    return x, y


def __scale_dataset(x_train, x_test, scaler):
    """Scale the dataset featurewise."""
    if scaler is None:
        return x_train, x_test

    match Scalers[scaler]:
        case Scalers.MinMaxZeroMean:
            scaler = MinMaxScaler(feature_range=(-1, 1))
        case Scalers.MinMaxZeroOne:
            scaler = MinMaxScaler(feature_range=(0, 1))
        case Scalers.StandardUnitVariance:
            scaler = StandardScaler()
        case _:
            raise ValueError("Unknown Scaler")

    scaler = scaler.fit(x_train)
    x_train = torch.from_numpy(scaler.transform(x_train)).float()
    x_test = torch.from_numpy(scaler.transform(x_test)).float()

    return x_train, x_test
