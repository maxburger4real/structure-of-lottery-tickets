"""This file contains everything to create datasets."""
import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from common.config import Config
from common.constants import *
from common import torch_utils

circles_inputs = moons_inputs = 2
circles_outputs = moons_outputs = 1
val_set_size = 200

# visible
def concat_datasets(list_of_datasets):
    list_of_x, list_of_y = list(zip(*list_of_datasets))
    
    x = torch.Tensor(np.concatenate(list_of_x, axis=1))

    list_of_y_unsqueezed = [y.reshape(-1,1) for y in list_of_y]
    y = torch.Tensor(np.concatenate(list_of_y_unsqueezed, axis=1))

    return x, y

def scale(x_train, x_test, Scaler):
    """Scale the dataset featurewise."""
    if Scaler is None:
        return x_train, x_test
    
    if Scaler == MinMaxZeroMean: 
        scaler = MinMaxScaler(feature_range=(-1,1))
    elif Scaler == MinMaxZeroOne:
        scaler = MinMaxScaler(feature_range=(0,1))
    elif Scaler == StandardUnitVariance:
        scaler = StandardScaler()

    scaler = scaler.fit(x_train)
    x_train = torch.from_numpy(scaler.transform(x_train)).float()
    x_test = torch.from_numpy(scaler.transform(x_test)).float()
    return x_train, x_test

def __make_old_moons(n_samples, noise, seed, Scaler):
    
    description = (
        ('moons-1', (moons_inputs, moons_outputs)),
        ('moons-2', (moons_inputs, moons_outputs)),
    )

    torch_utils.set_seed(seed)

    train_datasets = [datasets.make_moons(n_samples, noise=noise) for _ in range(2)]
    x_train, y_train = concat_datasets(train_datasets)

    test_datasets = [datasets.make_moons(n_samples, noise=noise) for _ in range(2)]
    x_test, y_test = concat_datasets(test_datasets)
    
    x_train, x_test = scale(x_train, x_test, Scaler)
 
    return x_train, y_train, x_test, y_test, description

def __make_flip_moons(n_samples, noise, seed, Scaler):

    description = (
        ('moons', (moons_inputs, moons_outputs)),
        ('snoom', (moons_inputs, moons_outputs)),
    )

    torch_utils.set_seed(seed)

    moons = datasets.make_moons(n_samples, noise=noise)
    moons_x, moons_y = datasets.make_moons(n_samples, noise=noise)
    snoom_x, snoom_y = np.flip(moons_x), np.flip(moons_y)
    x_train, y_train = concat_datasets([moons,(snoom_x, snoom_y)])

    test_moons = datasets.make_moons(n_samples, noise=noise)
    test_moons_x, test_moons_y = datasets.make_moons(n_samples, noise=noise)
    test_snoom_x, test_snoom_y = np.flip(test_moons_x), np.flip(test_moons_y)
    x_test, y_test = concat_datasets([test_moons,(test_snoom_x, test_snoom_y)])    
    
    x_train, x_test = scale(x_train, x_test, Scaler)
 
    return x_train, y_train, x_test, y_test, description

def __make_circles_and_moons(n_samples, noise, seed, factor, Scaler):
    description = (
        ('circles', (circles_inputs, circles_outputs)),
        ('moons', (moons_inputs, moons_outputs)),
    )

    torch_utils.set_seed(seed)

    circles = datasets.make_circles(int(n_samples), noise=noise, factor=factor)
    moons = datasets.make_moons(int(n_samples), noise=noise)
    x_train, y_train = concat_datasets([circles, moons])

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    test_circles = datasets.make_circles(n_samples, noise=noise, factor=factor)
    test_moons = datasets.make_moons(n_samples, noise=noise)
    x_test, y_test = concat_datasets([test_circles, test_moons])

    x_train, x_test = scale(x_train, x_test, Scaler)
 
    return x_train, y_train, x_test, y_test, description


def __make_circles_and_moons_2(n_samples, noise, seed, factor, Scaler):
    description = (
        ('circles', (circles_inputs, circles_outputs)),
        ('moons', (moons_inputs, moons_outputs)),
    )

    torch_utils.set_seed(seed)

    circles = datasets.make_circles(int(n_samples*2), noise=noise, factor=factor)
    moons = datasets.make_moons(int(n_samples*2), noise=noise)
    x, y = concat_datasets([circles, moons])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    #test_circles = datasets.make_circles(n_samples, noise=noise, factor=factor)
    #test_moons = datasets.make_moons(n_samples, noise=noise)
    #x_test, y_test = concat_datasets([test_circles, test_moons])

    x_train, x_test = scale(x_train, x_test, Scaler)
 
    return x_train, y_train, x_test, y_test, description


def build_dataloaders_from_config(config: Config):
    
    n_samples = config.n_samples
    noise = config.noise
    seed = config.data_seed
    batch_size = config.batch_size if config.batch_size is not None else n_samples
    factor = config.factor
    
    match config.dataset:
        case Datasets.OLD_MOONS.name:
            *data, description = __make_old_moons(n_samples, noise, seed, config.scaler)

        case Datasets.FLIP_MOONS.name:
            *data, description = __make_flip_moons(n_samples, noise, seed, config.scaler)

        case Datasets.CIRCLES_AND_MOONS.name:
            *data, description = __make_circles_and_moons(n_samples, noise, seed, factor, config.scaler)

        case _:
            raise ValueError(f'Unknown dataset {config.dataset}')
    
    x_train, y_train, x_test, y_test = data

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

    config.update({'task_description' : description }, allow_val_change=True)

    return train_dataloader, test_dataloader

    if config.dataset == Datasets.OLD_MOONS.name:
        config.update({
            'task_description' : (
                ('moons-1', (moons_inputs, moons_outputs)),
                ('moons-2', (moons_inputs, moons_outputs)),
            )
        }, allow_val_change=True)

        torch_utils.set_seed(config.data_seed)

        train_datasets = [datasets.make_moons(n_samples, noise=noise) for _ in range(2)]
        x_train, y_train = concat_datasets(train_datasets)

        test_datasets = [datasets.make_moons(n_samples, noise=noise) for _ in range(2)]
        x_test, y_test = concat_datasets(test_datasets)

        if config.batch_size is None: batch_size = n_samples
        train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(TensorDataset(x_test, y_test))

        return train_dataloader, test_dataloader
    
    if config.dataset == Datasets.MOONS_AND_CIRCLES.name:
        config.update({
            'task_description' : (
                ('moons' , (moons_inputs, moons_outputs)),
                ('circles', (circles_inputs, circles_outputs))
            )
        }, allow_val_change=True)
        return __build_moons_and_circles_dl(n_samples=n_samples, noise=noise, batch_size=config.batch_size)
    batch_size = config.batch_size

    if config.dataset == Datasets.OLD_MOONS.name:
        
        config.update({
            'task_description' : (
                ('moons-1', (moons_inputs, moons_outputs)),
                ('moons-2', (moons_inputs, moons_outputs)),
            )
        }, allow_val_change=True)
        torch_utils.set_seed(config.data_seed)
        return __build_moons_and_moons_dl(n_samples=n_samples, noise=noise, batch_size=config.batch_size)
    
    if config.dataset == Datasets.MOONS_AND_CIRCLES.name:
        config.update({
            'task_description' : (
                ('moons' , (moons_inputs, moons_outputs)),
                ('circles', (circles_inputs, circles_outputs))
            )
        }, allow_val_change=True)
        return __build_moons_and_circles_dl(n_samples=n_samples, noise=noise, batch_size=batch_size)

    if config.dataset == Datasets.MOONS_AND_MOONS.name:
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
        __make_moons(n, noise=noise, random_state=1),
        __make_circles(n, noise=noise, random_state=1),
    )

    n_test = int(val_set_size/2)
    test_dataset = __concat_datasets(
        __make_moons(n_test, noise=noise, random_state=2),
        __make_circles(n_test, noise=noise, random_state=2),
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

    n_test = int(val_set_size/2)
    test_dataset = __concat_datasets(
        __make_moons(n_test, noise=noise, random_state=2),
        __make_moons(n_test, noise=noise, random_state=3),
    )

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset)

    return train_loader, test_loader

def __build_moons_dl(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # TODO: 
    train_dataset = __make_moons(n_samples, noise=noise, random_state=1)
    test_dataset = __make_moons(val_set_size, noise=noise, random_state=2)

    train_loader = __build_dataloader(*train_dataset, batch_size=batch_size)
    test_loader = __build_dataloader(*test_dataset)

    return train_loader, test_loader

def __build_circles_dl(n_samples, noise, batch_size=None):
    """Deterministically sample a train and a test dataset of the same size."""
    # sample the data
    train_dataset = __make_circles(n_samples, noise=noise, random_state=1)
    test_dataset = __make_circles(val_set_size, noise=noise, random_state=2)

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
