import torch
import torch.nn as nn
import numpy as np
from common import torch_utils
from common.config import Config
from common.constants import *
"""
This module contains the MLP Architectures for reproducibility.
For reproducibility reasons, the *code* for each model architecture must be
saved. Otherwise it can not be recreated from state_dicts.
"""

__activations_map = {
    RELU : nn.ReLU,
    SILU : nn.SiLU
}

def build_model_from_config(config: Config):

    name = config.model_class
    shape = config.model_shape
    seed = config.model_seed
    activation = __activations_map[config.activation]

    if name == SimpleMLP.__name__:
        model =  SimpleMLP(shape, activation, seed)
        model = model.to(config.device)
        return model
    
    if name == InitMLP.__name__:
        model = InitMLP(shape=shape, activation=activation, seed=seed)
        model = model.to(config.device)
        return model
    
    raise ValueError('Model Unkown')

def _make_linear_init_kaiming_relu_zero_bias(in_features: int, out_features: int):

    linear = nn.Linear(in_features, out_features)
    nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
    nn.init.zeros_(linear.bias)

    return linear

def _make_linear_init_normal_relu_zero_bias(in_features: int, out_features: int):

    linear = nn.Linear(in_features, out_features)
    nn.init.normal_(linear.weight, mean=0.0, std=.5)
    nn.init.zeros_(linear.bias)

    return linear

class ReproducibleModel(nn.Module):
    """An abstract class that sets the seed for reproducible initialization."""
    def __init__(self, seed=None):
        super().__init__()
        if seed is None: seed = SEED
        torch_utils.set_seed(seed)


class SimpleMLP(ReproducibleModel):
    """A mini mlp for demo purposes."""
    def __init__(self, shape: torch.Size, activation=nn.ReLU, seed=None):
        super().__init__(seed)

        modules = []
        modules.append(nn.Linear(shape[0], shape[1]))

        for i in range(1, len(shape) - 1):
            modules.append(activation())
            in_dim = shape[i]
            out_dim = shape[i+1]
            modules.append(nn.Linear(in_dim, out_dim))

        self.modules = modules
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        y = self.model(x)
        return y
    

class InitMLP(ReproducibleModel):

    """A MLP where initialization is explicitly set."""
    def __init__(self, shape: torch.Size, activation=nn.ReLU, seed=None):
        super().__init__(seed)

        linear = _make_linear_init_normal_relu_zero_bias(shape[0], shape[1])
        modules = [linear]

        for i in range(1, len(shape) - 1):
            modules.append(activation())
            linear = _make_linear_init_normal_relu_zero_bias(shape[i], shape[i+1])
            modules.append(linear)

        self.modules = modules
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        y = self.model(x)
        return y


class MLP(ReproducibleModel):

    """A MLP where initialization is explicitly set."""
    def __init__(self, shape: torch.Size, activation=nn.ReLU, seed=None):
        super().__init__(seed)

        linear = _make_linear_init_kaiming_relu_zero_bias(shape[0], shape[1])
        modules = [linear]

        for i in range(1, len(shape) - 1):
            modules.append(activation())
            linear = _make_linear_init_kaiming_relu_zero_bias(shape[i], shape[i+1])
            modules.append(linear)

        self.modules = modules
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        y = self.model(x)
        return y