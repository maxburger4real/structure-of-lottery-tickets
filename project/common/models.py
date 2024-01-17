import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from common import torch_utils
from common.constants import *
from enum import Enum
"""
This module contains the MLP Architectures for reproducibility.
For reproducibility reasons, the *code* for each model architecture must be
saved. Otherwise it can not be recreated from state_dicts.
"""

__activations_map = {
    RELU : nn.ReLU,
    SILU : nn.SiLU,
    SIGM : nn.Sigmoid
}

class Init(Enum):
    '''functions that get a tensor and manipulate it inplace'''
    zero=(nn.init.zeros_,)
    kaiming_normal=(lambda tensor: nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu'),)
    
    def __call__(self, tensor):
        self.value[0](tensor)


class BaseModel(nn.Module):
    """An abstract class that sets the seed for reproducible initialization."""
    def __init__(self, seed, weight_init_func, bias_init_func):
        super().__init__()
        if seed is None: seed = 0
        torch_utils.set_seed(seed)

        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

    def init(self):
        def init_module(module):
            if not isinstance(module, nn.Linear): return
            self.weight_init_func(module.weight)
            self.bias_init_func(module.bias)

        self.apply(init_module)
        return self


class MLP(BaseModel):
    """Versatile MLP."""
    def __init__(self, shape, activation, seed, weight_init_func, bias_init_func):
        super().__init__(seed, weight_init_func, bias_init_func)

        modules = []
        modules.append(nn.Linear(shape[0], shape[1]))

        for i in range(1, len(shape) - 1):
            modules.append(activation())
            in_dim = shape[i]
            out_dim = shape[i+1]
            modules.append(nn.Linear(in_dim, out_dim))

        self.modules = modules
        self.model = nn.Sequential(*modules)
        self.layers = [m for m in self.modules if torch_utils.module_is_trainable(m)]

    def forward(self, x):
        y = self.model(x)
        return y

    def predict(self, logits):
        raise NotImplementedError('Abstract method')

    def loss(self, logits, targets):
        raise NotImplementedError('Abstract method')
        
    def accuracy(self, logits, targets):
        raise NotImplementedError('Abstract method')


class MultiTaskBinaryMLP(MLP):
    '''
    MLP that supports parallel binary classification.
    each output of the model is treated as a binary classification problem, which is independent
    of the other outputs.
    '''
    def __init__(self, shape: torch.Size, activation=nn.ReLU, seed=None, weight_init_func=None, bias_init_func=None):        
        super().__init__(shape, activation, seed, weight_init_func, bias_init_func)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.init()
    
    def predict(self, logits):
        '''Logits are between -inf and inf. The decision-border is at 0.'''
        return (logits > 0).int()
    
    def loss(self, logits, targets):
        loss = self.loss_fn(logits, targets)
        
        if self.training:
            return loss.mean()
        
        return loss.mean(0).detach().cpu()
        
    def accuracy(self, logits, targets):
        pred = self.predict(logits)
        num_tasks = targets.shape[1]

        accs = []
        for i in range(num_tasks):
            accuracy = accuracy_score(
                y_true=targets[:,i], 
                y_pred=pred[:,i]
            )
            accs.append(accuracy)
        return np.array(accs, dtype=np.float32)


class SingleTaskMultiClassMLP(MLP):
    '''
    MLP that supports parallel Multiclass classification.
    each output of the model is treated as a binary classification problem, which is independent
    of the other outputs.
    '''
    def __init__(self, shape: torch.Size, activation=nn.ReLU, seed=None, weight_init_func=None, bias_init_func=None):
        super().__init__(shape, activation, seed, weight_init_func, bias_init_func)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.init()

    def predict(self, logits):
        '''Logits represent class probabilities. Maximum value is the prediction.'''
        return logits.argmax(axis=1)

    def loss(self, logits, targets):
        loss = self.loss_fn(logits, targets)
        return loss
    
    def accuracy(self, logits, targets):
        predictions = self.predict(logits)
        accuracy = accuracy_score(targets, predictions, normalize=True)
        return accuracy
