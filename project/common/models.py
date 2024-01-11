import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
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
    SILU : nn.SiLU,
    SIGM : nn.Sigmoid
}

def build_model_from_config(config: Config):

    shape = config.model_shape
    seed = config.model_seed
    activation = __activations_map[config.activation]

    match config.model_class:
        case SingleTaskMultiClassMLP.__name__:
            model = SingleTaskMultiClassMLP(shape=shape, activation=activation, seed=seed)
        case MultiTaskBinaryMLP.__name__:
            model = MultiTaskBinaryMLP(shape=shape, activation=activation, seed=seed)
        case MLP.__name__:
            raise ValueError('You shouldnt use MLP, it doesnt have a loss defined.')
        case _:
            raise ValueError('Model Unkown')

    # because enums are parsed to strings in config, parse back and convert to enum
    model.init(
        weight_strategy = InitializationStrategy[config.init_strategy_weights],
        bias_strategy = InitializationStrategy[config.init_strategy_biases]
    )
    model = model.to(config.device)
    return model


class BaseModel(nn.Module):
    """An abstract class that sets the seed for reproducible initialization."""
    def __init__(self, seed):
        super().__init__()
        if seed is None: seed = 0
        torch_utils.set_seed(seed)

    def init(self, weight_strategy, bias_strategy):
        def init_module(module):
            if not isinstance(module, nn.Linear): return
            match weight_strategy:
                case InitializationStrategy.NORMAL:
                    nn.init.normal_(module.weight, mean=0, std=0.1)

                case InitializationStrategy.XAVIER_NORMAL:
                    nn.init.xavier_normal_(module.weight)

                case InitializationStrategy.XAVIER_UNIFORM:
                    nn.init.xavier_uniform_(module.weight)

                case InitializationStrategy.KAIMING_NORMAL:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

                case InitializationStrategy.FRANKLE_XOR_TRUNC_NORMAL:
                    # from original LT paper V1, initialization for XOR problem.
                    # https://arxiv.org/pdf/1803.03635v1.pdf
                    mean, stddev = 0, 0.1
                    nn.init.trunc_normal_(module.weight, mean, stddev, a=-2*stddev, b=2*stddev)
                    
                case _:
                    print('Using Default initialization for Weights')

            match bias_strategy:
                case InitializationStrategy.ZERO:
                    nn.init.zeros_(module.bias)

                case _:
                    print('Using Default initialization for Bias')
     
        self.apply(init_module)
        return self


class MLP(BaseModel):
    """Versatile MLP."""
    def __init__(self, shape, activation, seed):
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
    def __init__(self, shape: torch.Size, activation=nn.ReLU, seed=None):        
        super().__init__(shape, activation, seed)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    
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
    def __init__(self, shape: torch.Size, activation=nn.ReLU, seed=None):
        super().__init__(shape, activation, seed)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

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
