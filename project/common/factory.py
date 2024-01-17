import torch
from typing import Callable, Tuple
from copy import deepcopy
from common.config import Config
from common import datasets
from common import models
from common import pruning
from common.nxutils import GraphManager
from common.constants import GLOBAL
from common.log import Logger

class Factory:
    '''
    TODO:
    order of calling the functions shouldnt change anything. Each function should be deterministic.
    '''
    def __init__(self, config: Config):
        self.config = config

    def make_dataloaders(self) -> Tuple:
        #torch_utils.set_seed(config.data_seed)
        x_train, y_train, x_test, y_test = datasets.make_dataset(
            name=self.config.dataset,
            n_samples=self.config.n_samples,
            noise=self.config.noise,
            seed=self.config.data_seed,
            factor=self.config.factor,
            scaler=self.config.scaler,
        )
        batch_size = self.config.batch_size if self.config.batch_size is not None else self.config.n_samples
        train_dataloader, test_dataloader = datasets.make_dataloaders(x_train, y_train, x_test, y_test, batch_size)

        return train_dataloader, test_dataloader

    def make_model(self) -> torch.nn.Module:
        shape = self.config.model_shape
        seed = self.config.model_seed
        activation = None  # models.__activations_map[self.config.activation] TODO:

        match self.config.model_class:
            case models.SingleTaskMultiClassMLP.__name__:
                model = models.SingleTaskMultiClassMLP(
                    shape=shape, 
                    seed=seed,
                    weight_init_func=models.Init[self.config.init_strategy_weights],
                    bias_init_func=models.Init[self.config.init_strategy_biases]
                )
            case models.MultiTaskBinaryMLP.__name__:
                model = models.MultiTaskBinaryMLP(
                    shape=shape, 
                    seed=seed,
                    weight_init_func=models.Init[self.config.init_strategy_weights],
                    bias_init_func=models.Init[self.config.init_strategy_biases]
                )
            case models.MLP.__name__:
                raise ValueError('You shouldnt use MLP, it doesnt have a loss defined.')
            case _:
                raise ValueError('Model Unkown')

        model = model.to(self.config.device)
        return model

    def make_pruner(self, model: torch.nn.Module) -> Callable:
        """
        Returns a Closure that prunes the model on call.
        """
        pruning_method = pruning.get_pruning_method(self.config.pruning_method)
        parameters = pruning.prunify_model(model, self.config.prune_weights, self.config.prune_biases)

        if self.config.pruning_scope == GLOBAL:
            return pruning.make_global_pruner(parameters, pruning_method)

        raise ValueError(f'Pruning scope not supported : {self.config.pruning_scope}')
    
    def make_renititializer(self, model: torch.nn.Module) -> Callable:
        """
        Returns a Closure that reinitializes the model on call.
        """
        # deepcopy to kill off references to the model parameters
        state_dict =  deepcopy(model.state_dict())

        # remove masks, because they shouldnt be reinitialized
        state_dict = {k: v for k, v in state_dict.items() if "_mask" not in k}
        
        def reinitialize(model: torch.nn.Module):
            """Closure that contains the reinit state_dict to reinitialize any model."""
            if self.config.reinit: 
                model.load_state_dict(state_dict, strict=False)

        return reinitialize

    def make_graph_manager(self, model: torch.nn.Module) -> GraphManager:
        return GraphManager(model, self.config.model_shape, self.config.task_description)

    def make_logger(self):
        return Logger(self.config.task_description)

def make_dataloaders(config: Config):
    
    x_train, y_train, x_test, y_test = datasets.make_dataset(
        name=config.dataset,
        n_samples=config.n_samples,
        noise=config.noise,
        seed=config.data_seed,
        factor=config.factor,
        scaler=config.scaler,
    )

    batch_size = config.batch_size if config.batch_size is not None else config.n_samples
    train_dataloader, test_dataloader = datasets.make_dataloaders(x_train, y_train, x_test, y_test, batch_size)

    return train_dataloader, test_dataloader

def make_model(config: Config):

    shape = config.model_shape
    seed = config.model_seed
    activation = models.__activations_map[config.activation]

    match config.model_class:
        case models.SingleTaskMultiClassMLP.__name__:
            model = models.SingleTaskMultiClassMLP(shape=shape, activation=activation, seed=seed)
        case models.MultiTaskBinaryMLP.__name__:
            model = models.MultiTaskBinaryMLP(shape=shape, activation=activation, seed=seed)
        case models.MLP.__name__:
            raise ValueError('You shouldnt use MLP, it doesnt have a loss defined.')
        case _:
            raise ValueError('Model Unkown')

    # because enums are parsed to strings in config, parse back and convert to enum
    model.init(
        weight_init_func=models.Init[config.init_strategy_weights],
        bias_init_func=models.Init[config.init_strategy_biases]
    )
    model = model.to(config.device)
    return model

