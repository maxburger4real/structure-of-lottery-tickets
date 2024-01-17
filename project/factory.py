import torch

from typing import Callable, Tuple
from copy import deepcopy
from training import datasets, models, pruning, config

from training.config import Config
from utils.nxutils import GraphManager
from utils.logger import Logger


class Factory:
    """
    TODO:
    order of calling the functions shouldnt change anything. Each function should be deterministic.
    for now, inversing order of Data generation and model generation changes, probably just the data.
    convention:
    call model, then data.
    """

    def __init__(self, config: config.Config):
        self.config: Config = config

    def make_dataloaders(self) -> Tuple:
        # torch_utils.set_seed(config.data_seed)
        x_train, y_train, x_test, y_test = datasets.make_dataset(
            name=self.config.dataset,
            n_samples=self.config.n_samples,
            noise=self.config.noise,
            seed=self.config.data_seed,
            factor=self.config.factor,
            scaler=self.config.scaler,
        )
        batch_size = (
            self.config.batch_size
            if self.config.batch_size is not None
            else self.config.n_samples
        )
        train_dataloader, test_dataloader = datasets.make_dataloaders(
            x_train, y_train, x_test, y_test, batch_size
        )

        return train_dataloader, test_dataloader

    def make_model(self) -> torch.nn.Module:
        shape = self.config.model_shape
        seed = self.config.model_seed

        match self.config.model_class:
            case models.SingleTaskMultiClassMLP.__name__:
                model = models.SingleTaskMultiClassMLP(
                    shape=shape,
                    seed=seed,
                    weight_init_func=models.Init[self.config.init_strategy_weights],
                    bias_init_func=models.Init[self.config.init_strategy_biases],
                )
            case models.MultiTaskBinaryMLP.__name__:
                model = models.MultiTaskBinaryMLP(
                    shape=shape,
                    seed=seed,
                    weight_init_func=models.Init[self.config.init_strategy_weights],
                    bias_init_func=models.Init[self.config.init_strategy_biases],
                )
            case models.MLP.__name__:
                raise ValueError("You shouldnt use MLP, it doesnt have a loss defined.")
            case _:
                raise ValueError("Model Unkown")

        model = model.to(self.config.device)
        return model

    def make_pruner(self, model: torch.nn.Module) -> Callable:
        """
        Returns a Closure that prunes the model on call.
        """
        pruning_method = pruning.get_pruning_method(self.config.pruning_method)
        parameters = pruning.prunify_model(
            model, self.config.prune_weights, self.config.prune_biases
        )

        if self.config.pruning_scope == "global":
            return pruning.make_global_pruner(parameters, pruning_method)

        raise ValueError(f"Pruning scope not supported : {self.config.pruning_scope}")

    def make_renititializer(self, model: torch.nn.Module) -> Callable:
        """
        Returns a Closure that reinitializes the model on call.
        """
        # deepcopy to kill off references to the model parameters
        state_dict = deepcopy(model.state_dict())

        # remove masks, because they shouldnt be reinitialized
        state_dict = {k: v for k, v in state_dict.items() if "_mask" not in k}

        def reinitialize(model: torch.nn.Module):
            """Closure that contains the reinit state_dict to reinitialize any model."""
            if self.config.reinit:
                model.load_state_dict(state_dict, strict=False)

        return reinitialize

    def make_graph_manager(self, model: torch.nn.Module) -> GraphManager:
        return GraphManager(
            model, self.config.model_shape, self.config.task_description
        )

    def make_logger(self):
        return Logger(self.config.task_description)

    def make_optimizer(self, model):
        parameters = model.parameters()
        lr = self.config.lr

        match self.config.optimizer:
            case "sgd":
                momentum = 0 if self.config.momentum is None else self.config.momentum
                return torch.optim.SGD(parameters, lr, momentum)
            case "adam":
                return torch.optim.Adam(parameters, lr)
            case _:
                raise NotImplementedError("")

    def make_stopper(self):
        return EarlyStopper(
            patience=self.config.early_stop_patience,
            min_delta=self.config.early_stop_delta,
        )


class EarlyStopper:
    """from https://stackoverflow.com/a/73704579 assuming the loss is always larger than 0.
    - positive min_delta can be used to define,
        how big a loss increase must be to count as such
    - negative min_delta can be used to define,
        how big an improvement must be,
        to not count as a loss increase
    """

    def __init__(self, patience=None, min_delta=0):
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = float("inf")

    def reset(self):
        self.counter = 0
        self.min_loss = float("inf")

    def __call__(self, loss):
        loss_decreased = loss < self.min_loss
        loss_increased = loss > (self.min_loss + self.min_delta)

        if loss_decreased:
            self.min_loss = loss
            self.counter = 0

        if self.patience is None:
            return False

        elif loss_increased:
            self.counter += 1
            loss_increased_too_often = self.counter >= self.patience
            if loss_increased_too_often:
                return True

        return False
