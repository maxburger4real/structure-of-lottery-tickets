from collections import defaultdict
from typing import Callable, Iterable

import torch
import numpy as np
from torch import optim

from common.log import Logger
from common.config import Config
from common.constants import *

def evaluate(model, loader, device):
    """Evaluate the model and return a numpy array of losses for each batch."""

    model.eval()
    accs, losses = [],[]
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            x,y = x.to(device), y.to(device)

            logits = model(x)
            loss = model.loss(logits, y)
            accuracy = model.accuracy(logits, y)
            
            assert accuracy.shape == loss.shape, 'Metrics must have the same size'

            accs.append(accuracy)
            losses.append(loss)

    return np.array(losses), np.array(accs)

def update(model, loader, optim, device, lambda_l1=None):
    """
    Update the model and 
    return a numpy array of losses for each batch.
    """
    model.train()
    losses = []
    for _, (x, y) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = model.loss(logits, y)

        # L1 regularization. 
        if lambda_l1 is not None and 0 < lambda_l1 < 1:
            l1 = torch.tensor(0., requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1 = l1 + torch.linalg.norm(param, 1)
            loss += lambda_l1 * l1.to(device)

        loss.backward()
        optim.step()
        losses.append(loss.detach().cpu().numpy())

    return np.array(losses)

def train_and_evaluate(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    test_loader: torch.utils.data.DataLoader, 
    optim: torch.optim.Optimizer, 
    logger: Logger, 
    stopper: Callable, 
    epochs: Iterable, 
    device: torch.device, 
    log_every: int
):

    metrics = defaultdict(list)

    for epoch in epochs:

        train_loss = update(model, train_loader, optim, device)
        val_loss, val_acc = evaluate(model, test_loader, device)

        metrics[TRAIN_LOSS].append(train_loss.mean().item())
        metrics[VAL_LOSS].append(val_loss.mean().item())
        metrics[ACCURACY].append(val_acc.mean().item())

        if stopper(metrics[VAL_LOSS][-1]):
            break

        if log_every is not None and epoch % log_every == 0 and epoch != epochs[-1]:
            latest = {k: v[-1] for k, v in metrics.items()}
            logger.metrics(latest, prefix='epoch/')
            logger.commit()

    logger.metrics({k: v[-1] for k, v in metrics.items()})
    return metrics

def build_optimizer(model, config: Config):
    """inspired by wandb
    https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=KfkduI6qWBrb
    """
    if config.optimizer == SGD:
        momentum = 0 if config.momentun is None else config.momentum
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr, 
            momentum=momentum
        )
    elif config.optimizer == ADAM:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=0.0,
        )
    elif config.optimizer == ADAMW:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=0.0
        )
    return optimizer

def build_early_stopper(config: Config):
    """Returns a callable object that decides if to early stop."""
    return EarlyStopper(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_delta
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
        self.min_loss = float('inf')

    def reset(self):
        self.counter = 0
        self.min_loss = float('inf')

    def __call__(self, loss):
        loss_decreased = loss < self.min_loss
        loss_increased = loss > (self.min_loss + self.min_delta)

        if loss_decreased:
            self.min_loss = loss
            self.counter = 0

        if self.patience is None: return False

        elif loss_increased:
            self.counter += 1
            loss_increased_too_often = (self.counter >= self.patience)
            if loss_increased_too_often: return True
        
        return False
