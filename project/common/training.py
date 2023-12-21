import torch
import numpy as np
from torch import optim
from sklearn.metrics import accuracy_score

from common.config import Config
from common.constants import *

def calc_accuracy(logits, y):
    '''Calculate Accuracy from logits and labels.'''
    pred = (logits > 0).int()
    assert pred.shape == y.shape, 'True labels must habe same shape as predictions.'
    accs = []
    for i in range(y.shape[1]):
        y_true = y[:,i]
        y_pred = pred[:,i]
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc)
    return np.array(accs, dtype=np.float32)

def evaluate(model, loader, loss_fn, device, accuracy=True):
    """Evaluate the model and return a numpy array of losses for each batch."""

    model.eval()
    accs, losses = [],[]
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            x,y = x.to(device), y.to(device)

            pred  = model(x)
            batch_loss = loss_fn(pred, y).mean(axis=0).numpy()
            batch_acc = calc_accuracy(pred.detach().cpu(), y.detach().cpu())
            assert batch_acc.shape == batch_loss.shape, 'Metrics must have the same size'

            accs.append(batch_acc)
            losses.append(batch_loss)

    return np.array(losses), np.array(accs)

def update(model, loader, optim, loss_fn, device, lambda_l1=None):
    """
    Update the model and 
    return a numpy array of losses for each batch.
    """
    model.train()
    losses = []
    for _, (x, y) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        pred  = model(x)
        loss = loss_fn(pred, y).mean()

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

def train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, config:Config):
    """Train a model for the specified amount of epochs."""

    train_losses, eval_losses = [], []

    stop = build_early_stopper(config)

    # train for epochs
    for _ in range(0, config.epochs):
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
        loss_eval, acc = evaluate(model, test_loader, loss_fn, config.device).mean().item()

        train_losses += [loss_train]
        eval_losses += [loss_eval]

        if stop(loss_eval): break


    return np.array(train_losses), np.array(eval_losses)

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

def build_loss_from_config(config: Config):
    if config.loss_fn == MSE:
        return torch.nn.MSELoss(reduction='mean')
    
    elif config.loss_fn == CCE:
        # Multiclass [cat, dog, mouse]
        return torch.nn.CrossEntropyLoss(reduction='mean')

    elif config.loss_fn == BCE:
        # Binary and Multi-label-Binary
        return torch.nn.BCEWithLogitsLoss(reduction='none')

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

    def __init__(self, patience=None, min_delta=0, loss_cutoff=None):
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.loss_cutoff = loss_cutoff
        self.min_loss = float('inf')

    def __call__(self, loss):

        if self.loss_cutoff is not None: 
            if loss < self.loss_cutoff:
                return True

        loss_decreased = loss < self.min_loss
        loss_increased = loss > (self.min_loss + self.min_delta)

        if loss_decreased:
            self.min_loss = loss
            self.counter = 0

        if self.patience is None: return False

        elif loss_increased:
            self.counter += 1
            loss_increased_too_often = (self.counter >= self.patience)
            
            if loss_increased_too_often:
                return True
        
        
        return False
