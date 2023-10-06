import wandb
import torch
import numpy as np
from torch import optim
from common.torch_utils import measure_global_sparsity
from common.tracking import Config

EVAL_LOSS = 'loss/eval'
TRAIN_LOSS = 'loss/train'
EPOCH = 'epoch'

def evaluate(model, loader, loss_fn, device):
    """Evaluate the model and return a numpy array of losses for each batch."""

    model.eval()
    losses = []
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            x,y = x.to(device), y.to(device)

            pred  = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.detach().cpu().numpy())

    return np.array(losses)

def update(model, loader, optim, loss_fn, device):
    """Update the model and return a numpy array of losses for each batch."""
    model.train()
    losses = []
    for _, (x, y) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        pred  = model(x)
        loss =  loss_fn(pred, y)
        loss.backward()
        optim.step()
        losses.append(loss.detach().cpu().numpy())

    return np.array(losses)

def train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, device, epochs=1):
    """Train a model for the specified amount of epochs."""

    train_losses, eval_losses = [], []

    # train for epochs
    for _ in range(0, epochs):
        loss_train = update(model, train_loader, optim, loss_fn, device).mean()
        loss_eval = evaluate(model, test_loader, loss_fn, device).mean().item()

        train_losses += [loss_train]
        eval_losses += [loss_eval]

    return train_losses, eval_losses

def train(model, loader, optim, loss_fn, epochs=1):
    """
    @deprecated
    Train a model for the specified amount of epochs."""
    losses = []
    for _ in range(epochs):
        loss = update(model, loader, optim, loss_fn)
        avg_loss = loss.mean()
        losses.append(avg_loss)

    return np.array(losses)

def build_optimizer(model, config: Config):
    """inspired by wandb
    https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=KfkduI6qWBrb
    """
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr, 
            momentum=config.momentum
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr
        )

    return optimizer

def build_model(config: Config):
    ModelClass = config.model_class
    model = ModelClass.make_from_config(config)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    # send to device
    model.to(config.device)

    return model, loss_fn
