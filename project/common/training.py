import torch
import numpy as np
from torch import optim
from common.tracking import Config, ADAM, SGD, MSE, ADAMW

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

def update(model, loader, optim, loss_fn, device, lambda_l1=None):
    """Update the model and return a numpy array of losses for each batch."""
    model.train()
    losses = []
    for _, (x, y) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        pred  = model(x)
        loss = loss_fn(pred, y)

        # L1 regularization. 
        if lambda_l1 is not None and 0 < lambda_l1 < 1:
            l1 = torch.tensor(0., requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1 = l1 + torch.linalg.norm(param, 1)
            loss += lambda_l1.to(device) * l1.to(device)

        loss.backward()
        optim.step()
        losses.append(loss.detach().cpu().numpy())

    return np.array(losses)

def train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, config:Config):
    """Train a model for the specified amount of epochs."""

    train_losses, eval_losses = [], []

    # train for epochs
    for _ in range(0, config.training_epochs):
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.lambda_l1).mean()
        loss_eval = evaluate(model, test_loader, loss_fn, config.device).mean().item()

        train_losses += [loss_train]
        eval_losses += [loss_eval]

    return train_losses, eval_losses

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

def build_loss(config: Config):
    if config.loss_fn == MSE:
        loss_fn = torch.nn.MSELoss(reduction="mean")
        return loss_fn
    
    else:
        raise NotImplementedError
