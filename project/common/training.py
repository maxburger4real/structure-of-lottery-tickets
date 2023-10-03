import wandb
import numpy as np
from torch import optim
from common.torch_utils import measure_global_sparsity

EVAL_LOSS = 'loss/eval'
TRAIN_LOSS = 'loss/train'
EPOCH = 'epoch'

def evaluate(model, loader, loss_fn):
    """Evaluate the model and return a numpy array of losses for each batch."""
    model.eval()
    losses = []
    for _, (x, y) in enumerate(loader):
        pred  = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.detach().cpu().numpy())

    return np.array(losses)

def update(model, loader, optim, loss_fn):
    """Update the model and return a numpy array of losses for each batch."""

    model.train()
    losses = []
    for _, (x, y) in enumerate(loader):
        optim.zero_grad()
        pred  = model(x)
        loss =  loss_fn(pred, y)
        loss.backward()
        optim.step()
        losses.append(loss.detach().cpu().numpy())

    return np.array(losses)

def train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, epochs=1):
    """Train a model for the specified amount of epochs."""
        
    N0, N, sparsity = measure_global_sparsity(model, use_mask=True)
    wandb.log({'nonzero' : N-N0, 'sparsity': sparsity}, commit=False)
    
    # train for epochs
    for epoch in range(1, epochs+1):
        loss_train = update(model, train_loader, optim, loss_fn).mean().item()
        loss_eval = evaluate(model, test_loader, loss_fn).mean().item()
        
        wandb.log({
            TRAIN_LOSS + f"/{sparsity:.2f}" : loss_train,
            EVAL_LOSS + f"/{sparsity:.2f}": loss_eval,
            EPOCH : epoch
        })

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

def build_optimizer(model, config):
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
