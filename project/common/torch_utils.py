import torch
import numpy as np
import random

SEED = 64

def get_pytorch_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def set_seed(seed):
    """Sets all seeds of randomness sources"""
    # TODO: set MPS seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def check_if_models_equal(model1, model2):
    """Returns True if both models are equal, otherwise False"""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def module_is_trainable(module: torch.nn.Module):
    return any(param.requires_grad for param in module.parameters())

def evaluate(model, loader, loss_fn):
    model.eval()
    losses = []
    for _, (x, y) in enumerate(loader):
        pred  = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.detach().numpy())
    return losses

def update(model, loader, optim, loss_fn):
    model.train()
    losses = []
    for _, (x, y) in enumerate(loader):
        optim.zero_grad()
        pred  = model(x)
        loss =  loss_fn(pred, y)
        loss.backward()
        optim.step()
        losses.append(loss.detach().numpy())
    return losses


def save_model(model):
    # TODO: unify model saving
    raise NotImplementedError
    return
    torch.save({
        'epoch': step,
        STATE_DICT: model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss.item(),
        'test_loss': loss_test.item(),
        }, base / f"{step}.pt")