"""low level training utilities"""
import torch
import numpy as np
from collections import defaultdict
from typing import Callable, Iterable, Dict, Any

from utils.logger import Logger


def evaluate(model, loader, device) -> Dict[str, Any]:
    model.eval()
    accs, losses = [], []
    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = model.loss(logits, y)
            accuracy = model.accuracy(logits, y)

            #assert accuracy.shape == loss.shape, "Metrics must have the same size"

            accs.append(accuracy)
            losses.append(loss)

    return {"val_loss": np.array(losses), "accuracy": np.array(accs)}


def update(model, loader, optim, device, lambda_l1=None) -> Dict[str, Any]:
    model.train()
    losses = []
    for _, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = model.loss(logits, y)

        # L1 regularization.
        if lambda_l1 is not None and 0 < lambda_l1 < 1:
            l1 = torch.tensor(0.0, requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if "weight" in name:
                    l1 = l1 + torch.linalg.norm(param, 1)
            loss += lambda_l1 * l1.to(device)

        loss.backward()
        optim.step()
        losses.append(loss.detach().cpu().numpy())

    return {"train_loss": np.array(losses)}


def train_and_evaluate(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    stopper: Callable,
    epochs: Iterable,
    device: torch.device,
    log_every: int = None,
    logger: Logger = None,
) -> Dict[str, Any]:
    metrics = defaultdict(list)
    metrics = []
    for epoch in epochs:
        train_metrics = update(model, train_loader, optim, device)
        eval_metrics = evaluate(model, test_loader, device)
        metrics.append({**train_metrics, **eval_metrics})

        if stopper(metrics[-1]["val_loss"].mean().item()):
            break

        if logger is not None and log_every is not None and epoch % log_every == 0:
            #latest = {k: v[-1] for k, v in metrics.items()}
            logger.metrics(metrics[-1])
            logger.commit()

    return metrics[-1]
