import numpy as np
import wandb
from common.tracking import Config, save_model
from common.training import build_optimizer, evaluate

VAL_LOSS = 'val_loss'
TRAIN_LOSS = 'train_loss'

def run(model, train_loader, test_loader, loss_fn, config: Config):

    optim = build_optimizer(model, config)
    swap_log = config.bimt_swap if config.bimt_swap is not None else np.inf

    lamb = config.lamb
    if lamb <= 0: raise ValueError("Lambda of 0 or less is not allowed")

    LOCAL = config.bimt_local 
    weight_factor = 1. if LOCAL else 0.
    
    # log initial performance
    eval_loss_init = evaluate(model, test_loader, loss_fn, config.device).mean().item()
    wandb.log({VAL_LOSS : eval_loss_init})
    
    steps=config.training_epochs
    for step in range(steps):

        # crank up lambda after first quarter of training
        # damp down lambda after third quarter of training
        if step == int(steps/4): lamb *= 10
        if step == int(3*steps/4): lamb *= 0.1
        
        optim.zero_grad()
        model.train()
        losses = []
        for _, (x, y) in enumerate(train_loader):
            x,y = x.to(config.device), y.to(config.device)
            pred  = model(x)
            loss = loss_fn(pred,y)
            bias_penalize = False if step < int(3*steps/4) else True
            reg = model.get_cc(bias_penalize=bias_penalize, weight_factor=weight_factor)
            total_loss = loss + lamb*reg
            total_loss.backward()
            optim.step()
            losses.append(loss.detach().cpu().numpy())

        # evaluate forward
        model.eval()
        eval_loss = evaluate(model, test_loader, loss_fn, config.device).mean().item()
        wandb.log({VAL_LOSS : eval_loss, TRAIN_LOSS : np.array(losses).mean()})

        if (step+1) % swap_log == 0:
            save_model(model, config, step)
            model.relocate()
            save_model(model, config, step+1)
    
    if config.bimt_prune is not None:
        model.thresholding(config.bimt_prune)
        eval_loss = evaluate(model, test_loader, loss_fn, config.device).mean().item()
        wandb.log({VAL_LOSS : eval_loss})
        save_model(model, config, steps+1)


    