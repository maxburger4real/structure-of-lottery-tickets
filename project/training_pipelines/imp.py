import wandb
from common.tracking import Config, save_model, logdict
from common.pruning import build_pruning_func, build_reinit_func
from common.training import build_early_stopper, build_optimizer, update, evaluate
from common.constants import *

def run(model, train_loader, test_loader, loss_fn, config: Config):

    # preparing for pruning and [OPTIONALLY] save model state
    prune = build_pruning_func(model, config)
    reinit = build_reinit_func(model)
    save_model(model, config, 0)

    # log initial performance
    initial_performace = evaluate(model, test_loader, loss_fn, config.device)
    wandb.log({PRUNABLE : config.params_prunable}, commit=False)
    wandb.log(logdict(initial_performace, VAL_LOSS))
    
    # loop over pruning levels
    params_prunable = config.params_prunable
    for lvl in range(1, config.pruning_levels+1):
        
        # train and evaluate the model
        optim = build_optimizer(model, config)
        stop = build_early_stopper(config)

        for epoch in range(0, config.training_epochs):
            loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
            loss_eval = evaluate(model, test_loader, loss_fn, config.device)
            if stop(loss_eval.mean().item()): break

        wandb.log(logdict(loss_train, TRAIN_LOSS), commit=False)
        wandb.log(logdict(loss_eval, VAL_LOSS), commit=False)

        save_model(model, config, lvl)

        amount_pruned = prune()
        params_prunable -= amount_pruned

        wandb.log({
            STOP : epoch, 
            PRUNABLE : params_prunable, 
        })

        if config.reinit: reinit(model)

    # final finetuning (optionally dont stop early)
    stop = build_early_stopper(config)
    for epoch in range(0, config.training_epochs):
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
        loss_eval = evaluate(model, test_loader, loss_fn, config.device)
        if stop(loss_eval.mean().item()): break

    wandb.log(logdict(loss_train, TRAIN_LOSS), commit=False)
    wandb.log(logdict(loss_eval, VAL_LOSS), commit=False)
    wandb.log({STOP : epoch})

    # save the finetuned model
    save_model(model, config, lvl+1)
