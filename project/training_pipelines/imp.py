import wandb
from common.config import Config
from common.logging import logdict, log_param_aggregate_statistics, log_zombies_and_comatose, log_subnet_analysis
from common.nx_utils import build_nx_graph
from common.persistance import save_model
from common.pruning import build_pruning_func, build_reinit_func, count_trainable_and_prunable_params
from common.training import build_early_stopper, build_optimizer, update, evaluate
from common.constants import *

def run(model, train_loader, test_loader, loss_fn, config: Config):

    # preparing for pruning and [OPTIONALLY] save model state
    prune, trajectory = build_pruning_func(model, config)
    reinit = build_reinit_func(model)
    trainable, prunable = count_trainable_and_prunable_params(model)

    save_model(model, config, f'{-config.extension_levels}_init')

    # log initial performance
    initial_performace = evaluate(model, test_loader, loss_fn, config.device)
    
    # log initial weight distribution
    G = build_nx_graph(model, config)
    log_param_aggregate_statistics(G, config)

    wandb.log({
            PRUNABLE : prunable,
            **logdict(initial_performace, VAL_LOSS),
        })
    
    # get the complete levels
    levels = range(-config.extension_levels, config.pruning_levels)

    for level, pruning_amount in zip(levels, trajectory, strict=True):

        # train and evaluate the model and log the performance
        optim = build_optimizer(model, config)
        stop = build_early_stopper(config)

        for epoch in range(0, config.training_epochs):
            loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
            loss_eval = evaluate(model, test_loader, loss_fn, config.device)
            if loss_eval.mean().item() < config.loss_cutoff: break
            if stop(loss_eval.mean().item()): break
        
        save_model(model, config, level)

        # Log Graph based Statistics
        G = build_nx_graph(model, config)
        log_param_aggregate_statistics(G, config)
        log_zombies_and_comatose(G, config)
        log_subnet_analysis(G, config)

        # prune and return the number of params pruned
        pruning_border = prune(pruning_amount)
        prunable -= pruning_amount

        wandb.log({
            'level' : level,
            STOP : epoch,
            PRUNABLE : prunable,
            'border' : pruning_border,
            **logdict(loss_eval, VAL_LOSS),
            **logdict(loss_train, TRAIN_LOSS), 
        })
        
        if config.reinit: reinit(model)

    # final finetuning (optionally dont stop early)
    stop = build_early_stopper(config)
    optim = build_optimizer(model, config)

    for epoch in range(config.training_epochs):
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
        loss_eval = evaluate(model, test_loader, loss_fn, config.device)
        if stop(loss_eval.mean().item()): break

    wandb.log({
        STOP : epoch,
        PRUNABLE : prunable,
        **logdict(loss_eval, VAL_LOSS),
        **logdict(loss_train, TRAIN_LOSS), 
    })

    # save the finetuned model
    save_model(model, config, level+1)
