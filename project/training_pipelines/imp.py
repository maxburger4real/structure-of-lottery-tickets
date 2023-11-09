import wandb
from common.nx_utils import build_nx_graph, neuron_analysis, subnet_analysis
from common.tracking import Config, save_model, logdict, log_param_aggregate_statistics, log_zombies_and_comatose, log_subnet_analysis
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
    
    # log initial weight distribution
    G = build_nx_graph(model, config)

    log_param_aggregate_statistics(G, config)

    params_prunable = config.params_prunable
    wandb.log({
            PRUNABLE : params_prunable,
            **logdict(initial_performace, VAL_LOSS),
        })
    
    # loop over pruning levels
    for lvl in range(1, config.pruning_levels+1):
        # train and evaluate the model and log the performance
        optim = build_optimizer(model, config)
        stop = build_early_stopper(config)
        for epoch in range(0, config.training_epochs):
            loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
            loss_eval = evaluate(model, test_loader, loss_fn, config.device)
            if loss_eval.mean().item() < config.loss_cutoff: break
            if stop(loss_eval.mean().item()): break
        
        save_model(model, config, lvl)

        # log parameter statistics
        G = build_nx_graph(model, config)

        log_param_aggregate_statistics(G, config)

        zombies, comatose = neuron_analysis(G, config)
        log_zombies_and_comatose(G, zombies, comatose)
        log_subnet_analysis(subnet_analysis(G, config))

        # prune and return the number of params pruned
        amount_pruned, pruning_border = prune()
        params_prunable -= amount_pruned

        wandb.log({
            STOP : epoch,
            PRUNABLE : params_prunable,
            'border' : pruning_border,
            **logdict(loss_eval, VAL_LOSS),
            **logdict(loss_train, TRAIN_LOSS), 
        })
        
        if config.reinit: reinit(model)

    # final finetuning (optionally dont stop early)
    stop = build_early_stopper(config)
    for epoch in range(0, config.training_epochs):
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
        loss_eval = evaluate(model, test_loader, loss_fn, config.device)
        if stop(loss_eval.mean().item()): break

    wandb.log({
        STOP : epoch,
        PRUNABLE : params_prunable,
        **logdict(loss_eval, VAL_LOSS),
        **logdict(loss_train, TRAIN_LOSS), 
    })

    # save the finetuned model
    save_model(model, config, lvl+1)
