import wandb
from common.nx_utils import subnet_analysis, build_nx_graph, neuron_analysis
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
    
    # log initial weight distribution
    G, _ = build_nx_graph(model, config)
    for u, v, data in G.edges(data=True):
        l = G.nodes(data=True)[u][LAYER]
        wandb.log({f'w-{u,v}-l-{l}': data[WEIGHT]}, commit=False)

    for i, data in G.nodes(data=True):
        l = data[LAYER]
        wandb.log({f'b-{i}-l-{l}': data[BIAS]}, commit=False)

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

        # create the nx.Graph from torch.model
        G, _ = build_nx_graph(model, config)
        for u, v, data in G.edges(data=True):
            l = G.nodes(data=True)[u][LAYER]
            wandb.log({f'w-{u,v}-l-{l}': data[WEIGHT]}, commit=False)

        for i, data in G.nodes(data=True):
            l = data[LAYER]
            b = data[BIAS]
            if b == 0: continue
            wandb.log({f'b-{i}-l-{l}': b}, commit=False)

        subnet_report = subnet_analysis(G, config)
        zombies, comatose = neuron_analysis(G, config)

        # log zombies and comatose
        wandb.log({
            'zombies' : len(zombies), 
            'comatose' : len(comatose)
            }, commit=False)
        
        for i, data in G.subgraph(zombies).nodes(data=True):
            wandb.log({
                f'zombie-neuron-{i}': i,
                f'zombie-bias-{i}' : data[BIAS],
                f'zombie-out-{i}' : data['out']
            }, commit=False)

        for i, data in G.subgraph(comatose).nodes(data=True):
            wandb.log({
                f'comatose-neuron-{i}': i,
                f'comatose-bias-{i}' : data[BIAS],
                f'comatose-in-{i}' : data['in']
            }, commit=False)

        wandb.log({
            STOP : epoch, 
            PRUNABLE : params_prunable, 
        })

        amount_pruned = prune()
        params_prunable -= amount_pruned
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
