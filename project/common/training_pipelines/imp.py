import wandb
from tqdm import tqdm
from common.config import Config
from common.log import log_loss, log_zombies_and_comatose, log_subnet_analysis, log_descriptive_statistics, every_n
from common.nx_utils import build_nx_graph
from common.persistance import save_model_or_skip
from common.pruning import build_pruning_func, build_reinit_func, count_prunable_params
from common.training import build_early_stopper, build_optimizer, update, evaluate
from common.constants import *

def run(model, train_loader, test_loader, loss_fn, config: Config):

    # preparing for pruning and [OPTIONALLY] save model state
    prune, trajectory = build_pruning_func(model, config)
    reinit = build_reinit_func(model)
    pparams = count_prunable_params(model)
    
    save_model_or_skip(model, config, f'{-config.extension_levels}_init')

    # log initial performance and descriptive statistics
    initial_performace = evaluate(model, test_loader, loss_fn, config.device)
    log_loss(initial_performace, prefix=VAL_LOSS)
    log_descriptive_statistics(model, at_init=True)

    # get the complete levels
    levels = range(-config.extension_levels, config.pruning_levels)

    pborder = 0
    for level, pruning_amount in tqdm(
        zip(levels, trajectory, strict=True), 
        total=len(levels), 
        desc='running pruning levels'
    ):

        log_now = every_n(config.log_every_n_epochs)

        # train and evaluate the model and log the performance
        optim = build_optimizer(model, config)
        stop = build_early_stopper(config)

        for epoch in range(config.training_epochs):
            loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
            loss_eval = evaluate(model, test_loader, loss_fn, config.device)

            if log_now():
                log_descriptive_statistics(model, prefix=f'epochs Lv. {level} descriptive')
                wandb.log({
                    f'epochs Lv. {level} epoch-loss-train' : loss_train.item(),
                    f'epochs Lv. {level} epoch-loss-val' : loss_eval.mean().item(),
                    'epoch' : epoch
                })

            if loss_eval.mean().item() < config.loss_cutoff: break
            if stop(loss_eval.mean().item()): break

        save_model_or_skip(model, config, level)

        # log weights and biases metrics
        log_loss(loss_eval, VAL_LOSS)
        log_loss(loss_train, TRAIN_LOSS) 
        log_descriptive_statistics(model)

        # Log Graph based Statistics
        if config.log_graph_statistics:
            G = build_nx_graph(model, config)
            log_zombies_and_comatose(G, config)
            log_subnet_analysis(G, config)
        wandb.log({'pparams' : pparams, 'level' : level, 'stop' : epoch, 'pborder' : pborder})

        # prune the model  
        pborder = prune(pruning_amount)
        pparams -= pruning_amount

        if config.reinit: reinit(model)

    wandb.log({'level' : level+1, 'pborder' : pborder, 'pparams' : pparams})

    # final finetuning (optionally dont stop early)
    stop = build_early_stopper(config)
    optim = build_optimizer(model, config)
    log_now = every_n(config.log_every_n_epochs)

    for epoch in range(config.training_epochs):
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
        loss_eval = evaluate(model, test_loader, loss_fn, config.device)
        if log_now():
            wandb.log({
                f'Lv. {level} epoch-loss-train' : loss_train.item(),
                f'Lv. {level} epoch-loss-val' : loss_eval.mean().item(),
                'epoch' : epoch
            })
        if stop(loss_eval.mean().item()): break

     # log weights and biases metrics
    log_loss(loss_eval, VAL_LOSS)
    log_loss(loss_train, TRAIN_LOSS) 
    log_descriptive_statistics(model)

    # save the finetuned model
    save_model_or_skip(model, config, level+1)
