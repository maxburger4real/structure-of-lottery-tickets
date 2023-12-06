import wandb
from tqdm import tqdm
from common import log
from common.config import Config
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
    init_performace, init_accuracy = evaluate(model, test_loader, loss_fn, config.device)
    log.taskwise_metric(init_performace, VAL_LOSS)
    log.taskwise_metric(init_accuracy, ACCURACY)
    log.descriptive_statistics(model, at_init=True)

    # get the complete levels
    levels = range(-config.extension_levels, config.pruning_levels)
    epochs = config.epochs

    pborder = 0
    for level, pruning_amount in tqdm(
        zip(levels, trajectory, strict=True), 'Pruning Levels', len(levels) 
    ):

        log_now = log.returns_true_every_nth_time(n=config.log_every_n_epochs, and_at_0=True)

        # train and evaluate the model and log the performance
        optim = build_optimizer(model, config)
        stopper = build_early_stopper(config)

        for epoch in tqdm(range(epochs), f'Training Level {level}', epochs):
            loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda)
            loss_eval, acc = evaluate(model, test_loader, loss_fn, config.device)

            mean_train_loss: float = loss_train.mean().item()
            mean_eval_loss: float = loss_eval.mean().item()
            mean_eval_acc: float = acc.mean().item()

            if log_now():
                log.descriptive_statistics(model, prefix=f'epochwise-descriptive')
                wandb.log({
                    f'epochwise-train-loss' : mean_train_loss, 
                    f'epochwise-val-loss' : mean_eval_loss,
                    f'epochwise-accuracy' : mean_eval_acc
                })

            if stopper(mean_eval_loss): break

        save_model_or_skip(model, config, level)

        # log weights, biases, metrics at early stopping iteration
        log.scalar_metric(loss_train, TRAIN_LOSS) 
        log.taskwise_metric(loss_eval, VAL_LOSS)
        log.taskwise_metric(acc, ACCURACY)
        log.descriptive_statistics(model)

        # Log Graph based Statistics
        if config.log_graph_statistics:
            

            G = build_nx_graph(model, config)
            log.zombies_and_comatose(G, config)
            log.subnetworks(G, config)

        # IMPORTANT! this commits the logs that came before it.
        wandb.log({'pparams' : pparams, 'level' : level, 'stop' : epoch, 'pborder' : pborder})

        # prune the model  
        pborder = prune(pruning_amount)
        pparams -= pruning_amount

        # reinitialize
        if config.reinit: reinit(model)

    wandb.log({'level' : level+1, 'pborder' : pborder, 'pparams' : pparams})

    # final finetuning (optionally dont stop early)
    stopper = build_early_stopper(config)
    optim = build_optimizer(model, config)
    log_now = log.returns_true_every_nth_time(config.log_every_n_epochs)

    for epoch in tqdm(range(epochs), f'Final Finetuning',epochs):

        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda)
        loss_eval, acc = evaluate(model, test_loader, loss_fn, config.device)

        mean_train_loss: float = loss_train.mean().item()
        mean_eval_loss: float = loss_eval.mean().item()
        mean_eval_acc: float = acc.mean().item()

        if log_now():
            log.descriptive_statistics(model, prefix=f'epochwise-descriptive')
            wandb.log({
                f'epochwise-train-loss' : mean_train_loss, 
                f'epochwise-val-loss' : mean_eval_loss,
                f'epochwise-accuracy' : mean_eval_acc
            })

        if stopper(mean_eval_loss): break

     # log weights and biases metrics
    log.scalar_metric(loss_train, TRAIN_LOSS) 
    log.taskwise_metric(loss_eval, VAL_LOSS)
    log.taskwise_metric(acc, ACCURACY)
    log.descriptive_statistics(model)

    # save the finetuned model
    save_model_or_skip(model, config, level+1)
