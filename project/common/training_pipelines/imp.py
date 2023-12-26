from tqdm import tqdm
from common.log import Logger
from common.config import Config
from common.nxutils import GraphManager
from common.persistance import save_model_or_skip
from common.pruning import build_pruning_func, build_reinit_func
from common.training import build_early_stopper, build_optimizer, update, evaluate
from common.constants import *

def run(model, train_loader, test_loader, loss_fn, config: Config):

    # preparing for pruning and [OPTIONALLY] save model state
    prune = build_pruning_func(model, config)
    reinit = build_reinit_func(model)
    gm = GraphManager(model, config.model_shape, config.task_description) if config.log_graphs else None
    log = Logger(gm, config.task_description, config.log_graphs_before_split)
    save_model_or_skip(model, config, f'{-config.extension_levels}_init')
    
    # log initial performance and descriptive statistics
    init_loss, init_accuracy = evaluate(model, test_loader, loss_fn, config.device)
    log.metrics({VAL_LOSS:init_loss, ACCURACY:init_accuracy, 'level': -config.extension_levels-1})
    log.commit()  # LOG BEFORE TRAINING

    # get the complete levels
    levels = list(range(-config.extension_levels, config.pruning_levels))
    pparams, pborder = config.param_trajectory[0], 0

    ####################
    ### Training 
    ####################
    for level, pruning_amount in tqdm(zip(levels, config.pruning_trajectory, strict=True), 'Pruning Levels', len(levels)):

        optim = build_optimizer(model, config)
        stopper = build_early_stopper(config)

        for epoch in tqdm(range(config.epochs), f'Training Level {level}', config.epochs):
            train_loss = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda)
            val_loss, val_acc = evaluate(model, test_loader, loss_fn, config.device)

            if config.log_every is not None and epoch % config.log_every == 0:
                log.metrics(
                    prefix='epoch/',
                    values={TRAIN_LOSS : train_loss.mean(), VAL_LOSS : val_loss, ACCURACY : val_acc.mean()},
                )
                log.commit()

            if config.loss_cutoff is not None and val_loss.mean().item() < config.loss_cutoff: break
            if stopper(val_loss.mean().item()): break

        save_model_or_skip(model, config, level)

        # log weights, biases, metrics at early stopping iteration
        log.metrics({
            TRAIN_LOSS : train_loss, VAL_LOSS : val_loss, ACCURACY : val_acc,
            'pparams' : pparams, 'level' : level, 'stop' : epoch, 'pborder' : pborder
        })
        
        if gm is not None:
            gm.update(model, level) 
            log.feature_categorization()
            log.splitting()
            if gm.iteration == gm.split_iteration: log.graphs()
            log.commit()

            if gm.untapped_potential < 0 and config.stop_on_degradation: 
                break

        # prune the model and reinit
        pborder = prune(pruning_amount)
        pparams -= pruning_amount
        if config.reinit: reinit(model)
    
    log.metrics({'pparams' : pparams, 'level' : config.pruning_levels, 'pborder' : pborder})

    ####################
    ### final finetuning
    ####################
    stopper = build_early_stopper(config)
    optim = build_optimizer(model, config)

    for epoch in tqdm(range(config.epochs), f'Final Finetuning', config.epochs):

        train_loss = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda)
        val_loss, val_acc = evaluate(model, test_loader, loss_fn, config.device)

        if config.log_every is not None and epoch % config.log_every == 0:
            log.metrics(
                prefix='epoch/',
                values={TRAIN_LOSS : train_loss.mean(), VAL_LOSS : val_loss, ACCURACY : val_acc.mean()}
            )
            log.commit()

    log.metrics({TRAIN_LOSS : train_loss, VAL_LOSS : val_loss, ACCURACY : val_acc})

    if gm is not None:
        gm.update(model, config.pruning_levels)
        log.feature_categorization()
        log.splitting()
        if gm.iteration == gm.split_iteration: log.graphs()
        log.summary()
        log.commit()

    save_model_or_skip(model, config, config.pruning_levels)
