from tqdm import tqdm
from collections import defaultdict
from common.log import Logger
from common import factory
from common.config import Config
from common.nxutils import GraphManager
from common.persistance import save_model_or_skip
from common.pruning import build_pruning_func, build_reinit_func
from common.training.utils import build_early_stopper, build_optimizer, update, evaluate
from common.constants import *

def start_routine(config: Config):
    # run the pipeline defined in the config
    match Pipeline[config.pipeline]:
        case Pipeline.vanilla:
            raise
            return vanilla.run(model, train_loader, test_loader, config)

        case Pipeline.imp:
            model = factory.make_model(config)
            train_loader, test_loader = factory.make_dataloaders(config)
            pruner = build_pruning_func(model, config)
            reinitializer = build_reinit_func(model, config)
            gm = GraphManager(model, config.model_shape, config.task_description) if config.log_graphs else None
            logger = Logger(gm, config.task_description)

            return imp(model, train_loader, test_loader, pruner, reinitializer, gm, logger, config)
    
        case Pipeline.bimt:
            raise
            return bimt.run(model, train_loader, test_loader, config)


def train_and_evaluate(model, train_loader, test_loader, optim, logger: Logger, stopper, epochs, device, log_every):

    metrics = defaultdict(list)

    for epoch in epochs:

        train_loss = update(model, train_loader, optim, device)
        val_loss, val_acc = evaluate(model, test_loader, device)

        metrics[TRAIN_LOSS].append(train_loss.mean().item())
        metrics[VAL_LOSS].append(val_loss.mean().item())
        metrics[ACCURACY].append(val_acc.mean().item())

        if stopper(metrics[VAL_LOSS][-1]):
            break

        if log_every is not None and epoch % log_every == 0 and epoch != epochs[-1]:
            latest = {k: v[-1] for k, v in metrics.items()}
            logger.metrics(latest, prefix='epoch/')
            logger.commit()

    logger.metrics({k: v[-1] for k, v in metrics.items()})
    return metrics

#def imp(model, train_loader, test_loader, config: Config):
def imp(model, train_loader, test_loader, prune, reinit, gm: GraphManager, log: Logger, config: Config):

    # save_model_or_skip(model, config, f'{-config.extension_levels}_init')
    
    # log initial performance and descriptive statistics
    init_loss, init_accuracy = evaluate(model, test_loader, config.device)
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

        epochs = tqdm(range(config.epochs), f'Training Level {level}', config.epochs)
        metrics = train_and_evaluate(model, train_loader, test_loader, optim, log, stopper, epochs, config.device, config.log_every)

        save_model_or_skip(model, config, level)

        log.metrics(dict(pparams=pparams, level=level, stop=len(list(metrics.values())[0]), pborder=pborder))

        
        if gm is not None:
            gm.update(model, level) 
            log.feature_categorization()
            log.splitting()
            if gm.iteration == gm.split_iteration: 
                log.graphs()
                log.metrics({'split-acc':val_acc, 'split-loss': val_loss, 'split-pparams':pparams})
            log.commit()

            if gm.untapped_potential < 0 and config.stop_on_degradation: 
                break

        # prune the model and reinit
        pborder = prune(pruning_amount)
        pparams -= pruning_amount
        reinit(model)
    
    log.metrics({'pparams' : pparams, 'level' : config.pruning_levels, 'pborder' : pborder})

    ####################
    ### final finetuning
    ####################
    stopper = build_early_stopper(config)
    optim = build_optimizer(model, config)

    for epoch in tqdm(range(config.epochs), f'Final Finetuning', config.epochs):
        train_loss = update(model, train_loader, optim, config.device, config.l1_lambda)
        val_loss, val_acc = evaluate(model, test_loader, config.device)

        if config.log_every is not None and epoch % config.log_every == 0:
            log.metrics(
                prefix='epoch/',
                values={TRAIN_LOSS : train_loss.mean(), VAL_LOSS : val_loss, ACCURACY : val_acc.mean()}
            )
            log.commit()
            
        if stopper(val_loss.mean().item()): break

    log.metrics({TRAIN_LOSS : train_loss, VAL_LOSS : val_loss, ACCURACY : val_acc})

    if gm is not None:
        gm.update(model, config.pruning_levels)
        log.feature_categorization()
        log.splitting()

        if gm.iteration == gm.split_iteration:
            log.graphs()
            log.metrics({'split-acc':val_acc, 'split-loss': val_loss, 'split-pparams':pparams})

        log.summary()
        log.commit()

    save_model_or_skip(model, config, config.pruning_levels)
