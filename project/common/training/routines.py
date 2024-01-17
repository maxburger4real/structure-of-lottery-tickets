from tqdm import tqdm
from common.log import Logger
from common import factory as f
from common.config import Config
from common.factory import Factory
from common.nxutils import GraphManager
from common.pruning import build_pruning_func, build_reinit_func
from common.training.utils import build_early_stopper, build_optimizer, evaluate, train_and_evaluate
from common.constants import *

def start_routine(config: Config):
    # run the pipeline defined in the config
    match Pipeline[config.pipeline]:
        case Pipeline.vanilla:
            raise
            return vanilla.run(model, train_loader, test_loader, config)

        case Pipeline.imp:
            factory = Factory(config)
            model = factory.make_model()
            train_loader, test_loader = factory.make_dataloaders()
            pruner = factory.make_pruner(model)
            reinitializer = factory.make_renititializer(model)
            gm = factory.make_graph_manager(model)
            logger = factory.make_logger()

            return imp(model, train_loader, test_loader, pruner, reinitializer, gm, logger, config, Factory(config))
    
        case Pipeline.bimt:
            raise
            return bimt.run(model, train_loader, test_loader, config)

def evaluate_graph(model, gm, level, logger):
    if gm is None: return

    gm.update(model, level) 
    metrics = gm.metrics()
    logger.metrics(metrics)

def imp(model, train_loader, test_loader, prune, reinit, gm: GraphManager, log: Logger, config: Config, factory: Factory):

    #model = factory.make_model()

    levels = list(range(-config.extension_levels, config.pruning_levels+1))

    init_loss, init_accuracy = evaluate(model, test_loader, config.device)
    log.metrics({VAL_LOSS:init_loss, ACCURACY:init_accuracy, 'level': levels[0]-1})
    log.commit()  # LOG BEFORE TRAINING

    # get the complete levels
    pparams, pborder = config.param_trajectory[0], 0

    for i, level in tqdm(enumerate(levels, start=-1), 'Pruning Levels', len(levels)):

        log.metrics(dict(pparams=pparams, level=level, pborder=pborder))

        if i != -1:
            amount = config.pruning_trajectory[i]
            pborder = prune(amount)
            pparams -= amount
            reinit(model)

        train_and_evaluate(
            model, train_loader, test_loader, 
            optim=build_optimizer(model, config), 
            logger=log, 
            stopper=build_early_stopper(config), 
            epochs=tqdm(range(config.epochs), f'Training Level {level+1}/{len(levels)}', config.epochs), 
            device=config.device, 
            log_every=config.log_every
        )

        evaluate_graph(model, gm, level, log)

        if gm.untapped_potential < 0 and config.stop_on_degradation: 
            break

        log.commit()