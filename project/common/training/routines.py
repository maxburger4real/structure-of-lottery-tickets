'''high level training routines.'''
from enum import Enum
from tqdm import tqdm
from typing import Iterable
from common.config import Config
from common.factory import Factory
from common.training.utils import evaluate, train_and_evaluate

class Pipeline(Enum):
    vanilla = 'vanilla'
    imp = 'imp'
    bimt = 'bimt'

def start_routine(config: Config):
    # run the pipeline defined in the config
    match Pipeline[config.pipeline]:
        case Pipeline.vanilla:
            raise
            return vanilla.run(model, train_loader, test_loader, config)

        case Pipeline.imp:
            return imp(
                training_epochs=config.epochs,
                parameter_trajectory=config.param_trajectory,
                device=config.device,
                factory=Factory(config),
                first_level=-config.extension_levels,
                stop_when_model_degrades=config.stop_on_degradation
            )
    
        case Pipeline.bimt:
            raise
            return bimt.run(model, train_loader, test_loader, config)

def evaluate_graph(model, gm, level):
    if gm is None: return
    gm.update(model, level) 
    return gm.metrics()

def imp(
    training_epochs: int, 
    parameter_trajectory: Iterable,
    factory: Factory,
    device: str, 
    first_level: int = 0, 
    stop_when_model_degrades=True,
):
    # prepare model and data
    model = factory.make_model()
    train_loader, test_loader = factory.make_dataloaders()

    # prepare helpers
    pruner = factory.make_pruner(model)
    reinitializer = factory.make_renititializer(model)
    graph_manager = factory.make_graph_manager(model)
    logger = factory.make_logger()

    # test model before any training
    initial_metrics = evaluate(model, test_loader, device)
    logger.metrics(initial_metrics, prefix='performance/')
    logger.metrics(level=first_level-1,prefix='meta/')
    logger.commit()

    parameter_count = parameter_trajectory[0]
    for level, parameter_target in enumerate(parameter_trajectory, start=first_level):

        if level > first_level:
            pruning_amount = parameter_count - parameter_target
            parameter_count = parameter_target

            pruning_metrics = pruner(pruning_amount)
            logger.metrics(pruning_metrics, prefix='meta/')

            reinitializer(model)

        train_eval_metrics = train_and_evaluate(
            model, train_loader, test_loader, 
            optim=factory.make_optimizer(model), 
            stopper=factory.make_stopper(), 
            epochs=tqdm(range(training_epochs), f'Training Level {level-first_level+1}/{len(parameter_trajectory)}', training_epochs), 
            device=device, 
        )

        graph_metrics = evaluate_graph(model, graph_manager, level)

        if stop_when_model_degrades and graph_manager.is_degraded: break

        logger.metrics(graph_metrics, prefix='graph/')
        logger.metrics(train_eval_metrics, prefix='performance/')
        logger.metrics(
            pparams=parameter_count, 
            level=level, 
            prefix='meta/'
        )
        logger.commit()