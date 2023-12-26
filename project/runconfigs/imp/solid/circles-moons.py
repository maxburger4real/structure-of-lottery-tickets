from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''Good performance perfect split.''',
    pipeline=IMP,
    activation=RELU,
    loss_fn=BCE,    
    dataset=Datasets.CIRCLES_AND_MOONS.name,

    model_shape=[4, 320, 320, 2],
    model_class=MLP.__name__,
    scaler=StandardUnitVariance,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs= 3000,
    
    # seeds
    model_seed=7,
    data_seed=0,

    persist=False,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # yielded good results
    loss_cutoff=0.01,

    # pruning
    pruning_method=MAGNITUDE,
    pruning_scope=GLOBAL,
    prune_biases=False,
    prune_weights=True,
    pruning_target=50,
    pruning_levels=20,
    reinit=True,

    # newly added 
    init_strategy_weights = InitializationStrategy.DEFAULT.name,
    init_strategy_biases = InitializationStrategy.ZERO.name,
    n_samples=1000,
    noise=0.1
)
