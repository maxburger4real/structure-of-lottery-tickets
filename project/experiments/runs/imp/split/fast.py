from common.models import MLP
from common.config import Config
from common.constants import *


run_config = Config(

    pruning_levels=6, # 6, 7, 8
    model_seed=SEEDS_123[0],  # splits with any of the seeds

    pipeline=IMP,
    activation=RELU,
    loss_fn=BCE,    

    dataset=Datasets.CIRCLES_AND_MOONS.name,
    n_samples=1000,
    noise=0.1,

    model_shape=[4, 410, 410, 2],
    model_class=MLP.__name__,
    scaler=StandardUnitVariance,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs= 3000,
    batch_size=64,
    
    data_seed=0,
    persist=False,
    early_stop_patience=30,
    early_stop_delta=0.0,

    # pruning
    pruning_method=MAGNITUDE,
    pruning_scope=GLOBAL,
    prune_biases=False,
    prune_weights=True,

    pruning_target=112,  # 4*8 + 8*8 + 8*2  --> [4,8,8,2]
    reinit=True,
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL.name,
    init_strategy_biases = InitializationStrategy.ZERO.name,
)