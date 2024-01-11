'''A run config that is fast and splits.'''

from common.models import BinaryClassifierMLP
from common.config import Config
from common.datasets import Datasets
from common.constants import *

# verified seeds
data_seeds = [0,2]
model_seeds = SEEDS_123

run_config = Config(

    pruning_levels=2, # 6, 7, 8
    model_seed=model_seeds[0],  # splits with any of the seeds

    pipeline=Pipeline.imp,
    activation=RELU,

    dataset=Datasets.CIRCLES_MOONS,
    n_samples=1000,
    noise=0.1,

    model_shape=[4, 50, 50, 2],
    model_class=BinaryClassifierMLP,
    scaler=StandardUnitVariance,

    # training
    lr=0.5,
    optimizer=ADAM,
    epochs=20,
    batch_size=64,
    
    data_seed=data_seeds[0],
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
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL,
    init_strategy_biases = InitializationStrategy.ZERO,
)