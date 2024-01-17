'''A run config that is fast and splits.'''

from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from utils.constants import *

# verified seeds
data_seeds = [0,2]
model_seeds = SEEDS_123

run_config = Config(

    pruning_levels=6, # 6, 7, 8
    model_seed=model_seeds[0],  # splits with any of the seeds

    pipeline=Pipeline.imp,
    

    dataset=Datasets.CIRCLES_MOONS,
    n_samples=200,
    noise=0.1,

    model_shape='30_30',
    model_class=MultiTaskBinaryMLP,
    scaler=Scalers.StandardUnitVariance,

    # training
    lr=0.001,
    optimizer='adam',
    epochs= 128,
    batch_size=64,
    
    data_seed=data_seeds[0],
    early_stop_patience=30,
    early_stop_delta=0.0,

    # pruning
    pruning_method='magnitude',
    pruning_scope='global',
    prune_biases=False,
    prune_weights=True,

    pruning_target=112,
    reinit=True,
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,
)