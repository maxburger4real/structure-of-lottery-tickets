'''
sweep_id : wpcowdl5
'''

from common.models import BinaryClassifierMLP
from common.config import Config
from common.constants import *
from common.datasets import Datasets


description = '''
This experiment should show behaviour of splitting when starting with a small network,
namely [4, 8, 8, 2] and extending it up to 20 times with a pruning rate of 0.32

This pruning rate is derived from a well working experiment with a network of size 
[4, 320, 320, 2], which is pruned 20 times to 50 parameters. It yielded good splitting behaviour and had 
this pruning rate.
'''

extension_levels = list(range(20))
seeds = [0]

# TODO: seeds 1,2

sweep_config = {
    'method': 'grid', 
    'description': description,
    'name': str(__name__),
    'parameters':{
        "extension_levels" : { "values": extension_levels },
        "model_seed":  { "values": seeds },
    }
}


run_config = Config(

    # sweeped
    model_seed=8,
    extension_levels=0,

    model_shape=[4, 8, 8, 2],

    pipeline=Pipeline.imp.name,
    activation=RELU,
    dataset=Datasets.CIRCLES_AND_MOONS.name,

    model_class=BinaryClassifierMLP.__name__,
    scaler=StandardUnitVariance,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs= 3000,
    batch_size=64,
    
    # seeds
    data_seed=0,

    persist=False,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # pruning
    pruning_method=MAGNITUDE,
    pruning_scope=GLOBAL,
    prune_biases=False,
    prune_weights=True,
    pruning_rate=0.32,
    pruning_levels=0,
    reinit=True,

    # newly added 
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL.name,
    init_strategy_biases = InitializationStrategy.ZERO.name,
    n_samples=1000,
    noise=0.1
)
