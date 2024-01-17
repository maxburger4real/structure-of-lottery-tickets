'''
seed 0
sweep_id: p30nbq46

seed 123
sweep_id: r0b16unk

seed 01234 with larger shapes
ocfuu4ip

'p30nbq46', 'r0b16unk', 'ocfuu4ip'
'''

from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from utils.constants import *

description = '''
This experiment should show the splitting behaviour of the model shapes that resulted
from the model extension experiment, where each model was pruned a different number of times.
To compare it, in this experiment, the models are pruned to the same pruning target and with the same pruning levels.
pruning_rate is different in every run.
'''

# from exend-model-till-20 experiment
hidden_dims = [
    8, 10, 13, 16, 20,
    25, 31, 38, 47, 57, 
    70, 85, 104, 127, 154, 
    188, 229, 278, 337, 410
]

hidden_dims = [
    498, 604, 733, 890, 1080, 1310
]

shapes = [ [4] + [h]*2 + [2] for h in hidden_dims]

seeds = SEEDS_123 + [0]

sweep_config = {
    'method': 'grid', 
    'description' : description,
    'name': str(__name__),
    'parameters':{
        "model_shape" : { "values": shapes },
        "model_seed":  { "values": seeds },
    }
}


run_config = Config(

    # sweeped
    model_shape=[4, 8, 8, 2],
    model_seed=0,

    pipeline=Pipeline.imp,
    dataset=Datasets.CIRCLES_AND_MOONS,

    model_class=MultiTaskBinaryMLP,
    scaler=Scalers.StandardUnitVariance,

    # training
    lr=0.001,
    optimizer='adam',
    epochs= 3000,
    batch_size=64,
    
    # seeds
    data_seed=0,
    persist=False,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # pruning
    pruning_method='magnitude',
    pruning_scope='global',
    prune_biases=False,
    prune_weights=True,
    pruning_target=112,
    pruning_levels=20,
    reinit=True,

    # newly added 
    init_strategy_weights = Init.kaiming_normal,
    init_strategy_biases = Init.zero,
    n_samples=1000,
    noise=0.1
)