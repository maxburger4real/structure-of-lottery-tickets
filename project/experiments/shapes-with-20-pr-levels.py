'''
sweep_id: p30nbq46
'''

from common.models import MLP
from common.config import Config
from common.constants import *

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

shapes = [ [4] + [h]*2 + [2] for h in hidden_dims]

seeds = [0]
# TODO: seeds 1,2

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

    pipeline=IMP,
    activation=RELU,
    loss_fn=BCE,    
    dataset=Datasets.CIRCLES_AND_MOONS.name,

    model_class=MLP.__name__,
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
    pruning_target=112,
    pruning_levels=20,
    reinit=True,

    # newly added 
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL.name,
    init_strategy_biases = InitializationStrategy.ZERO.name,
    n_samples=1000,
    noise=0.1
)