'''
sweep_id : ix4onq8c
9cclxb43

'''


from utils.constants import *
from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init


description = '''
With this experiment, the plan is to see how many pruning levels are needed for a split.
The same model will be pruned to the same target, but with a different amount of pruning levels. 
Obviously, the pruning_rate will be different for each run. More levles, lower pruning_rate and vice
versa.
'''


levels = list(range(21, 26))
seeds = SEEDS_123
sweep_config = {
    'method': 'grid', 
    'name': str(__name__),
    'description' : description,
    'parameters':{
        "model_seed":  { "values": seeds },
        "pruning_levels":  { "values": levels },
    }
}


run_config = Config(
    # Sweeped
    pruning_levels=20,
    model_seed=8,

    pipeline=Pipeline.imp,
    

    dataset=Datasets.CIRCLES_AND_MOONS,
    n_samples=1000,
    noise=0.1,

    model_shape=[4, 410, 410, 2],
    model_class=MultiTaskBinaryMLP,
    scaler=Scalers.StandardUnitVariance,

    # training
    lr=0.001,
    optimizer='adam',
    epochs= 3000,
    batch_size=64,
    
    data_seed=0,
    persist=False,
    early_stop_patience=30,
    early_stop_delta=0.0,

    # pruning
    pruning_method='magnitude',
    pruning_scope='global',
    prune_biases=False,
    prune_weights=True,

    pruning_target=112,  # 4*8 + 8*8 + 8*2  --> [4,8,8,2]
    reinit=True,
    init_strategy_weights = Init.kaiming_normal,
    init_strategy_biases = Init.zero,
)
