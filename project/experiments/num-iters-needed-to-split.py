'''
sweep_id : ix4onq8c
'''


from common.constants import *
from common.models import MLP
from common.config import Config

description = '''
With this experiment, the plan is to see how many pruning levels are needed for a split.
The same model will be pruned to the same target, but with a different amount of pruning levels. 
Obviously, the pruning_rate will be different for each run. More levles, lower pruning_rate and vice
versa.
'''


levels = list(range(1, 21))
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
