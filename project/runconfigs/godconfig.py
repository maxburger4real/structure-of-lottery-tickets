from common.models import MLP
from common.config import Config
from common.constants import *

m = 2

run_config = Config(
    description='''God Formula.''',
    pipeline=IMP,
    activation=RELU,
    loss_fn= BCE,    
    dataset=Datasets.OLD_MOONS.name,  # works reliably
    
    model_shape=[m*2, 40, 40, m],
    model_class = MLP.__name__,

    # noise = 0.1

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs= 3000,
    
    # seeds
    model_seed=7, # good seeds : 2
    data_seed=0,  # split old but not new
    # data_seed=1, # splits old but not new
    # data_seed=0, # splits

    persist=False,

    # early stop
    early_stop_patience=10,
    early_stop_delta=0.0,

    # yielded good results
    loss_cutoff=0.01,  

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=True,
    prune_weights=True,
    pruning_target=50,
    pruning_levels=20,
    reinit=True,

    # newly added 
    init_strategy_weights = InitializationStrategy.DEFAULT.name,
    init_strategy_biases = InitializationStrategy.DEFAULT.name,
    n_samples=1000,
    #noise=0.0,
    noise=0.0,
    init_mean=None,
    init_std=None,
)
