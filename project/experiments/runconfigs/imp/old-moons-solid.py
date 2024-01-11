from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''God Formula.''',
    pipeline=IMP,
    activation=RELU,
    loss_fn= BCE,    
    dataset=Datasets.OLD_MOONS.name,  # works reliably
    
    model_shape=[4, 40, 40, 2],
    model_class = MLP.__name__,
    scaler=None,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs= 3000,
    
    # seeds
    #model_seed=7, # good seeds : 2
    model_seed=7, # good seeds : 2
    data_seed=0,  # split old but not new

    persist=False,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # yielded good results
    loss_cutoff=0.01,  

    # pruning
    pruning_method=MAGNITUDE,
    pruning_scope=GLOBAL,
    prune_biases=True,
    prune_weights=True,
    pruning_target=50,
    pruning_levels=20,
    reinit=True,

    # newly added 
    init_strategy_weights = InitializationStrategy.DEFAULT.name,
    init_strategy_biases = InitializationStrategy.DEFAULT.name,
    n_samples=1000,
    noise=0.1
)
