from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''Trying to find a configuration where it splits''',

    # dataset
    dataset=MOONS_AND_CIRCLES,
    #dataset=MULTI_MOONS,
    n_samples = 800,
    noise = 0.1,

    # initialization strategy for the weights
    model_seed=12, #[7, 9, 11]
    model_class = MLP.__name__,
    model_shape = [4, 80, 80, 2],
    init_strategy_weights = InitializationStrategy.DEFAULT,
    init_strategy_biases = InitializationStrategy.DEFAULT,
    activation=RELU,
    persist=True,

    # training
    pipeline=IMP,
    lr=0.001,
    optimizer=ADAM,
    epochs=3000,
    loss_fn=BCE,

    # early stop
    early_stop_patience=10,
    early_stop_delta=0.001,

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=True,
    prune_weights=True,
    pruning_target=50,
    pruning_levels=20,
    reinit=True,

    # logging
    log_graph_statistics = True,
)
