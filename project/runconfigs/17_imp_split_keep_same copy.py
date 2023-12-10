from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''A Well working setup for Splitting networks.''',

    # dataset
    dataset=MOONS_AND_CIRCLES,
    n_samples = 800,
    noise = 0.1,
    device=

    # initialization strategy for the weights
    model_seed=12,
    model_class = MLP.__name__,
    model_shape = [4, 40, 40, 2],
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
    log_graph_statistics = True
)
