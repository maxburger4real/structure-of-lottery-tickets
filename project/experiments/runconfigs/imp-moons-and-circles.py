from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''A Well working setup for Splitting networks.''',

    # dataset
    dataset=Datasets.MOONS_AND_CIRCLES.name,
    n_samples = 1000,
    noise = 0.1,

    # initialization strategy for the weights
    model_seed=12,
    model_class = MLP.__name__,
    model_shape = [4, 20, 20, 2],
    init_strategy_weights = InitializationStrategy.DEFAULT,
    init_strategy_biases = InitializationStrategy.DEFAULT,
    activation=RELU,
    persist=False,

    # training
    pipeline=IMP,
    lr=0.001,
    optimizer=ADAM,
    epochs=4000,
    loss_fn=BCE,

    # early stop
    early_stop_patience=10,
    early_stop_delta=0.0001,

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=True,
    prune_weights=True,
    pruning_target=50,
    pruning_levels=10,
    reinit=True,

    # logging
    log_graph_statistics = True
)