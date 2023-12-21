from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''A Well working setup for Splitting networks.''',

    # dataset
    dataset=Datasets.MOONS.name,
    n_samples=100,
    noise=0.1,

    # initialization strategy for the weights
    model_seed=123,
    model_class = MLP.__name__,
    model_shape = [2, 50, 1],
    init_strategy_weights = InitializationStrategy.DEFAULT,
    init_strategy_biases = InitializationStrategy.ZERO,
    activation=RELU,
    persist=False,

    # training
    pipeline=IMP,
    lr=0.001,
    optimizer=ADAM,
    epochs=4000,
    loss_fn=BCE,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.01,

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=False,
    prune_weights=True,
    #pruning_target=50,
    pruning_levels=0,
    pruning_rate=0.2,
    extension_levels=9,
    reinit=True,

    # logging
    log_graph_statistics = True
)