from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''Single Task Testing the old init. how do the weights change over time ?''',

    # dataset
    dataset=MOONS_AND_CIRCLES,
    n_samples = 800,
    noise = 0.1,

    # initialization strategy for the weights
    model_seed=12, #[7, 9, 11]
    model_class = MLP.__name__,
    model_shape = [4, 40, 40, 2],
    init_strategy_weights = InitializationStrategy.DEFAULT,
    init_strategy_biases = InitializationStrategy.ZERO,
    activation=RELU,
    persist=False,

    # training
    pipeline=IMP,
    lr=0.001,
    optimizer=ADAM,
    epochs=100,
    loss_fn=BCE,

    # early stop
    early_stop_patience=10,
    early_stop_delta=0.001,

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=True,
    prune_weights=True,
    pruning_target=100,
    pruning_levels=5,
    reinit=True,

    # logging
    log_graph_statistics = False,
)
