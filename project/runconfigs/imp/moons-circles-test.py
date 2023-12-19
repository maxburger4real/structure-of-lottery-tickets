from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=IMP,
    loss_fn=BCE,
    dataset=Datasets.MOONS_AND_CIRCLES.name,
    n_samples=600,
    noise=0.1,
    
    model_class = MLP.__name__,
    activation=RELU,
    model_shape=[4, 100, 2],
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL,
    init_strategy_biases = InitializationStrategy.ZERO,
    
    # training
    lr=0.001,
    optimizer=ADAM,
    epochs=10000,

    # seeds
    model_seed=0,
    persist=False,

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=False,
    prune_weights=True,
    
    pruning_target=30,
    pruning_levels=10,
    extension_levels=5,
    reinit=True,

    # logging
    log_graph_statistics = True
)