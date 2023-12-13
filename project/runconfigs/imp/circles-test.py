from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=IMP,
    loss_fn=BCE,
    dataset=Datasets.CIRCLES.name,
    n_samples=100,
    noise=0.1,
    
    model_class = MLP.__name__,
    activation=RELU,

    # 10/10
    model_shape=[2, 50, 1],

    # 10/10
    # model_shape=[2, 13, 13, 1],

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
    
    #pruning_target=50,
    pruning_target=32,

    pruning_levels=10,
    reinit=True,

    # logging
    log_graph_statistics = True
)
