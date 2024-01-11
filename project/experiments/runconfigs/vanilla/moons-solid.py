from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=VANILLA,
    loss_fn=BCE,
    dataset=Datasets.MOONS.name,
    n_samples=100,
    noise=0.1,
        
    model_class = MLP.__name__,
    activation=RELU,
    # 10/10
    # model_shape=[2, 50, 1],

    # 5/10
    # model_shape=[2,13,13,1],

    # 8/10
    model_shape=[2,16,16,1],

    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL,
    init_strategy_biases = InitializationStrategy.ZERO,
    
    # training
    lr=0.001,
    optimizer=ADAM,
    epochs=10000,

    # seeds
    model_seed=0,
    persist=False,
)
