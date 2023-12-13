from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=VANILLA,
    loss_fn=BCE,
    dataset=MOONS_AND_CIRCLES,
    n_samples=600,  # very important
    noise=0.1,
        
    model_class = MLP.__name__,
    activation=RELU,

    model_shape=[4,100,2],

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
