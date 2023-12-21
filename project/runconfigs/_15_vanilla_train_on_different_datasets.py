from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    description='''Single Task Testing the old init. how do the weights change over time ?''',
    pipeline=VANILLA,
    activation=RELU,
    loss_fn=BCE,
    dataset=MOONS_AND_CIRCLES,
    n_samples = 100,
    noise = 0.1,
    model_shape=[4, 40, 40, 2],
    model_class = MLP.__name__,

    # initialization strategy for the weights
    init_strategy_weights = InitializationStrategy.DEFAULT,
    init_mean = 0.0,
    init_std = 1.0,
    init_strategy_biases = InitializationStrategy.ZERO,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs=10000,

    # seeds
    model_seed=12, #[7, 9, 11]
    data_seed=0,

    persist=False,

    # early stop
    early_stopping=True,
    early_stop_patience=100,
    early_stop_delta=0.001,
)
