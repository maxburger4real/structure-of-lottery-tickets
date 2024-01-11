from common.models import SingleTaskMultiClassMLP
from common.config import Config
from common.datasets import Datasets
from common.constants import *

run_config = Config(
    pipeline=Pipeline.vanilla,
    dataset=Datasets.MNIST,
    scaler=StandardUnitVariance,
    model_class=SingleTaskMultiClassMLP,
    model_shape='300_100',

    # training
    lr=0.001,
    epochs=100,
    batch_size=256,
    optimizer=ADAM,

    # early stop
    early_stop_patience=10,
    early_stop_delta=0.0,

    # newly added 
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL,
    init_strategy_biases = InitializationStrategy.ZERO,
)
