from common.models import MultiClassClassifierMLP
from common.config import Config
from common.datasets import Datasets
from common.constants import *

run_config = Config(
    pipeline=Pipeline.vanilla,

    dataset=Datasets.MNIST,
    scaler=StandardUnitVariance,

    model_class=MultiClassClassifierMLP,
    model_shape=[784, 300, 100, 10],

    # training
    lr=0.1,
    epochs=40,
    batch_size=128,
    optimizer=ADAM,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # newly added 
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL,
    init_strategy_biases = InitializationStrategy.ZERO,
)
