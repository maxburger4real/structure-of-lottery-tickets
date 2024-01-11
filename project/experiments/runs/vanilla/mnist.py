from common.models import MultiClassClassifierMLP
from common.config import Config
from common.datasets import Datasets
from common.constants import *

run_config = Config(
    pipeline=Pipeline.vanilla,
    activation=RELU,
    dataset=Datasets.MNIST,

    model_shape=[784, 300, 100, 10],
    model_class=MultiClassClassifierMLP,
    scaler=StandardUnitVariance,

    # training
    lr=0.1,
    epochs=40,
    batch_size=128,
    optimizer=ADAM,
    
    # seeds
    model_seed=8,
    persist=False,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # newly added 
    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL,
    init_strategy_biases = InitializationStrategy.ZERO,
)
