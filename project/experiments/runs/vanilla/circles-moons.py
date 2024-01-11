from common.models import MultiTaskBinaryMLP
from common.config import Config
from common.datasets import Datasets
from common.constants import *

run_config = Config(
    pipeline=Pipeline.vanilla,
    dataset=Datasets.CIRCLES_AND_MOONS,

    model_shape='320_320',
    model_class=MultiTaskBinaryMLP,
    scaler=StandardUnitVariance,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs= 3000,
    batch_size=64,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # newly added 
    init_strategy_weights = InitializationStrategy.DEFAULT,
    init_strategy_biases = InitializationStrategy.ZERO,
    n_samples=1000,
    noise=0.1
)
