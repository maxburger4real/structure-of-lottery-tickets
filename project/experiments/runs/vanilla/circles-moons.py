from common.models import BinaryClassifierMLP
from common.config import Config
from common.datasets import Datasets
from common.constants import *

run_config = Config(
    description='''Early stopping works well.''',
    pipeline=Pipeline.vanilla.name,
    activation=RELU,
    dataset=Datasets.CIRCLES_AND_MOONS.name,

    model_shape=[4, 320, 320, 2],
    model_class=BinaryClassifierMLP.__name__,
    scaler=StandardUnitVariance,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs= 3000,
    batch_size=64,
    
    # seeds
    model_seed=8,
    data_seed=0,
    persist=False,

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # newly added 
    init_strategy_weights = InitializationStrategy.DEFAULT.name,
    init_strategy_biases = InitializationStrategy.ZERO.name,
    n_samples=1000,
    noise=0.1
)
