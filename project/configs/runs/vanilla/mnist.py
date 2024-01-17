from training.models import SingleTaskMultiClassMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines

run_config = Config(
    pipeline=Routines.vanilla,
    dataset=Datasets.MNIST,
    scaler=Scalers.StandardUnitVariance,
    model_class=SingleTaskMultiClassMLP,
    model_shape="300_100",
    # training
    lr=0.001,
    epochs=100,
    batch_size=256,
    optimizer="adam",
    # early stop
    early_stop_patience=10,
    early_stop_delta=0.0,
    # newly added
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,
)
