from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines

run_config = Config(
    pipeline=Routines.vanilla,
    dataset=Datasets.CIRCLES_AND_MOONS,
    model_shape="320_320",
    model_class=MultiTaskBinaryMLP,
    scaler=Scalers.StandardUnitVariance,
    # training
    lr=0.001,
    optimizer="adam",
    epochs=3000,
    batch_size=64,
    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,
    # newly added
    init_strategy_weights=Init.DEFAULT,
    init_strategy_biases=Init.zero,
    n_samples=1000,
    noise=0.1,
)
