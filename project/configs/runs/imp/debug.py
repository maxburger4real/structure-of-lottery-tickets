"""A run config that is fast and splits."""

from training.models import MultiTaskBinaryMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines

# verified seeds
data_seeds = [0, 2]
model_seeds = [1, 2, 3]

run_config = Config(
    pruning_levels=2,  # 6, 7, 8
    model_seed=model_seeds[0],  # splits with any of the seeds
    pipeline=Routines.imp,
    dataset=Datasets.CIRCLES_MOONS,
    n_samples=1000,
    noise=0.1,
    model_shape="50_50",
    model_class=MultiTaskBinaryMLP,
    scaler=Scalers.StandardUnitVariance,
    # training
    lr=0.5,
    optimizer="adam",
    epochs=20,
    batch_size=64,
    data_seed=data_seeds[0],
    persist=False,
    early_stop_patience=30,
    early_stop_delta=0.0,
    # pruning
    pruning_method="magnitude",
    pruning_scope="global",
    prune_biases=False,
    prune_weights=True,
    pruning_target=112,  # 4*8 + 8*8 + 8*2  --> [4,8,8,2]
    reinit=True,
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,
)
