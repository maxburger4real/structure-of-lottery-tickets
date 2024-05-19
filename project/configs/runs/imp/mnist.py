from training.models import MultiTaskMultiClassMLP, SingleTaskMultiClassMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines

pruning_levels, shape = 18, [784, 300, 100, 10]

run_config = Config(
    model_seed=0, # 0 nice
    pipeline=Routines.imp,
    dataset=Datasets.MNIST,
    scaler=Scalers.StandardUnitVariance,
    model_class=SingleTaskMultiClassMLP,
    model_shape=shape,
    base_model_shape=shape,

    # training
    lr=0.001,
    epochs=1000,
    batch_size=128,
    optimizer="adam",

    # early stop
    early_stop_patience=10,
    early_stop_delta=0.0,

    # newly added
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,
    stop_on_degradation=False,
    stop_on_seperation=False,

    # pruning
    pruning_method="magnitude",
    pruning_scope="global",
    prune_biases=False,
    prune_weights=True,
    pruning_target=600,
    pruning_levels=pruning_levels,
    reinit=True,
)
