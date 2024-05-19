from training.models import MultiTaskMultiClassMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines

# Ranking
# this worked amazingly well TOO!
pruning_levels, shape, batch_size, lr = 20, [392, 784, 784, 20], 128 ,0.001

run_config = Config(
    pipeline=Routines.imp,
    dataset=Datasets.RESIZE_FULL_FASHION_AND_MNIST,
    scaler=Scalers.StandardUnitVariance,
    model_class=MultiTaskMultiClassMLP,
    model_shape=shape,
    base_model_shape=shape,

    # training
    lr=lr,
    epochs=1000,
    batch_size=batch_size,
    optimizer="adam",

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # newly added
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,
    stop_on_degradation=True,
    stop_on_seperation=True,

    # pruning
    only_consider_out_features_for_degrading=True,
    pruning_method="magnitude",
    pruning_scope="global",
    prune_biases=False,
    prune_weights=True,
    pruning_target=100,
    pruning_levels=pruning_levels,
    reinit=True,
)
