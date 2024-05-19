from training.models import MultiTaskMultiClassMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines

pruning_levels, shape, batch_size, lr = 20, [1568, 784, 392, 20], 512 ,0.001
pruning_levels, shape, batch_size, lr = 18, [1568, 784, 784, 20], 128 ,0.001

run_config = Config(
    model_seed=0, # 0 nice
    pipeline=Routines.imp,
    dataset=Datasets.LARGEST_FASHION_AND_MNIST,
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
    early_stop_patience=10,
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
    pruning_target=600,
    pruning_levels=pruning_levels,
    reinit=True,
)
