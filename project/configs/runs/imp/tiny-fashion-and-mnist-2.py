from training.models import MultiTaskMultiClassMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines

# it splits but 


# AMAzing This actually seperates the model.
# this model learns to 98.375% accuracy.



# Ranking
# BEST
pruning_levels, shape, batch_size, lr = 15, [392, 300, 300, 4], 64 ,0.001 # split at 160, acc .99

pruning_levels, shape, batch_size, lr = 15, [392, 196, 196, 4], 64 ,0.001 # split at 160, acc .99
pruning_levels, shape, batch_size, lr = 25, [392, 196, 196, 4], 64, 0.001  # not so good

pruning_levels, shape, batch_size, lr = 15, [392, 196, 196, 4], 64, 0.0005  # split at 160, acc .99
pruning_levels, shape, batch_size, lr = 15, [392, 196, 196, 4], 64, 0.0001  # split at 160, acc .99

# in progress
pruning_levels, shape, batch_size, lr = 15, [392, 392, 392, 4], 64 ,0.001 # split at 160, acc .99
pruning_levels, shape, batch_size, lr = 15, [392, 784, 784, 4], 64 ,0.001 # split at 160, acc .99

run_config = Config(
    pipeline=Routines.imp,
    dataset=Datasets.TINY_FASHION_AND_MNIST_2,
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
