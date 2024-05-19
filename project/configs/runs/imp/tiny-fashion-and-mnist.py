from training.models import MultiTaskMultiClassMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines


# AMAzing This actually seperates the model.
# this model learns to 98.375% accuracy.

shape = [288,144,72,4] # didnt split with 12 levels
shape = [288,200,100,4] # this didnt split
shape = [288,288,200,4] # this did split
shape = [288,200,200,4] # nope

# Ranking
pruning_levels, shape, batch_size = 15, [288,144,144,4], 64  # split at 154, acc .99
pruning_levels, shape, batch_size = 15, [288,144,144,4], 128  #  split at 154, acc .973
pruning_levels, shape, batch_size = 16, [288,200,200,4], 64  # split at 154, acc .99

pruning_levels, shape, batch_size = 15, [288,144,144,4], 256  #  split at 100, acc .988
pruning_levels, shape, batch_size = 20, [288,144,144,4], 64  #  split at 100, acc .988

# # TOP:
pruning_levels, shape, batch_size = 15, [288,144,144,4], 64  # split at 154, acc .99


run_config = Config(
    pipeline=Routines.imp,
    dataset=Datasets.TINY_FASHION_AND_MNIST,
    scaler=Scalers.StandardUnitVariance,
    model_class=MultiTaskMultiClassMLP,
    model_shape=shape,
    base_model_shape=shape,

    # training
    lr=0.001,
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
