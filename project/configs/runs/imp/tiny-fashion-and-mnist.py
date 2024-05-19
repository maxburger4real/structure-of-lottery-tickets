from training.models import MultiTaskMultiClassMLP
from training.config import Config
from training.datasets import Datasets, Scalers
from training.models import Init
from training.routines import Routines


# AMAzing This actually seperates the model.
# this model learns to 98.375% accuracy.

run_config = Config(
    pipeline=Routines.imp,
    dataset=Datasets.TINY_FASHION_AND_MNIST,
    scaler=Scalers.StandardUnitVariance,
    model_class=MultiTaskMultiClassMLP,
    model_shape=[288,100,100,4],
    base_model_shape=[288,100,100,4],

    # training
    lr=0.001,
    epochs=1000,
    batch_size=256,
    optimizer="adam",

    # early stop
    early_stop_patience=30,
    early_stop_delta=0.0,

    # newly added
    init_strategy_weights=Init.kaiming_normal,
    init_strategy_biases=Init.zero,

    # pruning
    only_consider_out_features_for_degrading = True,
    pruning_method="magnitude",
    pruning_scope="global",
    prune_biases=False,
    prune_weights=True,
    pruning_target=100,
    pruning_levels=20,
    reinit=True,
)
