from common.models import InitMLP
from common.config import Config
from common.constants import *


run_config = Config(
    pipeline=IMP,
    activation=RELU,
    loss_fn= BCE,
    dataset=Datasets.MOONS_AND_CIRCLES.name,
    n_samples=800,
    model_shape=[4, 40, 40, 2],
    model_class = InitMLP.__name__,

    # training
    lr=0.1,
    optimizer=ADAM,
    epochs=100,

    # seeds
    model_seed=42,
    persist=False,

    # early stop
    early_stop_patience=20,
    early_stop_delta=0.0,

    extension_levels=2,

    # pruning
    prune_biases=True,
    prune_weights=True,
    pruning_target=80,
    pruning_levels=10,
    pruning_method=MAGNITUDE,
    reinit=True,

    log_graph_statistics=True,
)
