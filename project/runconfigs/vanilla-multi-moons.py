from common.models import SimpleMLP
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=VANILLA,
    activation=RELU,
    loss_fn= BCE,
    dataset=MULTI_MOONS,
    
    model_shape=[4, 9, 9, 2],
    model_class = SimpleMLP.__name__,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs=30000,

    # seeds
    model_seed=2,

    persist=False,

    # early stop
    early_stop_patience=20,
    early_stop_delta=0.0,

)
