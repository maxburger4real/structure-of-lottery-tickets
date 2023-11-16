from common.models import SimpleMLP
from common.config import Config
from common.constants import *

m = 1

run_config = Config(
    pipeline=VANILLA,
    activation=RELU,
    loss_fn= BCE,
    dataset=CONCAT_MOONS,
    num_concat_datasets=m,
    
    model_shape=[2*m, 9, 9, 1*m],
    model_class = SimpleMLP.__name__,

    # training
    lr=0.001,
    optimizer=ADAM,
    training_epochs=30000,

    # seeds
    model_seed=2,
    data_seed=0,

    persist=False,

    # early stop
    early_stopping=True,
    early_stop_patience=20,
    early_stop_delta=0.0,

)
