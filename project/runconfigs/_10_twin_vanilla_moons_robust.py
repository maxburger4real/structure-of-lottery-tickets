from common.models import SimpleMLP
from common.config import Config
from common.constants import *

m = 2

run_config = Config(
    pipeline=VANILLA,
    activation=RELU,
    loss_fn= BCE,
    dataset=MULTI_MOONS,
    num_concat_datasets=m,
    
    model_shape=[2*m, 20, 20, 1*m],
    model_class = SimpleMLP.__name__,

    # training
    lr=0.001,
    optimizer=ADAM,
    training_epochs=5000,

    # seeds
    model_seed=2,
    data_seed=0,


    persist=False,

    # early stop
    early_stopping=True,
    early_stop_patience=1,
    early_stop_delta= -0.1,
)
