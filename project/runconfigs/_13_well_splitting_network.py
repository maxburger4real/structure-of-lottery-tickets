from common.models import SimpleMLP, InitMLP
from common.config import Config
from common.constants import *

m=2
run_config = Config(
    description='''With new initialization technique''',
    pipeline=IMP,
    activation=RELU,
    loss_fn= BCE,
    dataset=MULTI_MOONS,
    num_concat_datasets=m,
    
    model_shape=[m*2, 80, 80, m],
    model_class = InitMLP.__name__,

    # training
    lr=0.001,
    optimizer=ADAM,
    epochs=3000,

    # seeds
    model_seed=11, #[7, 9, 11]
    data_seed=0,

    persist=True,

    # early stop
    early_stopping=True,
    early_stop_patience=10,
    early_stop_delta=0.0,
    # loss_cutoff=0.01,  # yielded good results

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=True, # TODO: 
    prune_weights=True,
    pruning_target=50,
    pruning_levels=20,
    reinit=True
)
