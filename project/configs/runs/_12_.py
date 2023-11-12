from common.architectures import SimpleMLP
from common.config import Config
from common.constants import *
from common.pruning_trajectory import calc_params

m = 2
in_dim = m*2
out_dim = m
hidden_dim = 40
num_hidden = 2

P0 = calc_params(in_dim, out_dim, hidden_dim)

for i in range(5):
    shape = [in_dim] + [hidden_dim]*num_hidden + [out_dim]

run_config = Config(
    description='''This is the winning formula. The netowrk splits before the performance degrades.''',
    pipeline=IMP,
    activation=RELU,
    loss_fn= BCE,
    dataset=CONCAT_MOONS,
    num_concat_datasets=m,
    
    model_shape=[m*2, 40, 40, m],
    model_class = SimpleMLP.__name__,

    # training
    lr=0.001,
    optimizer=ADAM,
    training_epochs= 3000,

    # seeds
    model_seed=5, # good seeds : 2
    data_seed=0,

    persist=False,

    # early stop
    early_stopping=True,
    early_stop_patience=10,
    early_stop_delta=0.0,
    loss_cutoff=0.01,  # yielded good results

    # pruning
    pruning_method=MAGNITUDE,
    prune_biases=True,
    prune_weights=True,
    pruning_target=50,
    pruning_levels=20,
    reinit=True
)
