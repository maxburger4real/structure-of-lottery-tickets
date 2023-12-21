from common.models import SimpleMLP
from common.datasets.independence import INPUT_DIM, OUTPUT_DIM, DATASET_NAME
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=VANILLA,
    activation=SILU,
    loss_fn=MSE,
    dataset=DATASET_NAME,
    
    model_shape=[INPUT_DIM, 20, 20, OUTPUT_DIM],
    model_class = SimpleMLP.__name__,


    # training
    lr=0.002,
    optimizer=ADAMW,
    epochs=20000,

    # seeds
    model_seed=0,
    data_seed=0,
    persist=False,
)
