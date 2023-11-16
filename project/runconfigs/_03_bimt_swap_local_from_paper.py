from common.architectures import BioMLP
from common.datasets.independence import INPUT_DIM, OUTPUT_DIM, DATASET_NAME
from common.config import Config
from common.constants import *

run_config = Config(
    description="""
    Train a simple BioLinear MLP (from BIMT) for some epochs.
    Locality and swapping is activated for alleged 'best' results from BIMT.
    """,
    pipeline=BIMT,
    activation=SILU,
    loss_fn=MSE,
    dataset=DATASET_NAME,
    
    model_shape=[INPUT_DIM, 20, 20, OUTPUT_DIM],
    model_class = BioMLP.__name__,

    # training
    lr=0.002,
    optimizer=ADAMW,
    training_epochs=20000,

    # seeds
    model_seed=0,
    data_seed=0,
    persist=True,

    l1_lambda=0.001,
    bimt_swap=200,
    )
