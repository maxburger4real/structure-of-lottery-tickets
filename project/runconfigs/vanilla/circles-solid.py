'''
Checkout the report.
https://wandb.ai/mxmn/thesis/reports/Vanilla-MoonsCircles-2-50-1---Vmlldzo2MjUxNTA3
'''

from common.models import MLP
from common.config import Config
from common.constants import *

run_config = Config(
    pipeline=VANILLA,
    loss_fn=BCE,
    dataset=Datasets.CIRCLES.name,
    n_samples=100,
    noise=0.1,
        
    model_class = MLP.__name__,
    activation=RELU,
    # 10/10
    model_shape=[2, 50, 1],

    # 10/10
    # model_shape=[2, 13, 13, 1],

    init_strategy_weights = InitializationStrategy.KAIMING_NORMAL,
    init_strategy_biases = InitializationStrategy.ZERO,
    
    # training
    lr=0.001,
    optimizer=ADAM,
    epochs=10000,

    # seeds
    model_seed=0,
    persist=False,
)
