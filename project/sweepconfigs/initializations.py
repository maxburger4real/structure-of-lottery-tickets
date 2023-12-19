"""Simply run the configuration with multiple seeds to see stability."""

from common.constants import InitializationStrategy
seeds = list(range(10,15))
seeds = list(range(10))
seeds = list(range(100,110))
seeds = [42,17,444]

w = (
        #InitializationStrategy.KAIMING_NORMAL.name,
        InitializationStrategy.XAVIER_NORMAL.name,
        #InitializationStrategy.XAVIER_UNIFORM.name,
    )

shapes = (
    (2,25,1),
    (2,40,1),
    (2,60,1),
    (2,10,10,1),
    (2,15,15,1),
    (2,20,20,1),
)

sweep_config = {
    'method': 'grid', 
    'name': f'Testing with multiple seeds : {seeds}',
    'parameters':{
        "init_strategy_weights":  {"values": w},
        "model_seed":  { "values": seeds },
        "activation": { "values" : ['sigmoid']},
        "model_shape": { "values" : shapes}
        }
    }