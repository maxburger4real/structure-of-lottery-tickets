"""Simply run the configuration with multiple seeds to see stability."""

seeds = [
    0,
    1,
    2,
    3,
    #4,5,6,7,
]

sweep_config = {
    'method': 'grid', 
    'name': f'Testing with multiple seeds : {seeds}',
    'parameters':{"model_seed":  {"values": seeds}}
    }