"""Simply run the configuration with multiple seeds to see stability."""

m = 2
sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "model_shape" : {
            "values":[
                [m*2, 40, 40, m],
                [m*2, 50, 50, m],
                [m*2, 60, 60, m],
                [m*2, 70, 70, m],
            ]
        },
        "model_seed":  {
            "values": [77,88,99]
        },
        }
    }