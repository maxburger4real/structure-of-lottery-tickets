"""Simply run the configuration with multiple seeds to see stability."""
sweep_config = {
    'method': 'grid', 
    'name': 'alternating seeds',
    'parameters':{
        "model_seed":  {
            #"values": [42, 21, 6, 91, 2, 4, 9, 11, 400, 1, 0, 17]
            "values": [0, 42, 21, 6, 91]
        },
        }
    }