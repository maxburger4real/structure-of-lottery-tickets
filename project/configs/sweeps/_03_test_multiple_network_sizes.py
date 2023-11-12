"""Simply run the configuration with multiple seeds to see stability."""

m = 2
sweep_config = {
    'method': 'grid', 
    'name': 'different network sizes',
    'parameters':{
        "model_shape" : {
            "values":[
                # [m*2, 80, 80, m],
                # [m*2, 100, 100, m],
                # [m*2, 120, 120, m],
                # [m*2, 140, 140, m],
                # [m*2, 160, 160, m],
                # [m*2, 180, 180, m],
                # [m*2, 200, 200, m],
                # [m*2, 250, 250, m],
                # [m*2, 300, 300, m],
                [m*2, 400, 400, m],
                [m*2, 500, 500, m],
                [m*2, 600, 600, m],
                [m*2, 700, 700, m],
                [m*2, 800, 800, m],
                [m*2, 900, 900, m],
                [m*2, 1000, 1000, m],
                [m*2, 1500, 1500, m],
                [m*2, 2000, 2000, m],
                [m*2, 3000, 3000, m],
            ]
        },
        "model_seed":  {
            "values": [77,88,99]
        },
        }
    }