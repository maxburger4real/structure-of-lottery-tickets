"""Simply run the configuration with multiple seeds to see stability."""

seeds = list(range(10, 15))
seeds = list(range(10))
seeds = list(range(100, 110))
seeds = [42, 17, 444]

sweep_config = {
    "method": "grid",
    "name": f"Testing with multiple seeds : {seeds}",
    "parameters": {"model_seed": {"values": seeds}},
}
