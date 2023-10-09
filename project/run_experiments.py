import wandb
import run_experiment
from configs.runs import (
    _00_baseline,
    _01_first_good,
)
from configs.sweeps import (
    _00_basic_hparam_search,
    _01_compare_with_multiple_seeds
)

# SELECT THE CONFIG YOU WANT FOR THE SWEEP HERE
sweep_config = _01_compare_with_multiple_seeds
make_run_config = _01_first_good.make_config

def main():

    # initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config.config, 
        project=sweep_config.config['project']
    )

    # start execution of the sweeps
    function = run_experiment.run_with_config(make_run_config)
    wandb.agent(
        sweep_id, 
        function, 
        #count=50
    )

if __name__ == "__main__":
    main()