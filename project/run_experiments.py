import wandb
import run_experiment
from common.tracking import PROJECT

# select the run_config to use
from configs.runs._05_imp_bimt_inspired import run_config

# select the sweep_config to use
from configs.sweeps._01_compare_with_multiple_seeds import sweep_config

project = PROJECT
entity = None  # not needed
count = None   # number of runs

def main():
    # initialize the sweep
    sweep_id = wandb.sweep(sweep_config, entity, project)

    # start execution of the sweeps
    function = lambda : run_experiment.main(run_config)
    wandb.agent(sweep_id, function, entity, project, count)

if __name__ == "__main__":
    main()