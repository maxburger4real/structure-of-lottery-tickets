import wandb
import run_experiment
from common.constants import *

# select the run_config to use
from configs.runs._06_imp_bimt_inspired_with_L1 import run_config

# select the sweep_config to use
from configs.sweeps._01_compare_with_multiple_seeds import sweep_config

project = PROJECT
entity = ENTITY
count = None # 30   # number of runs

def main():
    # initialize the sweep
    sweep_id = wandb.sweep(sweep_config, entity, project)

    # start execution of the sweeps
    function = lambda : run_experiment.main(run_config)
    wandb.agent(sweep_id, function, entity, project, count)

if __name__ == "__main__":
    main()