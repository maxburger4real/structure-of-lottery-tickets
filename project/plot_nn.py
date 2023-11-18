import wandb
import argparse
from settings import RUNS_DATA_DIR
from common.plotting import plot_checkpoints
from common.constants import *
entity = ENTITY
project = PROJECT

def main(run_ids=None, sweep_ids=None):
    
    if run_ids is not None:
        for name in run_ids:
            plot_checkpoints(RUNS_DATA_DIR / name)

    if sweep_ids is not None:
        for name in sweep_ids:
            api = wandb.Api()
            sweep = api.sweep(f"{entity}/{project}/{name}")
            run_ids = [run.id for run in sweep.runs]

            for name in run_ids:
                plot_checkpoints(RUNS_DATA_DIR / name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_ids', nargs='+')
    parser.add_argument('-s', '--sweep_ids',  nargs='+')
    args = parser.parse_args()  # Parse the arguments

    main(args.run_ids, args.sweep_ids)
    