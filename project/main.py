"""
USAGE
-d wandb disable
python project/run_experiment.py -d -r project/configs/runs/_000_debug.py
python project/run_experiment.py -d -r project/configs/runs/_000_debug.py
python project/run_experiment.py -d -r project/configs/runs/_000_debug.py -s project/configs/sweeps/_01_compare_with_multiple_seeds
"""

import wandb
import argparse
import argcomplete

from settings import WANDB_DIR, RUNS_DATA_DIR

from common.training_pipelines import pipeline_selector
from common.models import build_model_from_config
from common.datasets.dataset_selector import build_loaders
from common.persistance import save_hparams
from common.config import Config, import_config
from common.pruning_trajectory import update_pruning_config
from common.training import build_loss
from common.constants import *
from common.plotting import plot_checkpoints


def run_experiment(config, mode=None):
    """Run a wandb experiment from a config."""

    with wandb.init(project=PROJECT, config=config, mode=mode, dir=WANDB_DIR) as run:

        # optional config updates needed for model extension
        update_pruning_config(wandb.config)

        # make model, loss, optim and dataloaders
        model = build_model_from_config(wandb.config)
        loss_fn = build_loss(wandb.config)
        train_loader, test_loader = build_loaders(wandb.config)

        # save the config and add some wandb info to connect wandb with local files
        config_dict = Config(**wandb.config)
        config_dict.run_id = run.id
        config_dict.run_name = run.name
        config_dict.wandb_url = run.url
        config_dict.local_dir_name = run.id
        save_hparams(config_dict)

        # run the pipeline defined in the config
        pipeline_selector.run(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            config=config_dict,
        )

        return str(run.id)

def plot(run_ids=None, sweep_ids=None):
    """plot any number of runs or sweeps."""
    if run_ids is not None:
        for name in run_ids:
            plot_checkpoints(RUNS_DATA_DIR / name)

    if sweep_ids is not None:
        for name in sweep_ids:
            api = wandb.Api()
            sweep = api.sweep(f"{ENTITY}/{PROJECT}/{name}")
            run_ids = [run.id for run in sweep.runs]

            for name in run_ids:
                plot_checkpoints(RUNS_DATA_DIR / name)

def main():

    parser = argparse.ArgumentParser()
    argcomplete.autocomplete(parser)  # Enable autocompletion with argcomplete

    # runs
    parser.add_argument('-r', '--run_config', help='run config file', required=True)
    parser.add_argument('-p', '--plot', action='store_true', help="plot nn that are created. No sweeps yet.")

    # sweeps
    parser.add_argument('-s', '--sweep_config', help='sweep config file')
    parser.add_argument('--count', help="The maximum number of runs for this sweep")

    # for plotting seperatedly
    parser.add_argument('--run_ids', nargs='+', help="The run ids to plot. Any number ")
    parser.add_argument('--sweep_ids',  nargs='+',  help="The sweep ids to plot. Any number ")
    
    # debug - disbale wandb logging to speedup
    parser.add_argument('-d', '--disable', action='store_true', help="Disable WANDB mode.")
    
    # THE ARGHSSSSSS
    args = parser.parse_args()

    # read count if it exists or ignore
    count = None
    if args.count is not None:
        if args.count.isdigit():
            count = int(args.count)
        else: 
            print(f'Ignoring args.count {args.count}')

    run_config = import_config(args.run_config)
    sweep_config = import_config(args.sweep_config)
    mode = 'disabled' if args.disable else None
    
    # skip
    if run_config is None:
        print('No run config provided')
    
    # single run
    elif sweep_config is None:
        id = run_experiment(run_config, mode)
        if args.plot and run_config.persist: 
            plot(run_ids=[id])

    # sweep
    else: 
       
        sweep_id = wandb.sweep(sweep_config, ENTITY, PROJECT)  # initialize the sweep
        function = lambda : run_experiment(run_config, mode)
        wandb.agent(sweep_id, function, ENTITY, PROJECT, count) # start execution of the sweeps

    plot(run_ids=args.run_ids, sweep_ids=args.sweep_ids )


if __name__ == "__main__":
    main()
