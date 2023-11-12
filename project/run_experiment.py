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
from configs.config_importer import import_config
from training_pipelines import pipeline_selector
from common.architectures import build_model_from_config
from common.datasets.dataset_selector import build_loaders
from common.persistance import save_hparams
from common.config import Config
from common.pruning_trajectory import update_pruning_config
from common.training import build_loss
from common.constants import *

project = PROJECT
entity = ENTITY
count = None # 30   # number of runs

def run_experiment(config, mode=None):
    """Run a wandb experiment from a config."""

    with wandb.init(project=PROJECT, config=config, mode=mode) as run:

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_config', help='run config file', required=True)
    parser.add_argument('-s', '--sweep_config', help='sweep config file')
    parser.add_argument('-d', '--disable', action='store_true', help="Disable WANDB mode.")
    argcomplete.autocomplete(parser)  # Enable autocompletion with argcomplete
    args = parser.parse_args()  # Parse the arguments

    run_config = import_config(args.run_config)
    sweep_config = import_config(args.sweep_config)
    mode = 'disabled' if args.disable else None

    if run_config is None:
        raise ValueError

    if sweep_config is not None:
        # initialize the sweep
        sweep_id = wandb.sweep(sweep_config, entity, project)

        # start execution of the sweeps
        function = lambda : run_experiment(run_config, mode)
        wandb.agent(sweep_id, function, entity, project, count)

    else:
        run_experiment(run_config, mode)



if __name__ == "__main__":
    main()