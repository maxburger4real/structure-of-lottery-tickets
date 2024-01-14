'''Runner for wandb experiments.'''
import wandb

from common.training_pipelines import vanilla, imp, bimt
from common.pruning import update_pruning_config
from common import factory
from common.constants import *

from settings import wandb_kwargs

def run(runconfig, sweepconfig, **kwargs):
    if sweepconfig is not None:
        sweep_id = wandb.sweep(
            sweep=sweepconfig, 
            entity=wandb_kwargs['entity'], 
            project=wandb_kwargs['project']
        )
        wandb.agent(
            sweep_id, 
            function=lambda : __start_run(runconfig, kwargs.get('mode', None)), 
            entity=wandb_kwargs['entity'], 
            project=wandb_kwargs['project'], 
            count=kwargs.get('count', None)
        )

    else:
        __start_run(runconfig, kwargs.get('mode', None))

def __start_run(config, mode=None):
    """Start a wandb experiment from a config."""

    with wandb.init(config=config, mode=mode, **wandb_kwargs) as run:

        # get changes made by sweep into config
        config.__dict__.update(**wandb.config)
        
        # optional config updates needed for model extension
        config = update_pruning_config(config) if Pipeline[config.pipeline] == Pipeline.imp else config

        # make model, loss, optim and dataloaders
        model = factory.make_model(config)
        train_loader, test_loader = factory.make_dataloaders(config)

        # save the config and add some wandb info to connect wandb with local files
        config.run_id = run.id
        config.run_name = run.name
        config.wandb_url = run.url
        config.local_dir_name = run.id

        # push the updated config to wandb.
        wandb.config.update(config.__dict__, allow_val_change=True)

        # run the pipeline defined in the config
        match Pipeline[config.pipeline]:
            case Pipeline.vanilla:
                return vanilla.run(model, train_loader, test_loader, config)

            case Pipeline.imp:
                return imp.run(model, train_loader, test_loader, config)
        
            case Pipeline.bimt:
                return bimt.run(model, train_loader, test_loader, config)
