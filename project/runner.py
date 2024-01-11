'''Runner for wandb experiments.'''
import wandb

from common.config import Config
from common.models import build_model_from_config
from common.training_pipelines import vanilla, imp, bimt
from common.training import build_loss_from_config
from common.persistance import save_hparams
from common.pruning import update_pruning_config
from common.datasets import build_dataloaders_from_config
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

        # optional config updates needed for model extension
        if Pipeline[config.pipeline] == Pipeline.imp: 
            update_pruning_config(wandb.config)

        # make model, loss, optim and dataloaders
        model = build_model_from_config(wandb.config)
        loss_fn = build_loss_from_config(wandb.config)
        train_loader, test_loader = build_dataloaders_from_config(wandb.config)

        # save the config and add some wandb info to connect wandb with local files
        config_dict = Config(**wandb.config)
        config_dict.run_id = run.id
        config_dict.run_name = run.name
        config_dict.wandb_url = run.url
        config_dict.local_dir_name = run.id
        save_hparams(config_dict)

        # run the pipeline defined in the config
        match Pipeline[config.pipeline]:
            case Pipeline.vanilla:
                return vanilla.run(model, train_loader, test_loader, loss_fn, config_dict)

            case Pipeline.imp:
                return imp.run(model, train_loader, test_loader, loss_fn, config_dict)
        
            case Pipeline.bimt:
                return bimt.run(model, train_loader, test_loader, loss_fn, config_dict)
