'''
Runner for wandb experiments.
1. Retrieves the config files
2. creates a wandb run or sweep with the selected training routine
'''
import wandb

from training.pruning import update_pruning_config
from training.routines import start_routine
from training.config import import_config, Config

from settings import wandb_kwargs

def run(experiment_file, run_file, sweep_file, **kwargs):

    runconfig, sweepconfig = __import_configs(experiment_file, run_file, sweep_file)

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

def __import_configs(experiment_file, run_file, sweep_file):
    '''
    Import the configurations from the provided file. 
    returns at least a runconfig, optionally a sweepconfig
    '''
    if experiment_file is not None:
        runconfig, sweepconfig = import_config(experiment_file)
    elif run_file is not None:
        runconfig, _ = import_config(run_file)
        _ ,sweepconfig = import_config(sweep_file)
    else:
        raise ValueError('Must provide either experiment or runconfig.')
        
    if runconfig is None: raise ValueError('No valid runconfig found.')

    return runconfig, sweepconfig
    
def __start_run(config: Config, mode=None):
    """Start a wandb experiment from a config."""

    with wandb.init(config=config, mode=mode, **wandb_kwargs) as run:

        # get changes made by sweep into config
        config.__dict__.update(**wandb.config)
        
        # optional config updates needed for model extension
        config = update_pruning_config(config)

        # save the config and add some wandb info to connect wandb with local files
        config.run_id = run.id
        config.run_name = run.name
        config.wandb_url = run.url
        config.local_dir_name = run.id

        # push the updated config to wandb.
        wandb.config.update(config.__dict__, allow_val_change=True)

        start_routine(config)
