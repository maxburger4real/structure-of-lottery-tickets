import wandb
from training_pipelines import pipeline_selector
from common.datasets.dataset_selector import build_loaders
from common.tracking import Config, save_hparams
from common.training import build_loss
from common.architectures import build_model_from_config
from common.constants import *

# select the run_config to use
from configs.runs._08_imp_moons import run_config

def run_experiment(config):
    """Run a wandb experiment from a config."""

    with wandb.init(project=PROJECT, config=config, mode=MODE) as run:

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

def main(config=None):

    # SELECT THE CONFIG YOU WANT FOR THE RUN HERE
    if config is None: config = run_config
    run_experiment(config)

if __name__ == "__main__":
    main()