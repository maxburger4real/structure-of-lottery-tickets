import wandb
from training_pipelines import pipeline_selector
from common.datasets.independence import build_loaders
from common.tracking import Config, PROJECT, save_hparams, get_model_path
from common.training import build_optimizer, build_loss
from common.architectures import build_model_from_config
from configs.runs import (
    _00_baseline,
    _01_first_good,
    _02_vanilla_mlp_from_bimt
)

def run_experiment(config):
    """Run a wandb experiment from a config."""

    with wandb.init(project=PROJECT, config=config) as run:

        # make model, loss, optim and dataloaders
        model = build_model_from_config(wandb.config)
        loss_fn = build_loss(wandb.config)
        optim = build_optimizer(model, wandb.config)
        train_loader, test_loader = build_loaders(wandb.config.batch_size)

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
            optim=optim,
            loss_fn=loss_fn,
            config=config_dict,
        )

def main(config=None):

    # SELECT THE CONFIG YOU WANT FOR THE RUN HERE
    if config is None:
        config = _02_vanilla_mlp_from_bimt.config

    run_experiment(config)

if __name__ == "__main__":
    main()