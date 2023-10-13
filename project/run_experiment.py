import wandb
from training_pipelines import pipeline_selector
from common.datasets.independence import build_loaders
from common.tracking import Config, PROJECT, save_hparams
from common.training import build_optimizer, build_loss
from common.architectures import build_model_from_config
from configs.runs import (
    _00_baseline,
    _01_first_good,
    _02_vanilla_mlp_from_bimt
)

def run_experiment(build_config_func):

    config = build_config_func()
    with wandb.init(project=PROJECT, config=config):

        # make model, loss, optim and dataloaders
        config = wandb.config
        model = build_model_from_config(config)
        loss_fn = build_loss(config)
        optim = build_optimizer(model, config)
        train_loader, test_loader = build_loaders(config.batch_size)

        # must convert back for serialization
        save_hparams(Config(**config))

        # run the pipeline defined in the config
        pipeline_selector.run(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optim=optim,
            loss_fn=loss_fn,
            config=config,
        )

def main():
    # SELECT THE CONFIG YOU WANT FOR THE RUN HERE
    make_config = _02_vanilla_mlp_from_bimt.make_config
    run_experiment(make_config)

def run_with_config(build_config_func):
    """Used to run with config from run_experiments."""
    return lambda : run_experiment(build_config_func)

if __name__ == "__main__":
    main()