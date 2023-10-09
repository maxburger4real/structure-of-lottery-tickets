import wandb
from training_pipelines import imp
from common.datasets.independence import build_loaders
from common.tracking import Config, PROJECT, save_hparams
from common.training import build_optimizer, build_model
from configs.runs import (
    _00_baseline,
    _01_first_good,
)

def run_experiment(build_config_func):

    config = build_config_func()
    # import now creates the object and the timestamp
    with wandb.init(
        project=PROJECT, 
        config=config
    ):

        # create the model, optimizer and dataloaders
        config.model_seed = wandb.config.model_seed
        model, loss_fn = build_model(config)

        # the sweep will overwrite this
        config = wandb.config
        optim = build_optimizer(model, config)
        train_loader, test_loader = build_loaders(config.batch_size)

        # must convert back for serialization
        save_hparams(Config(**config))

        model = imp.run(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optim=optim,
            loss_fn=loss_fn,
            config=config,
        )

def main():
    # SELECT THE CONFIG YOU WANT FOR THE RUN HERE
    make_config = _01_first_good.make_config
    run_experiment(make_config)

def run_with_config(build_config_func):
    """Used to run with config from run_experiments."""
    return lambda : run_experiment(build_config_func)

if __name__ == "__main__":
    main()