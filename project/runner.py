
import wandb
from training_pipelines import imp
from common.datasets.independence import build_loaders
from common.tracking import Config, PROJECT, save_hparams
from common.training import build_optimizer, build_model

from run_configs.baseline import config

def run_experiment(config):

    with wandb.init(project=PROJECT, name=config.experiment, config=config):
        # create the model, optimizer and dataloaders
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
    run_experiment(config)

if __name__ == "__main__":
    main()