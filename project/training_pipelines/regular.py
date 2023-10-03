import wandb
from common import pruning

from common.tracking import Config
from common.training import evaluate, train_and_evaluate

def run(model, train_loader, test_loader, optim, loss_fn, config: Config):

    wandb.log({'loss/eval' : evaluate(model, test_loader, loss_fn).mean().item()})
    
    train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, epochs=config.training_epochs, logger=wandb.log)

    return model
