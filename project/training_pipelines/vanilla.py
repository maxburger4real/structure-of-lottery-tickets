import wandb
from common.tracking import Config, save_model
from common.training import build_optimizer, evaluate, train_and_evaluate

VAL_LOSS = 'val_loss'
TRAIN_LOSS = 'train_loss'

def run(model, train_loader, test_loader, loss_fn, config: Config):

    optim = build_optimizer(model, config)

    # log initial performance
    eval_loss_init = evaluate(model, test_loader, loss_fn, config.device).mean().item()
    wandb.log({VAL_LOSS : eval_loss_init})

    # train and evaluate
    train_losses, eval_losses = train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, config.device, config.training_epochs)

    # log 
    for t, v in zip(train_losses, eval_losses):
        wandb.log({VAL_LOSS : v, TRAIN_LOSS: t})
    
    # store model
    if config.persist: save_model(model, config, config.training_epochs)

    return model