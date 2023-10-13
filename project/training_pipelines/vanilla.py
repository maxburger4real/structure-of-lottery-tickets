import wandb
from common.tracking import Config, get_model_path, save_model
from common.training import evaluate, train_and_evaluate

VAL_LOSS = 'val_loss'
TRAIN_LOSS = 'train_loss'

def run(model, train_loader, test_loader, optim, loss_fn, config: Config):
    device = config.device

    # log initial performance
    eval_loss_init = evaluate(model, test_loader, loss_fn, device).mean().item()
    wandb.log({VAL_LOSS : eval_loss_init})

    # train and evaluate
    train_losses, eval_losses = train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, device, epochs=config.training_epochs)

    # log 
    for t, v in zip(train_losses, eval_losses):
        wandb.log({VAL_LOSS : v, TRAIN_LOSS: t})
    
    # store model
    if config.persist: 
        model_path = get_model_path(config)
        save_model(model, iteration=config.training_epochs, base=model_path)

    return model