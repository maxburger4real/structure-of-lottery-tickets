import wandb
from common.tracking import Config, save_model, logdict
from common.training import build_early_stopper, build_optimizer, evaluate, update

VAL_LOSS = 'val_loss'
TRAIN_LOSS = 'train_loss'

def run(model, train_loader, test_loader, loss_fn, config: Config):

    optim = build_optimizer(model, config)
    stop = build_early_stopper(config)

    # log initial performance
    loss_init = evaluate(model, test_loader, loss_fn, config.device)
    wandb.log(logdict(loss_init, VAL_LOSS))

    # train and evaluate
    for _ in range(0, config.training_epochs):
        
        # update
        loss_train = update(model, train_loader, optim, loss_fn, config.device, config.l1_lambda).mean()
        wandb.log(logdict(loss_train, TRAIN_LOSS), commit=False)

        # evaluate
        loss_eval = evaluate(model, test_loader, loss_fn, config.device)
        wandb.log(logdict(loss_eval, VAL_LOSS))

        if stop(loss_eval.mean().item()): break

    # store model
    if config.persist: save_model(model, config, config.training_epochs)

    return model