import wandb
from common import pruning
from common.torch_utils import measure_global_sparsity
from common.tracking import Config, get_model_path, save_model
from common.training import evaluate, update, train_and_evaluate, EVAL_LOSS, TRAIN_LOSS


def run(model, train_loader, test_loader, optim, loss_fn, config: Config):

    model_path = get_model_path(config)

    wandb.define_metric(EVAL_LOSS, summary="min")
    wandb.define_metric(TRAIN_LOSS, summary="min")
    wandb.define_metric('sparsity', summary="min")

    wandb.define_metric("p-lvl")
    wandb.define_metric("nonzero", step_metric="p-lvl")
    wandb.define_metric("sparsity", step_metric="p-lvl")
    wandb.define_metric("sparsity", step_metric="p-lvl")


    # preparing for pruning and [OPTIONALLY] save model state
    params_to_prune = pruning.convert_to_pruning_model(model.modules, prune_weights=True, prune_biases=True)
    if config.reinit: reinit_model_state_dict = pruning.get_model_state_dict(model, drop_masks=True)
    if config.persist: save_model(model, iteration=0, base=model_path)

    # log initial performance
    eval_loss_init = evaluate(model, test_loader, loss_fn).mean().item()
    wandb.log({
        EVAL_LOSS + 'init' : eval_loss_init,
        'epoch' : 0
    })
    
    # loop over pruning levels
    for lvl in range(1, config.pruning_levels+1):

        wandb.log({'p-lvl' : lvl}, commit=False)
        train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, epochs=config.training_epochs)
        if config.persist: save_model(model, iteration=lvl, base=model_path)

        # prune by global magnitude
        pruning.global_magnitude_pruning(params_to_prune, config.pruning_rate)

        # [OPTIONAL] reinit
        if config.reinit: model.load_state_dict(reinit_model_state_dict, strict=False)

    wandb.log({'p-lvl' : lvl+1}, commit=False)
    
    # final finetuning
    train_and_evaluate(model, train_loader, test_loader, optim, loss_fn, epochs=config.training_epochs)
    if config.persist: save_model(model, iteration=lvl+1, base=model_path)

    return model