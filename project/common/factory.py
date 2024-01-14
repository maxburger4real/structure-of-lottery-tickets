from common.config import Config
from common import datasets
from common import models

def make_dataloaders(config: Config):
    
    x_train, y_train, x_test, y_test = datasets.make_dataset(
        name=config.dataset,
        n_samples=config.n_samples,
        noise=config.noise,
        seed=config.data_seed,
        factor=config.factor,
        scaler=config.scaler,
    )

    batch_size = config.batch_size if config.batch_size is not None else config.n_samples
    train_dataloader, test_dataloader = datasets.make_dataloaders(x_train, y_train, x_test, y_test, batch_size)

    return train_dataloader, test_dataloader

def make_model(config: Config):

    shape = config.model_shape
    seed = config.model_seed
    activation = models.__activations_map[config.activation]

    match config.model_class:
        case models.SingleTaskMultiClassMLP.__name__:
            model = models.SingleTaskMultiClassMLP(shape=shape, activation=activation, seed=seed)
        case models.MultiTaskBinaryMLP.__name__:
            model = models.MultiTaskBinaryMLP(shape=shape, activation=activation, seed=seed)
        case models.MLP.__name__:
            raise ValueError('You shouldnt use MLP, it doesnt have a loss defined.')
        case _:
            raise ValueError('Model Unkown')

    # because enums are parsed to strings in config, parse back and convert to enum
    model.init(
        weight_init_func=models.Init[config.init_strategy_weights],
        bias_init_func=models.Init[config.init_strategy_biases]
    )
    model = model.to(config.device)
    return model