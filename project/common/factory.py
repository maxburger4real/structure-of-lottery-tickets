from common.config import Config
from common import datasets

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
