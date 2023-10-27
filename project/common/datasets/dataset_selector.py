import common.datasets.independence as indep
import common.datasets.concatenated_moons as moons
from common.tracking import Config
from common.constants import *

def build_loaders(config: Config):

    if config.dataset == SYMBOLIC_INDEPENDENCE_REGRESSION:
        return indep.build_loaders(config.batch_size)

    if config.dataset == CONCAT_MOONS:
        m = config.num_concat_datasets
        return moons.build_loaders(m, config.batch_size)
