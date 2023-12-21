from enum import Enum

SEED = 64

STATE_DICT = "model_state_dict"
HPARAMS_FILE = 'hparams.json'

# wandb
PROJECT='concat_moons' #'init-thesis'
ENTITY='mxmn'
MODE='online'
# MODE='disabled'


InitializationStrategy = Enum('InitializationStrategy', ['ZERO', 'NORMAL', 'DEFAULT', 'KAIMING_NORMAL'])
Datasets = Enum('Datasets', ['MOONS', 'CIRCLES', 'MOONS_AND_CIRCLES'])
TaskCoverage = Enum('TaskCoverage', ['COMPLETE', 'PARTIAL', 'ABSENT'])

# Datasets
OLD_MOONS = 'old moons'  # they still worked
MOONS = 'classic two moons dataset form sklearn'
CIRCLES = 'classic circles dataset from sklearn'
MOONS_AND_CIRCLES = 'moons and circles concatenated classification'
MULTI_MOONS = 'moons'
MULTI_CIRCLES = 'circles'
SYMBOLIC_INDEPENDENCE_REGRESSION = 'symbolic independece regression'


# Optimizers
ADAM = 'Adam'
SGD = 'sgd'
ADAMW = 'AdamW'

# Activations
RELU = 'relu'
SILU = 'silu'

# Loss Functions
MSE = 'mse'
CCE = 'cce'
BCE = 'bce'

# Pruning Methods
MAGNITUDE = 'magnitude'
RANDOM = 'random'

# Training Pipelines
VANILLA = 'vanilla'
IMP = 'imp'
BIMT = 'bimt'

# Metrics to Track
VAL, TRAIN = 'val', 'train'
ACC, LOSS = 'acc', 'loss'
VAL_LOSS = VAL + "-" + LOSS
TRAIN_LOSS = TRAIN + "-" + LOSS
ACCURACY = 'accuracy'

# 
PRUNABLE = 'prunable_params'
SPARSITY = 'sparsity'
STOP = 'stop'

# PLOTTING
POSITIVE_COLOR = "blue"
NEGATIVE_COLOR = "red"

# INDEPENDENT SUBNETWORKS N THE NETWORK
SUBNETWORK = "group"
GROUP_COLORS = [
    'red', 'green', 'blue', 'purple', 'orange',
    'pink', 'brown', 'gray', 'olive', 'cyan'
]

# EDGES
LINE_COLOR = 'line_color'
LINE_WIDTH = 'line_width'
LINE_ALPHA = 'line_alpha'

# NODES
WEIGHT='weight'
BIAS='bias'
LAYER='rank'
NODE_ALPHA = 'node_alpha'
NODE_COLOR = 'node_color'
NODE_LINE_COLOR = 'node_line_color'
NODE_LINE_ALPHA = 'node_line_alpha'
IS_ZOMBIE = 'zombie'
IS_HEADLESS = 'headless'