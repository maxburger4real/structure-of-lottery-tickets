from enum import Enum


SEEDS_123 = [1,2,3]
SEEDS_1 = [1]

bias = 'bias'
weight='weight'
layer='layer'
state='state'


ParamState = Enum(
    'ParamState', 
    ['active', 'inactive', 'zombious', 'zombie_downstream', 'pruned']
)

class Pipeline(Enum):
    vanilla = 'vanilla'
    imp = 'imp'
    bimt = 'bimt'


STATE_DICT = "model_state_dict"

TaskCoverage = Enum('TaskCoverage', ['COMPLETE', 'PARTIAL', 'ABSENT'])

# Optimizers
ADAM = 'adam'
SGD = 'sgd'
ADAMW = 'adamw'

# Scalers
MinMaxZeroMean = 'min-max-zero-mean'
MinMaxZeroOne = 'min-max-zero-one'
StandardUnitVariance = 'std-unit-variance'

# Activations
RELU = 'relu'
SILU = 'silu'
SIGM = 'sigmoid'

# Pruning Methods
MAGNITUDE = 'magnitude'
RANDOM = 'random'

# Pruning scopes
GLOBAL = 'global'
LAYERWISE = 'layerwise'



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