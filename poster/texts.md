# dead parameters
are edges (weights) or nodes (biases) in the DAG where there is no path from any input feature to any output feature, that contains the edge.There are two types of dead parameters:
inactive parameters
No path to the output. They do not influence the output
zombie parameters
have no path to the input, but a nonzero bias produces activations

# independent subnetworks
are subnetworks within a larger neural network, where there is no connection from one subnetwork to the other. They are discovered by transforming the network into a directed acyclig graph (DAG) and searching for disjoint graphs. Pruned and dead parameters are ignored. 

# lottery tickets
refers to a small, sparse subnetwork of a larger neural network at initialization that, when trained in isolation, achieves performance comparable to the full network. Iterative magnitude pruning with parameter resetting, among other algorithms, can uncover lottery tickets.

# iterative magnitude pruning 
is an algorithm than consists of training and pruning a neural network iteratively, until the final desired sparsity is reached. 