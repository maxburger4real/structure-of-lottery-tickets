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





# V2

Winning the Lottery Twice
Author: Maximilian Burger
Supervisor: Pieter-Jan Hoedt
Intro
The Lottery Ticket Hypothesis showed, that highly sparse networks can be trained from initialization. They are called ‘lottery tickets’. What makes them work remains unknown. The structure of lottery tickets might contain valuable insights.
Idea
Take a problem with a known structure.  Do lottery tickets trained in this problem, reveal it’s structure?
Experiments
The structure of the problem is independence. Concatenate two toy datasets into one. Train a fully connected network and use iterative magnitude pruning (IMP) with parameter resetting to find lottery tickets. Will the resulting lottery tickets contain two independent subnetworks?
Results
Iterative magnitude pruning indeed produces independent subnetworks.
The networks split rather consistently, when they contain between 75 and 125 weights, excluding dead parameters.
This remains the case, even when increasing the model size or changing the pruning rate.