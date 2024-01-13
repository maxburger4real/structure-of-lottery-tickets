## Introduction

The lottery ticket hypothesis showed that there are small subnetworks ( lottery tickets ) of a sufficiently overparameterized neural network, that can be trained to the same accuracy as the original network, often with faster convergence and better generalization.
Via Iterative Magnitude Pruning, these subnetworks can be uncovered. Yet it still remains elusive, why these subnetworks work or what makes them work.

This thesis aims to shed light on the structure of lottery tickets. 
The main idea of this thesis is to look at a problem with a known structure and examine the lottery ticket throught this lens.

A simple problem with a known structure would be a dataset that contains multiple independent tasks. Each tasks is accociated with inputs and outputs. Each input and output belongs to exactly one task.

### A concrete example:  
There is task A with inputs $x_1, x_2$ and output $y_1$.  
There is task B with inputs $x_3, x_4$  and output $y_2$.  
$x_1, x_2$ do not contain any information that is relevant for $y_2$.  
$x_3, x_4$ do not contain any information that is relevant for $y_1$.    
Let $\mathcal{F}$ be a fully connected network with inputs $x_1, x_2, x_3, x_4$ and outputs $y_1, y_2$.

the questions:
- will the resulting lottery tickets reveal this structure?
- will there be 2 independent networks?

## Experiment setup
For the experiment, two well known toy datasets are used and concatenated into a parallel tasks dataset.
The two moons dataset and the circles dataset. Both are 2 class classification problems with 2 inputs each.

The data is normalized and a neural network is trained with the datset.
Then, IMP is used to sparsify the network. In this experiments, global magnitude pruning is used, namely pruning $p$% of weights (not biases) with the lowest magnitude. 

The neural networks are fully connected, uses ReLU activation, the weights are initialized with Kaiming-Normal initialization and the biases with 0.

Experiments show that the network often splits while it still shows near perfect classification accuracy on each task.

### Considerations
As always with machine learning, there are very many hyperparameters or design decisions that can influence the structure of the lottery ticket.

The main parameters of concern for these experiments are the following:  

* $L$ - pruning levels : how many iterations if magnitude pruning are applied?
* $p$ - pruning rate: what percentage of weights are removed in a single pruning level:
* $T_L$ - pruning target : what is the final number of weights after finishing 
* $T_0$ - pruning parameters : number parameters considered for pruning. Mainly influenced by model architecture.

With these parameters, a pruning trajectory $T$ can be calculated. the trajectory starts with $T_0$, followed by $L$ values which are calculated by multiplying $T_0$ with $p^l$, where $l$ is the current level.

$$
T_l = T_0 * (1-p)^l
$$
$$
T_L = T_0(1-p)^{-l}
$$
$$
l = \frac{\log{T_L- \log{T_0}}}{\log{(1-p)}}
$$
$$
p = 1 - \sqrt[l]\frac{T_0}{T_L}
$$

To specify a pruning trajectory, 3 out of 4 are needed. Yet for practical reasons, each one of them is used in different contexts.