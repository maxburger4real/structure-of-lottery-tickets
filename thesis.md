Idea

Goal

Why is it interesting ? 

Experiments inspired by BIMT

Does IMP find independent tasks? 
If the lottery ticket is good, it should not cointain unnecessary components. Therefore shouldnt have connections to random inputs

Idea :
Concatenate the Dataset together
2 Input 2 output.
Then experiment with IMP and see if the Model ends up as 2 sperate models
So far, this happens quite consistently. It is really expensive to measure how close we are to 2 disconnected models, so either it is disconnected or not.

This can be done with arbitrary many tasks.
Only classification though

Experiments on Regresssion with iterative Magnitude Pruning were not successful. Probably due to the inherent numerical Delicacy with Regression. Regression is Analog, Classification is digital.

So: Concatenaded Moons or Two * N Moons Dataset.
Training Lottery ticket works.
\__/ This is what the val_los graph should look like. Used early stopping to save time.

So far it doesnt seem to work
By the time the Network splits, it does not seem to be a lottery ticket anymore. It merely is a weak classifier. Lets try without noise? 


TODO: Try without noise. Find way to say (I want this model down to 40 params.) and the pruning rate is implicitly calculated, depending on the number of pruning iterations (which are the metric of money).

Fit an exponential curve to 2 points : number of prunable params before pruning and the desired number after pruning. This fitted exponential is discretized with n steps, where n is the number of pruning levels that are selected. 
This effectively creates a setup where it is easily possible to tune the hyperparameters. 
Network size and pruning iterations are the parameters that change the training time significantly.


TODO:
There are interesting Metrics I would like to track, ideally during training of a Soon-to-be-Lottery ticket.
* number of prunable / non-prunable parameters remaining and number of weights and number of biases.
* number of zombie neurons (Life)



New experimental results 'hezzefi6' show that the zombies survive longer when they have a large positive bias.

TODO: 
track the weights over time, where they end up. Delete weights that are 0. dont log them.


YAY! Milestone
I found a setup, where the NN splits before the network quality degrades substantially.
This is cool
The question also is
How likely is it, that the network splits exactly how I want it to be?
And, at what point should/could the network split first, and when will it split for sure.

There seems to be a relationship between not splitting and bad performance.


### params

multiplying the number of neurons in the hidden layer by n, increases the number of weights by a factor of n**2
The largest possible split subnetworks. lets say neural network has the number of parameters $P$ that covers $k$ equal task.
Then, the minimum number of weights to remove, such that the largest possible complete subnetworks arise is 
$$\frac{P(k-1)}{k}$$
and the minimum number of parameters where this split occurs is obviously 
$$\frac{P}{k}$$


## Observations
* Zomie Neurons appear more often at higher sparsities
* Zombie Neurons can appear to survive longer if their associated bias is larger. Zombies with negative, 
* Comatose Neurons appear to only survive for few iterations
* Comatose Neurons behave very similarly across different seed-runs  
    * why? As soon as they become comatose, they do not receive any updates. When the average parameter rises, they lower relatively to the others.
* Negative Biases disappear (pruned, turn positive) --> RELU -> negative bias increases likelihood of 0-Activation and therefore no update which increases likelihood of pruning.

the pruning horizon. It increases steadily. 


## Experiments to run

What is actually interesting to test? What are the Hypothesis?

Hypothesis :


New experiment, broader hidden size.
What do I expect ? 
