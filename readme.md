# Winning The Lottery Twice
This Repository contains my Masters thesis, all the code and the thesis itself.

## Abstract:
The Lottery Ticket Hypothesis has sparked a novel field of research that focuses on finding well-trainable sparse neural networks.
**Iterative Magnitude Pruning** is a central method in this field.
It successfully uncovers sparse subnetworks that achieve performance comparable to the original dense network when trained in isolation. 
These networks are called **Winning Tickets**.
Why and how iterative magnitude pruning works and what characteristics of the winning tickets make them successful remain elusive.
To learn more about winning tickets, their network structure is studied in this thesis.
Since they are sparse subnetworks, their structure may contain valuable insights into their functionality.
Training data with a known structure is used to examine whether winning tickets trained on that data resemble its structure in any way.
The experiments in this thesis are conducted on datasets that contain two independent tasks - a simple toy dataset and a combination of the MNIST and Fashion-MNIST datasets.
With both datasets, the resulting winning tickets resemble the structure of the datasets, namely the independence.
The winning tickets contain separate, independent subnetworks where each subnetwork solves one independent task.

