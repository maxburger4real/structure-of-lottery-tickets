import networkx as nx
import numpy as np
from itertools import count
from torch import load
from pathlib import Path

STATE_DICT = "model_state_dict"
POSITIVE_COLOR = "blue"
NEGATIVE_COLOR = "red"

def black_and_white(layers_of_weights):

    layers = []
    for weight_matrix in layers_of_weights:
        X = weight_matrix.numpy()
        mask = X > 0

        # Create X with the same shape as Y and fill it with 'no'
        Y = np.full(X.shape, fill_value='red', dtype=object)

        # Fill X with 'yes' where Y is True
        Y[mask] = 'blue'

        layers.append(Y)
    
    return layers

def alpha_value(weights):
    # TODO: remove complexity of this function style.
    # simply use a list of matrices.

    norm_weights = [ np.abs(w)/ np.abs(w).max() for w in weights]

    def func(G, edge, weight_pos, matrix_idx):
        W = norm_weights[matrix_idx]
        return W[weight_pos].item()

    return func


def load_state_dict_from_file(file: Path):
    return load(file)[STATE_DICT]

def get_shape_from_state_dict(state_dict):
    shapes = [v.shape for k,v in state_dict.items() if 'weight' in k]
    return shapes

def get_weights_and_biases_from_state_dict(state_dict):
    weights = [v for k,v in state_dict.items() if 'weight' in k]
    biases = [v for k,v in state_dict.items() if 'bias' in k]
    return weights, biases

def get_layers_of_nodes(G: nx.DiGraph):
    """
    Returns a dictionary of Attributes as keys, 
    with a list of nodes that have that attribute as the value
    """
    mydict = nx.get_node_attributes(G, 'layer')
    layers_of_nodes = []
    for key, value in mydict.items():
        if value == len(layers_of_nodes):
            layers_of_nodes.append([])

        layers_of_nodes[value].append(key)

    return layers_of_nodes



# Populate the Graph

def add_weight_edges_arrays(
    G: nx.DiGraph,
    layers_to_nodes, 
    attribute_set,  # list of functions that return attribute and value
):
    """Inplace add attributes to all edges, according to attribute functions."""
    
    attrs = {} 
    for name, layers_of_weights in attribute_set.items():
        for layer, timeful_weight_matrix in enumerate(layers_of_weights):
            for i, in_node in enumerate(layers_to_nodes[layer]):
                for o, out_node in enumerate(layers_to_nodes[layer+1]):
                    graph_pos = (in_node, out_node)
                    attrs[graph_pos] = {name : timeful_weight_matrix[:, o, i]}

    nx.set_edge_attributes(G, attrs)


def add_weight_edges(
    G: nx.DiGraph,
    layers_to_nodes, 
    iteration,
    attribute_functions,  # list of functions that return attribute and value
):
    """Inplace add attributes to all edges, according to attribute functions."""
    
    attrs = {}

    for layer in range(len(layers_to_nodes)-1):
        for i, in_node in enumerate(layers_to_nodes[layer]):
            for o, out_node in enumerate(layers_to_nodes[layer+1]):

                weight_pos = (o, i)
                edge = (in_node, out_node)

                attrs[edge] = {}

                for name, func in attribute_functions.items():
                    attrs[edge][f'{name}-{iteration}'] = func(G, edge, weight_pos, layer) * 0.1 * iteration

    nx.set_edge_attributes(G, attrs)

def add_neuron_nodes(G: nx.DiGraph, weight_shapes):
    """Create all nodes of a neural network from the weight matrices."""
    
    layers = []

    OUT, IN = weight_shapes[0]
    new_nodes = np.arange(IN) + len(G)
    G.add_nodes_from(new_nodes, layer = len(layers))
    layers.append(new_nodes)

    # go through all weight matrices of the network
    for OUT,IN in weight_shapes:
        new_nodes = np.arange(OUT) + len(G)
        G.add_nodes_from(new_nodes, layer = len(layers))
        layers.append(new_nodes)

        # create edges between all nodes of neighbouring layers
        for a in layers[-2]:
            for b in layers[-1]:
                G.add_edge(a, b)
