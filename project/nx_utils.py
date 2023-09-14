import networkx as nx
import numpy as np
from itertools import count
from torch import load
from pathlib import Path

STATE_DICT = "model_state_dict"
POSITIVE_COLOR = "blue"
NEGATIVE_COLOR = "red"


def positive_negative(G, edge, weight_pos, matrix_idx):
    #color = POSITIVE_COLOR if weight > 0 else NEGATIVE_COLOR
    #return "color", color
    pass

def alpha_value(weights):
    norm_weights = [ np.abs(w)/ np.abs(w).max() for w in weights]

    def func(G, edge, weight_pos, matrix_idx):
        return "alpha", norm_weights[matrix_idx][weight_pos].item()

    return func

def load_state_dict_from_file(file: Path):
    return load(file)[STATE_DICT]

def get_shape_from_state_dict(state_dict):
    shapes = [v.shape for k,v in state_dict.items() if 'weight' in k]
    return shapes

def torch_state_dict_unpacker(state_dict):
    weights = [v for k,v in state_dict.items() if 'weight' in k]
    biases = [v for k,v in state_dict.items() if 'bias' in k]
    return weights, biases

def add_weight_edges(
    G: nx.DiGraph,
    list_of_weights, 
    layers_to_nodes_dict, 
    iteration,
    attribute_functions,  # list of functions that return attribute and value
):
    
    attrs = {}

    for matrix_idx, _ in enumerate(list_of_weights):

        in_layer = layers_to_nodes_dict[matrix_idx]
        out_layer = layers_to_nodes_dict[matrix_idx+1]

        for i, in_node in enumerate(in_layer):
            for o, out_node in enumerate(out_layer):
                weight_pos = (o,i)
                edge = (in_node, out_node)

                attrs[edge] = {}
                for func in attribute_functions:
                    name, attribute = func(G, edge, weight_pos, matrix_idx)
                    attrs[edge][f'{name}-{iteration}'] = attribute

    nx.set_edge_attributes(G, attrs)

def add_weight_edges_deprecated(G: nx.DiGraph, list_of_weights, layers, idx):
    # go through all weight matrices of the network
    edge_attribute_values = {}

    for i, W in enumerate(list_of_weights):

        in_layer = layers[i]
        out_layer = layers[i+1]

        OUT,IN = W.shape
        for i in range(IN):
            for o in range(OUT):
                w = W[o][i]
                edge = (in_layer[i], out_layer[o])
                edge_attribute_values[edge] = {f'weight_{idx}':w.item()}

    nx.set_edge_attributes(G, edge_attribute_values)

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

def get_dict_of_node_layers(G: nx.DiGraph):
    """
    Returns a dictionary of Attributes as keys, 
    with a list of nodes that have that attribute as the value
    """
    mydict = nx.get_node_attributes(G, 'layer')
    reversed_dict = {}
    for key, value in mydict.items():
        reversed_dict.setdefault(value, [])
        reversed_dict[value].append(key)

    return reversed_dict

def add_neuron_nodes_deprecated(G: nx.DiGraph, list_of_weights):
    """Create all nodes of a neural network from the weight matrices."""
    node_id = count(0)

    layers = []

    # go through all weight matrices of the network
    for W in list_of_weights:

        OUT,IN = W.shape

        if not layers:
            layer = len(layers)
            layers.append([])
            for _ in  range(IN):
                node = next(node_id)
                G.add_node(node, layer=layer)
                layers[-1].append(node)


        layer = len(layers)
        layers.append([])
        for _ in  range(OUT):
            node = next(node_id)
            G.add_node(node, layer=layer)
            layers[-1].append(node)

        # create edges between all nodes of neighbouring layers
        for a in layers[-2]:
            for b in layers[-1]:
                G.add_edge(a, b)

    return layers