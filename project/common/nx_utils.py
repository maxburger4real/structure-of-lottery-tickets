import networkx as nx
from common.config import Config
from common.constants import *

def build_nx_graph(model, config: Config):
    """
    Create a networkx graph from torch state dict. 
    The Graph contains all Nodes and all Edges, even if 0.
    """

    _omit_zero_edges = True  # do not add edges with weight 0 to the graph.
    _omit_unconnected_nodes = True  # remove nodes that have no inputs and no outputs.

    weights, biases = _get_w_and_b(model)

    input_node_ids = list(range(config.model_shape[0]))
    input_nodes = [ (id, {LAYER:0, BIAS:0}) for id in input_node_ids]

    G = nx.DiGraph()
    G.add_nodes_from(input_nodes)
    layers = [input_node_ids]

    # add all the nodes
    for layer_id, bias in enumerate(biases, start=1):

        nodes = []
        for i, b in enumerate(bias, start=len(G)):
            node = i, { LAYER:layer_id, BIAS:b}
            nodes.append(node)

        G.add_nodes_from(nodes)
        layers += [[node_id for node_id, _ in nodes]]


    # Generate all edges of the Neural Network
    edges = []
    l = len(config.model_shape)

    # loop over all layers but the last one
    for layer_id in range(l-1):
        weight_matrix = weights[layer_id]

        for i, i_node_id in enumerate(layers[layer_id]):
            for j, j_node_id in enumerate(layers[layer_id+1]):
                w = weight_matrix[j,i].item()

                if _omit_zero_edges and w == 0: continue  # do not add pruned weights

                edge_attr = {WEIGHT : w}
                edge = (i_node_id, j_node_id, edge_attr)
                edges += [edge]
    
    # Add the generated edges to the Graph Object
    G.add_edges_from(edges)
    
    # remove nodes without any connections
    G = _remove_isolated_nodes(G)

    return G

def subnet_analysis(G: nx.DiGraph, config: Config):

    l = len(config.model_shape)

    H = G.to_undirected()

    input_features = [x for x, data in H.nodes(data=True) if data[LAYER]==0]
    output_features = [x for x, data in H.nodes(data=True) if data[LAYER]==l-1]

    # parse out the tasks of the network
    m = config.num_concat_datasets
    in_features_per_task = len(input_features)  / m
    out_features_per_task = len(output_features)  / m
    iit = [iter(input_features)] * int(in_features_per_task)
    oit = [iter(output_features)] * int(out_features_per_task)
    task_inputs, task_outputs = list(zip(*iit)), list(zip(*oit))

    # sort each component into a bin
    subnet_metadatas = []
    components = [H.subgraph(c) for c in nx.connected_components(H)]
    for c in components:

        input_layer_nodes = [x for x, data in c.nodes(data=True) if data[LAYER]==0]
        output_layer_nodes = [x for x, data in c.nodes(data=True) if data[LAYER]==l-1]
        
        # each component gets a subnet metadata dict
        subnet_metadata = {
            'input': { 'complete' : [], 'incomplete' : []},
            'output':{ 'complete' : [], 'incomplete' : []},
            'num_weights' : len(c.edges())
        }

        # check if the infeatures or outfeatures are contained in the subnetwork
        for i, (in_features, out_features) in enumerate(zip(task_inputs, task_outputs)):
            
            # inputs
            in_features_in_subnet = [n for n in in_features if n in input_layer_nodes]

            # all features are in the subnet
            if len(in_features_in_subnet) == len(in_features):
                subnet_metadata['input']['complete'].append(i)

            # some features are in the subnet
            elif 0 < len(in_features_in_subnet) < len(in_features):
                subnet_metadata['input']['incomplete'].append(i)

            # outputs
            out_features_in_subnet = [n for n in out_features if n in output_layer_nodes]

            # all outputs are in subnet
            if len(out_features_in_subnet) == len(out_features):
                subnet_metadata['output']['complete'].append(i)

            # some outputs are in subnet
            elif 0 < len(out_features_in_subnet) < len(out_features):
                subnet_metadata['output']['incomplete'].append(i)
        
        subnet_metadatas.append(subnet_metadata)

    return subnet_metadatas

def neuron_analysis(G: nx.DiGraph, config: Config):
    """
    - a neuron is a zombie if
        - it has no incoming connections but outgoing connections.
        - input feautres are no real neurons, therefore cannot be zombies.
    - a neuron is a comatose if
        - it has incoming connections, but no outgoing ones
        - the output neurons are exempt, because they alwaays have connections
    """
    l = len(config.model_shape)
    zombies, comatose = [], []
    for i, attrs in G.nodes(data=True):  
        node_is_not_in_input_layer = attrs[LAYER] != 0
        node_is_not_in_output_layer = attrs[LAYER] != l-1
        in_degree, out_degree = G.in_degree(i), G.out_degree(i)

        G.nodes[i]['in'] = in_degree
        G.nodes[i]['out'] = out_degree

        if node_is_not_in_input_layer and in_degree == 0:
            zombies.append(i)

        if node_is_not_in_output_layer and out_degree == 0:
            comatose.append(i)

    return zombies, comatose

# helpers
def _remove_isolated_nodes(G):
    """ This function removes all nodes from a NetworkX graph that do not have any edges connected to them.    """

    # Get a list of all nodes in the graph that have no edges
    isolated_nodes = [node for node in G.nodes if G.degree(node) == 0]

    # Remove the isolated nodes from the graph
    G.remove_nodes_from(isolated_nodes)

    return G

def _get_w_and_b(model):
    state_dict = model.state_dict()
    weights = _weights_from_state_dict(state_dict)
    biases = _biases_from_state_dict(state_dict)
    return weights, biases

def _biases_from_state_dict(state_dict):
    biases, masks = [], []

    for  k,v in state_dict.items():
        if 'bias_mask' in k:
            masks.append(v)
        elif 'bias_orig' in k:
            biases.append(v)
        elif 'bias' in k:
            biases.append(v)

    if masks:
        masked = [weight*mask for weight, mask in zip(biases, masks)]
        return masked
    elif biases:
        return biases

def _weights_from_state_dict(state_dict):
    weights, masks = [], []

    for  k,v in state_dict.items():
        if 'weight_mask' in k:
            masks.append(v)
        elif 'weight_orig' in k:
            weights.append(v)
        elif 'weight' in k:
            weights.append(v)

    if masks:
        masked = [weight*mask for weight, mask in zip(weights, masks)]
        return masked
    elif weights:
        return weights
