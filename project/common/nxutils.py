''' there are features, weights and biases
- a feature corresponds to a node or neuron
- a weight corresponds to an edge
- a bias is an attribute of a feature
'''

import networkx as nx
from common.config import Config
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

from common.constants import *

class GraphManager():
    '''A class that manages the state over the pruning iterations'''
    def __init__(self, model, shape, task_description):
        '''Initialize the contant values of the Network, that will remain true over pruning iterations.'''
        G = build_graph_from_model(model, shape)
        self.G = G
        self.shape = shape
        self.input_layer, self.output_layer = find_min_max_attribute(G, LAYER)

        # retreive the ids of the input and output features from G
        self.in_features, self.out_features = get_in_and_out_features(
            G, self.input_layer, self.output_layer
        )

        # map the tasks to the network features
        self.task_description = map_tasks_to_features(
            task_description, self.in_features, self.out_features
        )

        # a dict of parameters that are part of the zombie network, and their lifetime
        self.zombie_params = {'existing': {}, 'pruned': {}}

        # a list of parameters that are unproductive and their lifetime
        self.unproductive_params = {'existing': {}, 'pruned': {}}

        # a list of subnetworks described by their task descriptions, and the number of weights, biases
        self.subnetworks = {}

    def print_info(self):
        print(
f'''unpr : {self.unproductive_params}
zomb : {self.zombie_params}
subn : {len(self.subnetworks)}'''
        )    

    def __update_lifetimes(self, params, params_dict):
        existing_params = params_dict['existing'].keys()
        
        # new parameters from this update
        new_params = set(params).difference(existing_params)

        # parameters that existed but not anymore in this update
        lost_params = set(existing_params).difference(params)
        
        # parameters that have existed and still do after this update
        remaining_params = set(existing_params).intersection(params)

        # TODO: remove
        assert new_params.isdisjoint(lost_params)
        assert new_params.isdisjoint(remaining_params)
        assert lost_params.isdisjoint(remaining_params)

        for p in new_params: 
            params_dict['existing'][p] = 0
        
        for p in lost_params:
            assert p not in params_dict
            params_dict['pruned'][p] = params_dict['existing'].pop(p)

        for p in remaining_params:
            params_dict['existing'][p] += 1

    def __update_task_coverage(self, productive):
        task_coverage = []
        for g in productive:
            coverage = get_task_coverage(g, self.task_description, self.input_layer, self.output_layer)
            self.subnetworks[g] = coverage
            
            task_coverage.append(coverage)

    def __get_unproductive_and_zombie_params(self, productive):
        
        unproductive_params = []
        zombie_params = []
        for g in productive:
            pg = get_productive_subgraph(g, self.out_features)
            unprod_features, unprod_weights = get_difference(g, pg)
            unproductive_params += unprod_features + unprod_weights

            nzg = get_non_zombious_subnetwork(pg, self.in_features)
            zombie_features, zombie_weights = get_difference(pg, nzg)
            zombie_params += zombie_features + zombie_weights

            node_colors = []
            for feature in g.nodes():
                if feature in unproductive_params: node_colors.append('blue')
                elif feature in zombie_params: node_colors.append('red')
                else: node_colors.append('green')  # Default color for other nodes

            edge_colors = []
            for weight in g.edges():
                if weight in unproductive_params: edge_colors.append('blue')
                elif weight in zombie_params: edge_colors.append('red')
                else: edge_colors.append('green')  # Default color for other nodes

            pos=nx.multipartite_layout(self.G, LAYER)
            nx.draw(g, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors)
            plt.show()
        
        return unproductive_params, zombie_params

    def update(self, model):

        G = build_graph_from_model(model, self.shape)

        # split the network into subnetworks
        subnetworks = split_into_subnetworks(G)

        # sort the networks by productivity
        productive_subnetworks, unproductive_subnetworks = sort_by_productivity(
            subnetworks, self.out_features
        )

        # update lifetimes of the unproductive parameters
        params = []
        for g in unproductive_subnetworks:
            weights, biases = get_param_ids(g)
            params += weights + biases
        unproductive_params, zombie_params = self.__get_unproductive_and_zombie_params(productive_subnetworks)
        unproductive_params += params

        #print(f'#productive              : {len(productive_subnetworks)}')
        #print(f'#unproductive subnetworks: {len(unproductive_subnetworks)}')
        print(f'#unproductive params(frag): {len(params)}, {params}')
        print(f'#unproductive params(prod): {len(unproductive_params)}, {params}')

        self.__update_lifetimes(unproductive_params, self.unproductive_params)
        self.__update_lifetimes(zombie_params, self.zombie_params)
        self.__update_task_coverage(productive_subnetworks)

def build_graph_from_model(model, shape):
    
    G = nx.DiGraph()

    num_input_neurons = shape[0]
    in_features = list(range(num_input_neurons))

    for id in in_features:
        G.add_node(id, **{LAYER:0})

    # add all the nodes
    layers = [in_features]

    weights, biases = _get_w_and_b(model)

    # add all features of the models
    for layer_id, bias_vector in enumerate(biases, start=1):

        # list of feature ids for current layer
        feature_ids = list(range(len(G), len(bias_vector)+len(G)))

        # add them to the graph 
        for id, bias_term in zip(feature_ids, bias_vector, strict=True):
            attrs = {LAYER : layer_id}
            attrs[BIAS] = bias_term.item()

            G.add_node(id, **attrs)

        layers += [feature_ids]


    # Generate all edges of the Neural Network
    # loop over all layers but the last one
    for layer_id in range(len(shape)-1):

        weight_matrix = weights[layer_id]
        in_features = [id for id, data in G.nodes(data=True) if data[LAYER]==layer_id]
        out_features = [id for id, data in G.nodes(data=True) if data[LAYER]==layer_id+1]

        for i, in_feature_id in enumerate(in_features):
            for j, out_feature_id in enumerate(out_features):
                w = weight_matrix[j,i].item()
                
                if w == 0: continue  # do not add pruned weights, they do not exist
                attrs = {WEIGHT : w}
                G.add_edge(
                    in_feature_id, 
                    out_feature_id, 
                    **attrs
                )

    return G

def find_min_max_attribute(G: nx.DiGraph, key):
    values = [data[key] for _, data in G.nodes(data=True) if key in data]
    return min(values), max(values)

def get_in_and_out_features(G: nx.DiGraph, input_layer, output_layer):
    in_features  = [id for id, data in G.nodes(data=True) if data[LAYER]==input_layer]
    out_features = [id for id, data in G.nodes(data=True) if data[LAYER]==output_layer]
    return in_features, out_features

def split_into_subnetworks(G: nx.DiGraph):
    H = G.to_undirected()
    components = [G.subgraph(c) for c in nx.connected_components(H)]
    return components

def is_unproductive(G: nx.DiGraph, out_features):
    '''
    Check if the network contains any of the output features.
    If not, it is unproductive.
    (apparently fastest way https://stackoverflow.com/a/17735466)
    '''
    network_features = set(G.nodes())

    no_features_contained = network_features.isdisjoint(out_features)

    return no_features_contained

def sort_by_productivity(subnetworks: List[nx.DiGraph], out_features):
    '''Take a graph and split it into '''
    prod, nprod = [], []
    
    for g in subnetworks:
        nprod.append(g) if is_unproductive(g, out_features) else prod.append(g)

    return prod, nprod

def get_coverage(to_be_covered: List, cover: List):

    coverage = set(to_be_covered).intersection(cover)

    complete_coverage = (coverage == to_be_covered)

    if complete_coverage:
        return 'complete'
    
    partial_coverage = (len(coverage) > 0)
    if partial_coverage:
        return 'partial'
    
    return 'absent'

def get_task_coverage(
    G: nx.DiGraph, 
    task_description: Dict, 
    input_layer: int, 
    output_layer: int
) -> Dict:
    '''
    Returns a dict that describes the extent to which the network covers a task,
    meaning if it completely, partialy or not at all contains the input features or 
    output features of the tasks,
    '''
    
    data = {'in' : defaultdict(list),'out': defaultdict(list),}

    in_features  = [id for id, data in G.nodes(data=True) if data[LAYER]==input_layer]
    out_features = [id for id, data in G.nodes(data=True) if data[LAYER]==output_layer]

    # go through every task and sort by coverage
    for name, (task_in_features, task_out_features) in task_description.items():
        
        # The Inputs
        coverage = get_coverage(task_in_features, in_features)
        data['in'][coverage].append(name)

        # The Outputs
        coverage = get_coverage(task_out_features, out_features)
        data['out'][coverage].append(name)

    return data

def map_tasks_to_features(
    task_description: Tuple, 
    in_features: List, 
    out_features: List
) -> Dict:
    '''Map the provided task description to the feature ids of the network.'''
    # Single Task
    if task_description is None:
        return {'main' : (in_features, out_features)}
    
    # Multiple tasks
    ret = {}

    in_features_generator = (x for x in in_features)
    out_features_generator = (x for x in out_features)

    # for each task, take the specified number of features from the generator
    for name, (num_in, num_out) in task_description:

        task_in_features = [next(in_features_generator) for _ in range(num_in)]
        task_out_features = [next(out_features_generator) for _ in range(num_out)]
        ret[name] = (task_in_features, task_out_features)

    return ret

def get_param_ids(G: nx.Graph):
    '''Get the identification of each parameter in the graph.
    TODO: currently checks if parameters are 0. but should check if the mask is 1.
    '''
    weights = [(u,v) for u,v,data in G.edges(data=True) if data[WEIGHT] != 0]
    biases = [feature for feature, data in G.nodes(data=True) if BIAS in data and data[BIAS] != 0]
    return weights, biases

def get_productive_subgraph(G: nx.DiGraph, out_features):
    assert isinstance(G, nx.DiGraph), 'only works for Directed Graphs'
    productive = set(out_features)
    for feature in out_features:

        if feature not in G.nodes():
            continue
        
        ancestors = nx.ancestors(G, feature)
        productive.update(ancestors)

    return G.subgraph(productive)

def get_non_zombious_subnetwork(G: nx.DiGraph, in_features):
    assert isinstance(G, nx.DiGraph), 'only works for Directed Graphs'

    # features are connnected to at least one input feature
    non_zombious_features = set(in_features)

    # for each input feature, find all descendants and collect in set
    for in_feature in in_features:
        if in_feature not in G.nodes(): continue
        descendants = nx.descendants(G, in_feature)
        non_zombious_features.update(descendants)

    return G.subgraph(non_zombious_features)

def get_difference(G: nx.Graph, H: nx.Graph):
    '''Get all parameters that are not in both Graphs'''
    diff_nodes = set(G.nodes()).symmetric_difference(H.nodes())
    diff_edges = set(G.edges()).symmetric_difference(H.edges())
    return list(diff_nodes), list(diff_edges)

    
# helpers
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
