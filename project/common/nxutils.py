''' there are features, weights and biases
- a feature corresponds to a node or neuron
- a weight corresponds to an edge
- a bias is an attribute of a feature
'''

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt

from common.constants import *

class GraphManager():
    '''A class that manages the state over the pruning iterations'''
    ALIVE, AUDIENCE, UNPRODUCTIVE, ZOMBIE, PRUNED = range(5)

    def __init__(self, unpruned_model, shape, task_description):
        '''Initialize the contant values of the Network, that will remain true over pruning iterations.'''
        self.shape = shape
        self.iteration = 0
        self.graveyard = {}  # the pruned parameters are here

        self.G = build_graph_from_model(unpruned_model, shape)
        self.input_layer, self.output_layer = _find_min_max_attribute(self.G, LAYER)

        # retreive the ids of the input and output features from G
        self.in_features, self.out_features = get_in_and_out_features(self.G, self.input_layer, self.output_layer)

        # map the tasks to the network features
        self.task_description = map_tasks_to_features(task_description, self.in_features, self.out_features)

        # a dictionary with keys as parameter ids and values are 4-Tuples, that contain the number of iterations 
        # the parameter was in each state. The order is implicit. The sum of the numbers is the number of iterations this parameter survived
        biases: Set = set(self.G.nodes()) - set(self.in_features)
        weights: Set = set(self.G.edges())
        params: Set = set().union(biases, weights)

        self.lifecycles = {k:{'current_state' : self.ALIVE, 'lifecycle': [0,0,0,0]} for k in params}

    def update(self, model):
        self.G = build_graph_from_model(model, self.shape)

        # split the network into subnetworks
        subnetworks = split_into_subnetworks(self.G)

        # sort the networks by productivity (productive means has output connections)
        productive_subnetworks, fragments = sort_by_productivity(
            subnetworks, self.out_features
        )

        # fragments are seperate graphs that contain parameters
        fragment_params = []
        for g in fragments:
            params = _filter_pruned_params(g)
            fragment_params.extend(params)

        # sort all params into the following categories
        unproductive_params, zombie_params, alive_params, audience_params = [], [], [], []
        for g in productive_subnetworks:
            
            # sort the parameters into the buckets of states
            alive, audience, unproductive, zombies = sort_network_parameters(g, self.in_features, self.out_features)
            alive_params.extend(alive)
            audience_params.extend(audience)
            unproductive_params.extend(unproductive)
            zombie_params.extend(zombies)

        unproductive.extend(fragment_params)

        print(f'#subnets : {len(productive_subnetworks)}')
        tX = task_matrix(productive_subnetworks, self.task_description)

        potential = np.sum(tX) / len(self.task_description)
        print(tX)

        full_potential = np.isclose(potential, 1)
        print(f'potential : {potential}, full={full_potential}')

        # mitigate floating point errors

        self.__update_lifecycle(alive_params, audience_params, unproductive_params, zombie_params)
                
        self.iteration += 1

    def print_info(self):
        print(f'unproductive :{len(self.unproductive_params_list)} {self.unproductive_params_list}')
        print(f'zomb :{len(self.zombie_params_list)} {self.zombie_params_list}')

    def plot(self):
        '''For debugging purposes. Plots the Neural Network graph.
        - green : productive (or at least not proven otherwise)
        - black : pruned
        - red   : zombie
        - blue  : unproductive
        '''
        node_colors = []
        for feature in self.G.nodes():
            if feature in self.unproductive_params_list: node_colors.append('blue')
            elif feature in self.audience_params_list: node_colors.append('pink')
            elif feature in self.zombie_params_list: node_colors.append('red')
            elif feature in self.alive_params_list: node_colors.append('green')  # Default color for other nodes
            elif feature in self.pruned_params_list: node_colors.append('black')  # Default color for other nodes
            else: node_colors.append('yellow')  # Default color for other nodes

        edge_colors = []
        for weight in self.G.edges():
            if weight in self.unproductive_params_list: edge_colors.append('blue')
            elif weight in self.zombie_params_list: edge_colors.append('red')
            elif weight in self.audience_params_list: edge_colors.append('pink')
            elif weight in self.alive_params_list: edge_colors.append('green')  # Default color for other nodes
            elif weight in self.pruned_params_list: edge_colors.append('black')  # Default color for other nodes
            else: edge_colors.append('yellow')  # Default color for other nodes

        pos=nx.multipartite_layout(self.G, LAYER)
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors)
        plt.show()

    def __update_lifecycle(self, alive, audience, unproductive, zombie):
        # TODO: for testing
        l_before = len(self.lifecycles.keys()) + len(self.graveyard.keys())

        for p in list(self.lifecycles.keys()):
            if p in alive:
                self.lifecycles[p]['lifecycle'][self.ALIVE] += 1
                self.lifecycles[p]['current_state'] = self.ALIVE
            elif p in audience:
                self.lifecycles[p]['lifecycle'][self.AUDIENCE] += 1
                self.lifecycles[p]['current_state'] = self.AUDIENCE
            elif p in zombie:
                self.lifecycles[p]['lifecycle'][self.ZOMBIE] += 1
                self.lifecycles[p]['current_state'] = self.ZOMBIE
            elif p in unproductive:
                self.lifecycles[p]['lifecycle'][self.UNPRODUCTIVE] += 1
                self.lifecycles[p]['current_state'] = self.UNPRODUCTIVE
            elif p not in self.in_features:
                self.graveyard[p] = {
                    'survived_iterations': self.iteration,
                    'lifecycle':self.lifecycles[p]['lifecycle']
                }
                del self.lifecycles[p]
            else:
                # p is a in_feature, and they have no parameters.
                pass

        l_after = len(self.lifecycles.keys()) + len(self.graveyard.keys())

        assert l_before == l_after

    @property
    def unproductive_params_list(self):
        return [k for k,v in self.lifecycles.items() if v['current_state'] == self.UNPRODUCTIVE]
    @property
    def zombie_params_list(self):
        return [k for k,v in self.lifecycles.items() if v['current_state'] == self.ZOMBIE]
    @property
    def audience_params_list(self):
        return [k for k,v in self.lifecycles.items() if v['current_state'] == self.AUDIENCE]
    @property
    def alive_params_list(self):
        return [k for k,v in self.lifecycles.items() if v['current_state'] == self.ALIVE]
    @property
    def pruned_params_list(self):
        return list(self.graveyard.keys())


def task_matrix(subnetworks :List[nx.DiGraph], task_description):
    
    L = len(task_description)
    X = np.zeros(shape=(L,L))
    #X_out = np.zeros_like(X_in)

    for i, g in enumerate(subnetworks):
        for j, (name, (in_features, out_features)) in enumerate(task_description.items()):

            # find the features contained in the subnetwork
            in_features_in_subnetwork = set(g.nodes()).intersection(in_features)
            out_features_in_subnetwork = set(g.nodes()).intersection(out_features)

            # their product is the number of inout pairs in the network.
            num_in_out_pairs = len(in_features_in_subnetwork) * len(out_features_in_subnetwork)

            percentage_num_in_out_pairs = num_in_out_pairs / (len(in_features)*len(out_features))

            X[i,j] = percentage_num_in_out_pairs
            #X_out[i,j] = ratio_out_features

    return X

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

def get_in_and_out_features(G: nx.DiGraph, input_layer, output_layer) -> Tuple[List[int], List[int]]:
    in_features  = [id for id, data in G.nodes(data=True) if data[LAYER]==input_layer]
    out_features = [id for id, data in G.nodes(data=True) if data[LAYER]==output_layer]
    return in_features, out_features

def split_into_subnetworks(G: nx.DiGraph):
    H = G.to_undirected()
    components = [G.subgraph(c) for c in nx.connected_components(H)]
    return components

def sort_network_parameters(
    G: nx.DiGraph, 
    in_features: List[int], 
    out_features: List[int]
) -> Tuple[List, List, List]:
    '''Sort the parameters of into alive, zombie and unproductive'''    

    G_prod, unproductive = get_productive_subnetwork(G, out_features)

    G_alive, audience, _unproductive, zombies = get_alive_subnetwork(G_prod, in_features)

    # add the unproductive parameters from the productive subnetwork
    assert lists_are_mutually_disjoint(unproductive, _unproductive)
    unproductive += _unproductive

    alive = _filter_pruned_params(G_alive)

    assert lists_are_mutually_disjoint(alive, audience, unproductive, zombies)

    return alive, audience, unproductive, zombies

def sort_by_productivity(subnetworks: List[nx.DiGraph], out_features):
    '''Take a graph and split it into '''
    prod, nprod = [], []
    
    for g in subnetworks:
        prod.append(g) if _contains_out_features(g, out_features) else nprod.append(g)

    return prod, nprod

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

def get_productive_subnetwork(G: nx.DiGraph, out_features) -> Tuple[nx.DiGraph, List]:
    assert isinstance(G, nx.DiGraph), 'only works for Directed Graphs'
    
    productive = set(out_features)
    for feature in out_features:

        if feature not in G.nodes():
            continue
        
        ancestors = nx.ancestors(G, feature)
        productive.update(ancestors)

    G_prod = G.subgraph(productive)

    # get all nodes and edges that are not in the productive subnetwork
    nodes, edges = _get_difference(G, G_prod)

    # get all non-pruned parameters from those nodes and edges
    params = _filter_pruned_params(G, nodes, edges)

    return G_prod, params

def get_alive_subnetwork(G: nx.DiGraph, in_features) -> Tuple[nx.DiGraph, List, List, List]:
    '''Take the subgraph of the network with all nodes/features that
    are reachable from at least on of the input features.
    Return this subgraph G_alive and the ids of the parameters that are unproductive, zombies or their audiences.

    Return G_alive, audience, unproductive, zombies
    '''
    assert isinstance(G, nx.DiGraph), 'only works for Directed Graphs'

    # features are connnected to at least one input feature
    alive_features = set(in_features)

    # for each input feature, find all descendants and collect in set
    for in_feature in in_features:
        if in_feature not in G.nodes(): continue
        descendants = nx.descendants(G, in_feature)
        alive_features.update(descendants)

    G_alive = G.subgraph(alive_features)

    # get all nodes and edges that are not in the alive subgraph
    nodes, edges = _get_difference(G, G_alive)

    if not nodes and not edges:
        return G_alive, [], [], []
    
    print('not all nodes in this graph are alive')

    # get all unpruned parameters of those zombies
    # zombie_nodes, zombie_edges = _filter_pruned_params(G, nodes, edges, merge=False)

    # canonicalize the zombies
    G_Dangling = G.subgraph(nodes).copy()
    G_Dangling.add_edges_from(edges)

    extra_nodes = set(nodes).symmetric_difference(G_Dangling.nodes())

    # G_Dangling might have some nodes, that are added through adding the edges, yet belong to G_alive.
    audience, unproductive, zombies = seperate_zombies_audience_unproductive(G_Dangling, extra_nodes)

    # TEST if the graphs are indeed
    assert set(G_Dangling.edges()).isdisjoint(G_alive.edges())
    assert nx.is_isomorphic(nx.compose(G_alive, G_Dangling), G)

    return G_alive, audience, unproductive, zombies

def seperate_zombies_audience_unproductive(G: nx.DiGraph, borrowed_features: List[int]):
    '''
    Sort each parameter in a graph into buckets: 
    - zombies      : dangling features with bias != 0 (ignore RELu dangling)
    - audience     : parameters (w, b) that are nonzero and have input connections
    - unproductive : parameters that do not contribute to the output
                     - dangling feature with bias == 0
                     - weight connected to unproductive feature
    
    Assuming that the Graph is productive (all nodes connected to output features)
    '''
    unpruned_params = _filter_pruned_params(G)
    unproductive, zombies = [], []
    H = G.copy()
    # find Dangling features
    done = False



    while not done:
        done = True

        # find the dangling features (no input connections) and remove the ones that dont do anything
        features = list(H.nodes())
        for i in features:

            # feature is already categorized in other graph.
            if i in borrowed_features: 
                continue

            feature_is_dangling = (H.in_degree(i) == 0)
            if not feature_is_dangling: 
                continue

            feature_has_bias = i in unpruned_params
            if feature_has_bias: 
                zombies.append(i)
                continue
            
            # feature is not productive -> put edges in the unproductive bucket and remove feature from the graph
            unproductive.extend(H.edges(i))
            H.remove_node(i)
            done = False

    # TODO: Remove
    for zf in zombies:
        assert H.nodes[zf][BIAS] != 0, 'There is a Zombie without a bias. That is impossible.'

    audience = [i for i in H.nodes() if i not in zombies and i not in borrowed_features] + list(H.edges())

    assert lists_are_mutually_disjoint(audience, unproductive, zombies)

    return audience, unproductive, zombies


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

def _find_min_max_attribute(G: nx.DiGraph, key):
    values = [data[key] for _, data in G.nodes(data=True) if key in data]
    return min(values), max(values)

def _contains_out_features(G: nx.DiGraph, out_features):
    '''
    Check if the network contains any of the output features.
    If not, it is unproductive.
    (apparently fastest way https://stackoverflow.com/a/17735466)
    '''
    network_features = set(G.nodes())

    contains_out_features = not network_features.isdisjoint(out_features)

    return contains_out_features
  
def _get_difference(G: nx.Graph, H: nx.Graph):
    '''Get all parameters that are not in both Graphs'''
    diff_nodes = set(G.nodes()).symmetric_difference(H.nodes())
    diff_edges = set(G.edges()).symmetric_difference(H.edges())
    return list(diff_nodes), list(diff_edges)

def _filter_pruned_params(G: nx.Graph, nodes: List[int] = None, edges: List[Tuple[int,int]] = None, merge=True) -> List:
    '''Filter out pruned parameters, namely params that are 0. Either a list of nodes and edges is provided, then they are filtered. Or the nodes and edges of the provided graph are filtered.'''
    if nodes is None:
        nodes = G.nodes()
    if edges is None:
        edges = G.edges()

    filtered_nodes = [i for i in nodes if BIAS in G.nodes[i] and G.nodes[i][BIAS] != 0]

    # TODO: since there are no unpruned edges in the graph, this can be omitted.
    # filtered_edges = [edge for edge in edges if G.edges[edge][WEIGHT] != 0]
    filtered_edges = list(edges)
    if not merge: return filtered_nodes, filtered_edges

    return filtered_nodes + filtered_edges


# GENERAL UTILS for testing and so on.
def lists_are_mutually_disjoint(*lists):
    combined = set()
    for lst in lists:
        lst_set = set(lst)

        if not combined.isdisjoint(lst_set):
            return False
        
        combined.update(lst_set)
    return True