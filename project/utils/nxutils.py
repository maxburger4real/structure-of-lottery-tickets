""" there are features, weights and biases
- a feature corresponds to a node or neuron
- a weight corresponds to an edge
- a bias is an attribute of a feature
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
from enum import Enum
from typing import List, Dict, Tuple
import plotly.graph_objects as go

ParamState = Enum(
    "ParamState", ["active", "inactive", "zombious", "zombie_downstream", "pruned"]
)

STATE = "state"


class GraphManager:
    """A class that manages the state over the pruning iterations"""

    def __init__(self, unpruned_model, shape, task_description, output_only=False):
        """Initialize the contant values of the Network, that will remain true over pruning iterations."""
        self.is_connected = self.is_split = self.is_degraded = False

        self.output_only = output_only
        self.shape = shape

        self.num_tasks = len(task_description) if task_description is not None else 1
        self.untapped_potential = self.num_tasks - 1
        self.split_iteration = None
        self.degradation_iteration = None

        self.G = build_nx(unpruned_model)
        self.pos = nx.multipartite_layout(self.G, "layer")
        self.in_features = find_nodes(self.G, layer=0)
        self.out_features = find_nodes(self.G, layer=len(shape) - 1)

        # map the tasks to the network features
        self.task_description = map_tasks_to_features(
            task_description, self.in_features, self.out_features
        )

        self.layerwise_split_metrics = {}

        self.graveyard = {}  # the pruned parameters are here
        self.catalogue: Dict[str, nx.Graph] = {}

        self.node_statistics, self.edge_statistics = statistics(self.G)

    def metrics(self):
        metrics = {}

        total_nodes = sum(
            [
                value
                for key, value in self.node_statistics.items()
                if key != ParamState.pruned
            ]
        )
        total_edges = sum(
            [
                value
                for key, value in self.edge_statistics.items()
                if key != ParamState.pruned
            ]
        )

        for state in ParamState:
            num_nodes = self.node_statistics[state]
            num_edges = self.edge_statistics[state]
            metrics[state.name + "-features" + "-abs"] = num_nodes
            metrics[state.name + "-features" + "-rel"] = num_nodes / total_nodes
            metrics[state.name + "-weights" + "-abs"] = num_edges
            metrics[state.name + "-weights" + "-rel"] = num_edges / total_edges
            metrics[state.name + "-rel"] = (num_edges + num_nodes) / (
                total_edges + total_nodes
            )
            metrics[state.name + "-abs"] = num_edges + num_nodes

        # log everything on split as well
        if self.iteration == self.split_iteration:
            items = list(metrics.items())
            for k, v in items:
                metrics[f"split/{k}"] = v

            for name, g in self.catalogue.items():
                metrics[name] = self.fig(g)

        metrics["untapped-potential"] = self.untapped_potential
        return metrics

    def update(self, model, iteration):
        self.iteration = iteration

        # tag the parameters based on new pruning mask
        tag_parameters(self.G, model, self.in_features, self.out_features)

        # split the network into subnetworks
        G = subgraph_by_state(self.G, include=[ParamState.active])
        subnetworks = [
            G.subgraph(c) for c in nx.connected_components(G.to_undirected())
        ]

        # WARNING: This only works for 2 tasks


        self.layerwise_split_metrics = {}
        self.remaining_in_and_outputs = {}
        if True and len(self.task_description) == 2:

            t0_name, (t0_in, t0_out) = self.task_description[0]
            t1_name, (t1_in, t1_out) = self.task_description[1]

            self.remaining_in_and_outputs[t0_name + '-out'] = len(set(t0_out).intersection(G.nodes))
            self.remaining_in_and_outputs[t0_name + '-in'] =  len(set(t0_in).intersection(G.nodes))
            self.remaining_in_and_outputs[t1_name + '-out'] = len(set(t1_out).intersection(G.nodes))
            self.remaining_in_and_outputs[t1_name + '-in'] =  len(set(t1_in).intersection(G.nodes))
            
            # from outputs
            out_t0 = set()
            out_t1 = set()
            for tout in t0_out:
                if G.has_node(tout):
                    out_t0.update(nx.ancestors(G, tout))
            for tout in t1_out: 
                if G.has_node(tout):
                    out_t1.update(nx.ancestors(G, tout))

            t0 = out_t0 - out_t1 # t0 and not t1
            t1 =  out_t1 - out_t0 # t1 and not t0
            t12 = out_t0 & out_t1
            
            for layer in range(len(self.shape)-1):
                n0 = len([n for n in t0 if G.has_node(n) and G.nodes[n]['layer'] == layer])
                n1 = len([n for n in t1 if G.has_node(n) and G.nodes[n]['layer'] == layer])
                n01 = len([n for n in t12 if G.has_node(n) and G.nodes[n]['layer'] == layer])
                ratio = (n0 + n1) / (n0 + n1 + n01)

                self.layerwise_split_metrics[f'outview-{layer}-{t0_name}'] = n0
                self.layerwise_split_metrics[f'outview-{layer}-{t1_name}'] = n1
                self.layerwise_split_metrics[f'outview-{layer}-decided'] = n0 + n1
                self.layerwise_split_metrics[f'outview-{layer}-undecided'] = n01
                self.layerwise_split_metrics[f'outview-{layer}-p-decided'] = ratio

            # FROM INPUTS
            bottom_t0 = set()
            bottom_t1 = set()
            for tin in t0_in: 
                if G.has_node(tin):
                    bottom_t0.update(nx.descendants(G, tin))
            for tin in t1_in: 
                if G.has_node(tin):
                    bottom_t1.update(nx.descendants(G, tin))

            t0 = bottom_t0 - bottom_t1  
            t1 =  bottom_t1 - bottom_t0 
            t12 = bottom_t0 & bottom_t1
            for layer in range(1, len(self.shape)):
                n0 = len([n for n in t0 if G.has_node(n) and G.nodes[n]['layer'] == layer])
                n1 = len([n for n in t1 if G.has_node(n) and G.nodes[n]['layer'] == layer])
                n01 = len([n for n in t12 if G.has_node(n) and G.nodes[n]['layer'] == layer])
                ratio = (n0 + n1) / (n0 + n1 + n01)

                self.layerwise_split_metrics[f'inview-{layer}-{t0_name}'] = n0
                self.layerwise_split_metrics[f'inview-{layer}-{t1_name}'] = n1
                self.layerwise_split_metrics[f'inview-{layer}-decided'] = n0 + n1
                self.layerwise_split_metrics[f'inview-{layer}-undecided'] = n01
                self.layerwise_split_metrics[f'inview-{layer}-p-decided'] = ratio

        # update the catalogue of subnetworks
        self.__update_catalogue(subnetworks)

        self.node_statistics, self.edge_statistics = statistics(self.G)

        self.is_connected = self.is_split = self.is_degraded = False
        if self.untapped_potential > 0:
            self.is_connected = True
        if self.untapped_potential == 0:
            self.is_split = True
        elif self.untapped_potential < 0:
            self.is_degraded = True

    def fig(
        self,
        G: nx.Graph = None,
        include: List[ParamState] = [ParamState.active],
        exclude: List[ParamState] = None,
    ) -> go.Figure:
        if G is None:
            G = self.G
        """Return a plotly figure of the graph in current state."""
        G = subgraph_by_state(G, include, exclude)
        return make_plotly_fig(G, self.pos)

    def __update_catalogue(self, subnetworks):
        # update the task matrix
        self.task_matrix = task_matrix(subnetworks, self.task_description, self.output_only)

        potential = np.sum(self.task_matrix) / len(self.task_description)

        # has full potential
        if np.isclose(potential, 1):
            assert all(
                i.is_integer() for i in self.task_matrix.flatten()
            ), f"values must be integers. got {self.task_matrix}"
            self.task_matrix = self.task_matrix.astype(int)

            # number of values larger than 1
            num_tasks_per_network = np.sum(self.task_matrix, axis=1)
            splits_remaining = np.sum(num_tasks_per_network - 1)
            self.untapped_potential = splits_remaining

            if (
                self.untapped_potential == 0
                and self.split_iteration is None
                and self.num_tasks > 1
            ):
                self.split_iteration = self.iteration

        # changed
        elif self.untapped_potential != potential - 1:
            if self.degradation_iteration is None:
                self.degradation_iteration = self.iteration
            self.untapped_potential = potential - 1

        # update the catalogue
        self.catalogue = {}
        for i, g in enumerate(subnetworks):
            name = ""
            network_tasks = self.task_matrix[i]
            for j, (task_name, _) in enumerate(self.task_description):
                task = network_tasks[j]
                if task == 0:
                    continue
                if name != "":
                    name += "-"
                name += f"{task_name}"

            self.catalogue[name] = g


def build_nx(model: torch.nn.Module) -> nx.DiGraph:
    """Layers from [0, 1, ... ,-1]"""
    G = nx.DiGraph()
    for previous_node_idx, module in enumerate(model.layers, start=1):
        num_current_nodes, num_previous_nodes = module.weight.shape

        # add 0-th layer (input features)
        if previous_node_idx == 1:
            previous_nodes = [(0, i) for i in range(num_previous_nodes)]
            G.add_nodes_from(previous_nodes, layer=0)

        current_nodes = [(previous_node_idx, i) for i in range(num_current_nodes)]
        biases = [bias.detach().item() for bias in module.bias]
        node_biases = dict(zip(current_nodes, biases))

        # create all nodes of the current output layer
        G.add_nodes_from(current_nodes, layer=previous_node_idx)
        nx.set_node_attributes(G, node_biases, "bias")

        # connect the nodes of current and previous layer
        for previous_layer, previous_node_idx in previous_nodes:
            for current_layer, current_node_idx in current_nodes:
                weight = module.weight[current_node_idx, previous_node_idx].item()
                G.add_edge(
                    u_of_edge=(previous_layer, previous_node_idx),
                    v_of_edge=(current_layer, current_node_idx),
                    weight=weight,
                )

        previous_nodes = current_nodes

    # set all weights and edges as active
    nx.set_node_attributes(G, ParamState.active, STATE)
    nx.set_edge_attributes(G, ParamState.active, STATE)
    return G


def subgraph_by_state(
    G: nx.DiGraph,
    include: List[ParamState] = None,
    exclude: List[ParamState] = None,
    copy=False,
):
    """allow either to include or exclude."""
    if include is None and exclude is None:
        raise ValueError("unnecessary function call.")
    if include is not None and exclude is not None:
        raise ValueError("Either use include or exclude, not both.")

    if include is not None:

        def criterion(attrs):
            return attrs[STATE] in include
    else:

        def criterion(attrs):
            return attrs[STATE] not in exclude

    edges = [
        (u, v)
        for u, v, attrs in G.edges(data=True)
        if STATE not in attrs or criterion(attrs)
    ]

    # Create the subgraph
    g = G.edge_subgraph(edges)

    if copy:
        return g.copy()

    return g


def task_matrix(subnetworks: List[nx.DiGraph], task_description, output_only: bool):
    L = len(task_description)
    N = len(subnetworks)
    X = np.ones(shape=(N, L)) * np.inf  # Just to be sure that it is always overwritten.

    for i, g in enumerate(subnetworks):
        for j, (name, (in_features, out_features)) in enumerate(task_description):
            # find the features contained in the subnetwork
            in_features_in_subnetwork = set(g.nodes()).intersection(in_features)
            out_features_in_subnetwork = set(g.nodes()).intersection(out_features)

            if output_only:
                X[i, j] = len(out_features_in_subnetwork) / len(out_features)
            else:
                X[i, j] = (
                    (len(in_features_in_subnetwork) * len(out_features_in_subnetwork))/
                    (len(in_features) * len(out_features))
                )

    return X


def make_plotly_fig(G: nx.Graph, pos):
    edge_traces = __make_edge_traces(G, pos)
    node_trace = __make_node_trace(G, pos)

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.update_layout(plot_bgcolor="white", autosize=False, width=400, height=400)
    # fig.write_image("large.svg")
    return fig


def statistics(G: nx.DiGraph) -> Tuple[Dict, Dict]:
    node_count = {state: 0 for state in ParamState}
    for _, data in G.nodes(data=True):
        state = data.get(STATE)
        assert state is not None
        node_count[state] += 1

    edge_count = {state: 0 for state in ParamState}
    for *_, data in G.edges(data=True):
        state = data.get(STATE)
        assert state is not None
        edge_count[state] = edge_count.get(state, 0) + 1

    return node_count, edge_count


def tag_parameters(G: nx.DiGraph, model: torch.nn.Module, in_features, out_features):
    # tag the newly pruned parameters first
    __tag_pruned(G, model)
    G_pruned = subgraph_by_state(G, exclude=[ParamState.pruned])

    # find inactive parameters on the pruned Graph
    __tag_inactive(G_pruned, out_features)
    G_active_with_zombious = subgraph_by_state(
        G, include=[ParamState.active, ParamState.zombious]
    )

    # find zombious parameters
    __tag_zombious(G_active_with_zombious, in_features)


def find_nodes(G: nx.Graph, **kwargs):
    nodes = []
    for node, data in G.nodes(data=True):
        add_node = False
        for key, value in kwargs.items():
            if key not in data:
                continue
            if data[key] == value:
                add_node = True

        if add_node:
            nodes.append(node)

    return nodes


def map_tasks_to_features(
    task_description: Tuple, in_features: List[int], out_features: List[int]
) -> List[Tuple[str, Tuple[int, int]]]:
    """Map the provided task description to the feature ids of the network."""
    # Single Task
    if task_description is None:
        return [("main", (in_features, out_features))]

    # Multiple tasks
    ret = []

    in_features_generator = (x for x in in_features)
    out_features_generator = (x for x in out_features)

    # for each task, take the specified number of features from the generator
    for name, (num_in, num_out) in task_description:
        task_in_features = [next(in_features_generator) for _ in range(num_in)]
        task_out_features = [next(out_features_generator) for _ in range(num_out)]
        ret.append((name, (task_in_features, task_out_features)))

    return ret


def __tag_pruned(G: nx.Graph, model: torch.nn.Module):
    for layer, module in enumerate(model.layers, start=1):
        # find all pruned weights
        indices = torch.nonzero(module.weight_mask == 0).tolist()

        # translate their indices to edges
        pruned_edges = [((layer - 1, i), (layer, j)) for j, i in indices]

        nx.set_edge_attributes(
            G=G.edge_subgraph(pruned_edges), values=ParamState.pruned, name=STATE
        )

        if not hasattr(module, "bias_mask"):
            continue

        indices = torch.nonzero(module.bias_mask == 0).tolist()
        pruned_nodes = [(layer, i) for (i,) in indices]

        nx.set_node_attributes(G.subgraph(pruned_nodes), ParamState.pruned, STATE)


def __tag_inactive(G: nx.DiGraph, out_features):
    assert isinstance(G, nx.DiGraph), "only works for Directed Graphs"

    # outfeatures are
    active = set(out_features)

    for node in out_features:
        if node not in G.nodes():
            continue
        active.update(nx.ancestors(G, node))

    inactive_features = set(G.nodes()).symmetric_difference(active)
    inactive_weights = set()

    for node in inactive_features:
        if node not in G.nodes():
            continue
        if G.nodes[node][STATE] != ParamState.pruned:
            G.nodes[node][STATE] = ParamState.inactive

        inactive_weights.update(G.in_edges(node))
        inactive_weights.update(G.out_edges(node))

    attrs = {edge: ParamState.inactive for edge in inactive_weights}

    # Set the attributes for the specified edges
    nx.set_edge_attributes(G, attrs, STATE)


def __tag_zombious(G: nx.DiGraph, in_features):
    """Assumes no pruned weights in the graph."""

    assert isinstance(G, nx.DiGraph), "only works for Directed Graphs"

    # features are connnected to at least one input feature
    nodes_with_input = set(in_features)

    # for each input feature, find all descendants and collect in set
    for in_feature in in_features:
        if in_feature not in G.nodes():
            continue
        descendants = nx.descendants(G, in_feature)
        nodes_with_input.update(descendants)

    nodes_without_input = set(G.nodes()).symmetric_difference(nodes_with_input)

    # sort the nodes based on number of input connections
    nodes_without_input_sorted = sorted(
        G.subgraph(nodes_without_input).in_degree, key=lambda x: x[1]
    )

    for node, in_degree in nodes_without_input_sorted:
        if node not in G.nodes():
            raise

        in_edges_states = [
            G.edges[e][STATE]
            for e in G.in_edges(node)
            if G.edges[e][STATE] != ParamState.inactive
        ]

        # all in_edges are inactive ->> turn the node and all out_edges inactive
        if not in_edges_states and G.nodes[node][STATE] == ParamState.pruned:
            for edge in G.out_edges(node):
                G.edges[edge][STATE] = ParamState.inactive

        # there is at least one zombie --> turn into zombie and all out_edges to zombious
        else:
            for edge in G.out_edges(node):
                G.edges[edge][STATE] = ParamState.zombious

            if G.nodes[node][STATE] != ParamState.pruned:
                G.nodes[node][STATE] = ParamState.zombious


def __make_node_trace(G, pos):
    node_x = []
    node_y = []
    colors = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        colors.append(__colormap(data[STATE]))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            color=colors,
            line_width=3,
            line_color="white",
        ),
    )
    return node_trace


def __make_edge_traces(G, pos):
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_traces = []
    for *edge, data in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        color = __colormap(data[STATE])
        rgba = to_rgba(color, alpha=0.5)
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            # line=dict(width=2*abs(data['weight']), color=rgba),
            # line=dict(width=.5, color=rgba),  # for very many weights
            line=dict(width=7, color=rgba),
            hoverinfo="none",
            mode="lines",
        )
        edge_traces.append(edge_trace)

    return edge_traces


def __colormap(state: str):
    if state == ParamState.active:
        return "#FFC107"
    if state == ParamState.inactive:
        return "#1E88E5"
    if state == ParamState.zombious:
        return "#D81B60"
    if state == ParamState.zombie_downstream:
        return "pink"
    if state == ParamState.pruned:
        return "black"
    raise


def to_rgba(colorname, alpha):
    rgb_color = mcolors.to_rgba(colorname)  # Convert color name to RGBA
    r, g, b, _ = [int(255 * comp) for comp in rgb_color]  # Scale to 0-255 range
    return f"rgba({r}, {g}, {b}, {alpha})"
