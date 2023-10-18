import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
from networkx import multipartite_layout

# BOKEH
from bokeh.plotting import figure
from bokeh.models import Circle, HoverTool,  MultiLine
from bokeh.plotting import figure, from_networkx, show
from bokeh.models import Slider, CustomJS, LinearColorMapper
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, DataTable
from bokeh.models.widgets import TableColumn

# plot func
import re
from common.nx_utils import load_state_dict
from common import tracking

# INDEPENDENT SUBNETWORKS N THE NETWORK
SUBNETWORK = "group"
GROUP_COLORS = [
    'red', 'green', 'blue', 'purple', 'orange',
    'pink', 'brown', 'gray', 'olive', 'cyan'
]

# EDGES
LINE_COLOR = 'line_color'
LINE_WIDTH = 'line_width'
LINE_ALPHA = 'line_alpha'

# NODES
LAYER='rank'
NODE_ALPHA = 'node_alpha'
NODE_COLOR = 'node_color'
NODE_LINE_COLOR = 'node_line_color'
NODE_LINE_ALPHA = 'node_line_alpha'
IS_ZOMBIE = 'zombie'
IS_HEADLESS = 'headless'

def _get_sorted_checkpoint_files(path):
    chkpts = list(path.glob("*.pt"))
    sort_by_integer_in_filename_key = lambda x : int(*re.findall("(\d+)",x.name))
    sorted_chkpts = sorted(chkpts, key=sort_by_integer_in_filename_key)
    return list(sorted_chkpts)

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

def _neurons_per_layer(layers_of_weights):
    _, input_dim = layers_of_weights[0].shape
    layers = [input_dim]

    # go through all weight matrices of the network
    for weight_matrix in layers_of_weights: 
        output_dim, input_dim = weight_matrix.shape
        layers.append(output_dim)

    return layers

def _make_graph(state_dict):
    """
    Create a networkx graph from torch state dict. 
    The Graph contains all Nodes and all Edges, even if 0.
    """

    weights = _weights_from_state_dict(state_dict)
    neurons_per_layer = _neurons_per_layer(weights)
    offset = 0
    layers = []

    G = nx.DiGraph()

    # add all the nodes
    for layer, vals in enumerate(neurons_per_layer):
        nodes = [(i, {LAYER: layer}) for i in range(offset, offset+vals)]
        layers.append(nodes)
        G.add_nodes_from(nodes)
        offset += vals


    # Generate all edges of the Neural Network
    edges = []

    # loop over all layers but the last one
    all_but_output_layer = range(len(layers)-1)
    for layer in all_but_output_layer:

        for i, (i_node_id, _) in enumerate(layers[layer]):
            for j, (j_node_id, _) in enumerate(layers[layer+1]):
                weight_matrix = weights[layer]

                # extract weight from matrix
                weight = weight_matrix[j,i].item()
                attrs = {'weight' : weight}

                # add the edge
                edge = (i_node_id, j_node_id, attrs)
                edges.append(edge)
    
    # Add the generated edges to the Graph Object
    G.add_edges_from(edges)
    return G

def _remove_isolated_nodes(G):
    """ This function removes all nodes from a NetworkX graph that do not have any edges connected to them.    """

    # Get a list of all nodes in the graph that have no edges
    isolated_nodes = [node for node in G.nodes if G.degree(node) == 0]

    # Remove the isolated nodes from the graph
    G.remove_nodes_from(isolated_nodes)

    return G

def _tag_nodes_incoming_outgoing_connections(G):
    
    H = _remove_isolated_nodes(G.copy()).to_undirected()
    subgraphs = [c for c in nx.connected_components(H)]
    output_layer = 0
    # Iterate through nodes to find the highest 'layer' value for each node
    for node in G.nodes():

        layer = G.nodes[node][LAYER]
        output_layer = layer if layer > output_layer else output_layer
        
        G.nodes[node][SUBNETWORK] = 'deeppink'
        for i, sub_G in enumerate(subgraphs):
            if node in sub_G:
                G.nodes[node][SUBNETWORK] = GROUP_COLORS[i]

    for node in G.nodes():  
        is_input_layer = G.nodes[node][LAYER] == 0
        is_output_layer = G.nodes[node][LAYER] == output_layer

        indegree = G.in_degree(node)
        outdegree =  G.out_degree(node)

        G.nodes[node]['incoming_connections'] = indegree
        G.nodes[node]['outgoing_connections'] = outdegree

        zombie, headless = False, False
        line_alpha = 0.
        alpha = 1.
        color = G.nodes[node][SUBNETWORK]

        if indegree == 0 and outdegree == 0:
            alpha = 0
            color = 'white'
            line_color = 'white'
            line_alpha = 0.

        elif indegree == 0 and not is_input_layer:
            zombie = True
            color = 'black'
            line_color = color
        elif outdegree == 0 and not is_output_layer:
            headless = True
            color='white'
            line_color = 'black'
            line_alpha = 1.
        else:
            line_color = 'black'
        
        G.nodes[node][NODE_ALPHA] = alpha
        G.nodes[node][NODE_COLOR] = color
        G.nodes[node][NODE_LINE_COLOR] = line_color
        G.nodes[node][NODE_LINE_ALPHA] = line_alpha
        G.nodes[node][IS_ZOMBIE] = zombie
        G.nodes[node][IS_HEADLESS] = headless
    return G

def _tag_edges_with_plus_minus_color(G):

    # sG = [G.subgraph(c) for c in nx.connected_components(G)]


    for u, v in G.edges():
        w = G.edges[u, v]['weight']

        alpha = 1.
        
        if G.nodes[u][IS_ZOMBIE]:
            color = 'black'
        elif G.nodes[v][IS_HEADLESS]:
            color = 'pink'
        elif w > 0:
            color = G.nodes[u][SUBNETWORK] #'darkturquoise'
        elif w < 0:
            color = G.nodes[u][SUBNETWORK] # 'teal'
        else:
            color = 'white'
            alpha = .0

        # Add the calculated x value as an attribute to the edge
        G.edges[u, v][LINE_ALPHA] = alpha
        G.edges[u, v][LINE_COLOR] = color
    return G

def _delete_zero_weight_edges(G):

    # Get a list of all nodes in the graph that have no edges
    zero_weight_edges = [edge for edge in G.edges(data=True) if edge[2]['weight'] == 0]
   
    G.remove_edges_from(zero_weight_edges)

    return G

def _layerlayout(G):
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(
            "requires pygraphviz " "http://pygraphviz.github.io/"
        ) from err

    A = nx.nx_agraph.to_agraph(G)
    last_layer = max([ attrs[LAYER] for id, attrs in G.nodes(data=True)])
    for layer in range(last_layer+1):
        nodes = [ str(id) for id, attrs in G.nodes(data=True) if attrs[LAYER] == layer]

        A.add_subgraph(nodes, rank='same')

    #output_nodes = [ str(id) for id, attrs in G.nodes(data=True) if attrs['layer'] == last_layer]
    #A.add_subgraph(input_nodes, rank='same')
    #A.add_subgraph(output_nodes, rank='same')
    A.layout(prog='dot')
    node_pos = {}
    for n in G:
        node = pygraphviz.Node(A, n)
        try:
            xs = node.attr["pos"].split(",")
            node_pos[n] = tuple(float(x) for x in xs)
        except:
            print("no position for node", n)
            node_pos[n] = (0.0, 0.0)
    return node_pos

def _rearranged_layout(G):
    items = _layerlayout(G).items()

    layers = []
    for id, (x,y) in items:
        if y not in layers:
            layers.append(y)

    # relabel nodes from the Gr
    for layer in layers:
        pos = [x for id, (x,y) in items if y==layer]
        nodes = np.array([id for id, (x,y) in items if y==layer])

        order = np.argsort(pos)
        original_order = nodes
        new_order = nodes[order]

        G = nx.relabel_nodes(G, {old: new for old, new in zip(original_order, new_order)})

    return nx.multipartite_layout(G, subset_key=LAYER)

def _make_datasources(checkpoint_dir):
    """Create Datasources from custom checkpoint directory."""
    # get the state dicts
    checkpoints = _get_sorted_checkpoint_files(checkpoint_dir)
    
    node_layouts, edges_data_sources, nodes_data_sources = [], [], []
    for path in checkpoints:
        # load the graph object from torch_state_dict
        state_dict = load_state_dict(path)

        G = _make_graph(state_dict)
        G = _delete_zero_weight_edges(G)
        G = _tag_nodes_incoming_outgoing_connections(G)
        G = _tag_edges_with_plus_minus_color(G)
        # G = _remove_isolated_nodes(G)

        # create the graph layout 
        #layout = graphviz_layout(G, prog='dot')
        #layout = _rearranged_layout(G)
        #layout = _layerlayout(G)
        layout = multipartite_layout(G, subset_key=LAYER)

        graph_renderer = from_networkx(G, layout)
        node_layouts.append(layout.copy())
        edges_data_sources.append(graph_renderer.edge_renderer.data_source.data.copy())
        nodes_data_sources.append(graph_renderer.node_renderer.data_source.data.copy())

    return graph_renderer, node_layouts, edges_data_sources, nodes_data_sources


# entry point
def plot_checkpoints(path):
    """Plot the interactive 2D Network Structure Graph."""
    if not path.exists(): raise ValueError(f"NOT EXIST: {path}")

    (
        renderer,
        node_layouts,
        edges_data_sources,
        nodes_data_sources
    ) = _make_datasources(path)

    plot = figure(title='NetworkX Graph', background_fill_color="#FFFFFF")

    plot.renderers.append(renderer)
    plot.grid.visible = False

    # Create a slider
    slider = Slider(
        start=0, 
        end=len(edges_data_sources)-1, 
        value=len(edges_data_sources)-1, 
        step=1, 
        title="Pruning Levels"
    )
    
    # Create a CustomJS callback
    callback = CustomJS(
        args=dict(
            renderer=renderer,
            layouts=node_layouts,
            edge_data=edges_data_sources,
            node_data=nodes_data_sources),
        code="""
            // Get the pruning iteration from the slider
            let i = cb_obj.value;

            // Update the data to current iteration and emit a change
            renderer.layout_provider.graph_layout = layouts[i];
            renderer.edge_renderer.data_source.data = edge_data[i];
            renderer.node_renderer.data_source.data = node_data[i];
            renderer.change.emit();
            """
    )
    slider.js_on_change('value', callback)

    # Style for Nodes
    renderer.node_renderer.glyph.update(
        size=25,
        fill_alpha=NODE_ALPHA,
        fill_color=NODE_COLOR,
        line_color=NODE_LINE_COLOR,
        line_alpha=NODE_LINE_ALPHA
    ) #  fill_color={'field': 'inout', 'transform' :node_mapper
    
    # Style for Edges
    mapper = LinearColorMapper(palette='Viridis256', low=-1, high=1)
    renderer.edge_renderer.glyph.update(
        line_width=5, 
        line_color=LINE_COLOR,
        line_alpha=LINE_ALPHA,
        #line_color={'field': 'weight', 'transform': mapper}
    )

    # Add hover tool
    hover = HoverTool(tooltips=[("index", "@index")])
    plot.add_tools(hover)
    
    # Convert the config dict to a Bokeh ColumnDataSource
    config_dict = tracking.load_hparams(path)
    source = ColumnDataSource({
        'names' : list(config_dict.keys()),
        'values' : list(config_dict.values())
    })

    # Define the columns in the DataTable
    columns = [TableColumn(field="names", title="Name"), TableColumn(field="values", title="Value")]
    data_table = DataTable(source=source, columns=columns, height=600)

    show(
        row(
            column(slider, plot), 
            column(data_table), 
        )
    )