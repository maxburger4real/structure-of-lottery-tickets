from networkx import multipartite_layout, spring_layout, kamada_kawai_layout, planar_layout
import numpy as np
import networkx as nx
import json
from bokeh.plotting import figure
from bokeh.models import Circle, HoverTool,  MultiLine
from bokeh.plotting import figure, from_networkx, show
from bokeh.models import Slider, CustomJS, LinearColorMapper, Div
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, DataTable, StringFormatter, NumberFormatter, StringEditor, NumberEditor, SelectEditor, IntEditor
from bokeh.models.widgets import TableColumn

from networkx.drawing.nx_agraph import graphviz_layout, write_dot

# plot func
import re
import torch
from common.nx_utils import( 
    get_weights_from_state_dict, 
    load_state_dict, 
    get_shape_from_state_dict,
    add_neuron_nodes,
    black_and_white,
    add_weight_edges_arrays,
    get_layers_of_nodes,
    layerwise_normalized_abs_value
)

from common import tracking



LINE_COLOR = 'line_color'
LINE_COLOR_LIST = 'line_color_list'
LINE_WIDTH_LIST = 'line_width_list'
LINE_WIDTH = 'line_width'
LINE_ALPHA = 'line_alpha'
LAYER='rank'


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

    weights = _weights_from_state_dict(state_dict)
    neurons_per_layer = _neurons_per_layer(weights)
    offset = 0
    layers = []

    G = nx.DiGraph()

    # add all the nodes
    for layer, vals in enumerate(neurons_per_layer):
        nodes = [(i, {'layer': layer}) for i in range(offset, offset+vals)]
        layers.append(nodes)
        G.add_nodes_from(nodes)

        offset += vals

    # add all the edges with weights
    edges = []
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
    
    G.add_edges_from(edges)
    return G

def _remove_isolated_nodes(G):
    """ This function removes all nodes from a NetworkX graph that do not have any edges connected to them.    """

    # Get a list of all nodes in the graph that have no edges
    isolated_nodes = [node for node in G.nodes if G.degree(node) == 0]

    # Remove the isolated nodes from the graph
    G.remove_nodes_from(isolated_nodes)

    return G

def _delete_zero_weight_edges(G):

    # Get a list of all nodes in the graph that have no edges
    zero_weight_edges = [edge for edge in G.edges(data=True) if edge[2]['weight'] == 0]
   
    G.remove_edges_from(zero_weight_edges)

    return G

def _graph_from_state_dict(state_dict):
    """Retrieve a Graph with weights from state_dict."""
    G = _make_graph(state_dict)

    # transform
    G = _delete_zero_weight_edges(G)
    G = _remove_isolated_nodes(G)

    return G

def _layerlayout(G):
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(
            "requires pygraphviz " "http://pygraphviz.github.io/"
        ) from err

    A = nx.nx_agraph.to_agraph(G)
    last_layer = max([ attrs['layer'] for id, attrs in G.nodes(data=True)])
    for layer in range(last_layer+1):
        nodes = [ str(id) for id, attrs in G.nodes(data=True) if attrs['layer'] == layer]

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

    return nx.multipartite_layout(G, subset_key='layer')


def _renderer_data_from_graph(G):

    #layout = graphviz_layout(G, prog='dot')
    layout = _rearranged_layout(G)
    #layout = _layerlayout(G)

    graph_renderer = from_networkx(G, layout)

    return (
        graph_renderer,
        layout.copy(),
        graph_renderer.edge_renderer.data_source.data.copy(),
        graph_renderer.node_renderer.data_source.data.copy()
    )

def plot_checkpoints(path):

    config: tracking.Config = tracking.load_hparams(path)

    # get the state dicts
    checkpoints = _get_sorted_checkpoint_files(path)
    
    node_layouts, edges_data_sources, nodes_data_sources = [], [], []

    for path in checkpoints:
        G = _graph_from_state_dict(load_state_dict(path))

        (
            graph_renderer,
            node_layout, 
            edge_data_source, 
            node_data_source
        ) = _renderer_data_from_graph(G)

        node_layouts.append(node_layout)
        edges_data_sources.append(edge_data_source)
        nodes_data_sources.append(node_data_source)

    plot = figure(
        title='NetworkX Graph', 
        background_fill_color="#000000",
        #height=600,
        #width=1000
    )
    plot.renderers.append(graph_renderer)
    plot.grid.visible = False

    # Create a slider
    slider = Slider(
        start=0, 
        end=len(checkpoints)-1, 
        value=len(checkpoints)-1, 
        step=1, 
        title="Pruning Levels"
    )
    # Create a CustomJS callback
    callback = CustomJS(
        args=dict(
            graph_renderer=graph_renderer,
            layouts=node_layouts,
            edge_renderers_sources=edges_data_sources,
            node_renderers_sources=nodes_data_sources),
        code="""
            // Get the current slider value
            var value = cb_obj.value;

            // Update the graph layout and emit a change
            graph_renderer.layout_provider.graph_layout = layouts[value];
            graph_renderer.edge_renderer.data_source.data = edge_renderers_sources[value];
            graph_renderer.node_renderer.data_source.data = node_renderers_sources[value];
            graph_renderer.change.emit();
            """
    )
    slider.js_on_change('value', callback)

    # Style for Nodes
    graph_renderer.node_renderer.glyph.update(size=25) #  fill_color={'field': 'inout', 'transform' :node_mapper
    
    # Style for Edges
    mapper = LinearColorMapper(palette='Viridis256', low=-1, high=1)
    graph_renderer.edge_renderer.glyph.update(
        line_width=5, 
        line_color={'field': 'weight', 'transform': mapper}
    )

    # Add hover tool
    hover = HoverTool(tooltips=[("index", "@index")])
    plot.add_tools(hover)
    
    # Convert the dictionary to a Bokeh ColumnDataSource
    names = list(config.to_dict().keys())
    values = list(config.to_dict().values())
    source = ColumnDataSource({'names':names, 'values':values})

    # Define the columns in the DataTable
    columns = [TableColumn(field="names", title="Name"), TableColumn(field="values", title="Value")]
    data_table = DataTable(source=source, columns=columns, height=600)

    layout = row(
        column(slider, plot), 
        column(data_table), 
        )
    show(layout)

    return G 


# deprecated
def make_the_graph(path):
    chkpts = list(path.glob("*.pt"))
    sort_by_integer_in_filename_key = lambda x : int(*re.findall("(\d+)",x.name))
    sorted_chkpts = sorted(chkpts, key=sort_by_integer_in_filename_key)

    state_dict = load_state_dict(chkpts[0])
    weight_shapes = get_shape_from_state_dict(state_dict)

    # retrieve weights from state_dicts
    state_dicts = [load_state_dict(f) for f in sorted_chkpts]
    weights = [get_weights_from_state_dict(sd)[0] for sd in state_dicts]
    masks = [get_weights_from_state_dict(sd)[1] for sd in state_dicts]

    # reshape weights to include "time"-dimension as the first dim. 
    layers_of_weights = [torch.stack(x) for x in zip(*weights)]
    layers_of_masks = [torch.stack(x) for x in zip(*masks)]

    layers_of_mask_weights = [w*m for w,m in zip(layers_of_weights, layers_of_masks)]

    # add functions with names, which add attributes to the graph
    attribute_functions = {
        LINE_COLOR_LIST : black_and_white(layers_of_mask_weights),
        LINE_WIDTH_LIST : layerwise_normalized_abs_value(layers_of_mask_weights)
        }

    # create a graph and populate with nodes
    G = nx.DiGraph()
    add_neuron_nodes(G, weight_shapes)
    add_weight_edges_arrays(G, get_layers_of_nodes(G), attribute_functions)

    return G

def draw_interactive_mlp_graph(G):

    #layout_function = spring_layout(G)
    #layout_function = kamada_kawai_layout(G)
    #layout_function = planar_layout(G)
    #layout_function = graphviz_layout(G, prog='dot')
    
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout, write_dot

    # write_dot(G,'test.dot')
    # TODO: find a better way
    num_timesteps = list(nx.get_edge_attributes(G, 'line_width_list').values())[0].__len__()
    positions_list = []
    for i in range(num_timesteps):
        edges_with_weight = [(u, v) for u, v, d in G.edges(data=True) if d['line_width_list'][i] != 0]
        # Create a subgraph with these edges
        G_subgraph = G.edge_subgraph(edges_with_weight)
        positions = multipartite_layout(G_subgraph, subset_key=LAYER)
        positions_list.append(positions)


    # G_iso = list(nx.isolates(G_subgraph))

    # positions = multipartite_layout(G, subset_key=LAYER)
    
    # subset_nodes = [n for n, d in G.nodes(data=True) if 'your_attribute' in d]

    graph = from_networkx(
        G, 
        layout_function=positions_list[0]
        )

    # graph.edge_renderer.data_source.data['layouts_list'] =  positions_list

    graph.node_renderer.glyph = Circle(size=15, fill_color="lightblue")

    # to initialize the Line color with something
    line_color_list = graph.edge_renderer.data_source.data[LINE_COLOR_LIST]
    graph.edge_renderer.data_source.data[LINE_COLOR] =  [l[0] for l in line_color_list]
    
    line_width_list = graph.edge_renderer.data_source.data[LINE_WIDTH_LIST]
    graph.edge_renderer.data_source.data[LINE_WIDTH] =  [l[0] for l in line_width_list]

    graph.edge_renderer.glyph = MultiLine(
        line_color=LINE_COLOR, # the field of the edges
        line_width=LINE_WIDTH,
        # line_alpha=LINE_WIDTH
    )

    plot = figure()

    callback = CustomJS(
        args=dict(source=graph.edge_renderer.data_source, layouts=positions_list), 
        code="""

            console.log(layouts)
            console.log(source.data)

            // make a shallow copy of the current data dict
            const new_data = Object.assign({}, source.data)

            // update the y column in the new data dict from the appropriate other column
            const LC = source.data['line_color_list']
            new_data.line_color  = LC.map(subArray => subArray[cb_obj.value]);

            const LW = source.data['line_width_list']
            new_data.line_width  = LW.map(subArray => subArray[cb_obj.value]);
            console.log(new_data)

            // set the new data on source, BokehJS will pick this up automatically
            source.data = new_data
        """)
      
    slider = Slider(
        start=0, 
        end=len(graph.edge_renderer.data_source.data[LINE_COLOR_LIST][0]) - 1, 
        value=0, 
        step=1, 
        title="iteration"
    )
    
    slider.js_on_change('value', callback)

    # oranize the layout of the plotting-window
    plot.renderers.append(graph)
    layout = column(slider, plot)
    show(layout)

def plot(path):
    G = make_the_graph(path)
    draw_interactive_mlp_graph(G)
