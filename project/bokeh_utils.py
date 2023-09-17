from networkx import multipartite_layout

from bokeh import events
from bokeh.io import output_file, output_notebook, curdoc
from bokeh.plotting import figure, show, from_networkx
from bokeh.models import ColumnDataSource, Circle, MultiLine
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.colors import Color

LINE_COLOR = 'line_color'
LINE_COLOR_LIST = 'line_color_list'
LINE_WIDTH_LIST = 'line_width_list'
LINE_WIDTH = 'line_width'
LINE_ALPHA = 'line_alpha'
LAYER='layer'

def draw_interactive_mlp_graph(G):
    graph = from_networkx(G, layout_function=multipartite_layout(G, subset_key=LAYER))


    graph.node_renderer.glyph = Circle(size=15, fill_color="lightblue")

    # to initialize the Line color with something
    line_color_list = graph.edge_renderer.data_source.data[LINE_COLOR_LIST]
    graph.edge_renderer.data_source.data[LINE_COLOR] =  [l[0] for l in line_color_list]
    line_width_list = graph.edge_renderer.data_source.data[LINE_WIDTH_LIST]
    graph.edge_renderer.data_source.data[LINE_WIDTH] =  [l[0] for l in line_width_list]

    graph.edge_renderer.glyph = MultiLine(
        line_color=LINE_COLOR, # the field of the edges
        line_width=LINE_WIDTH,
        line_alpha=LINE_WIDTH
    )
    plot = figure()

    callback = CustomJS(
        args=dict(source=graph.edge_renderer.data_source), 
        code="""
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
    slider.update
    
    slider.js_on_change('value', callback)

    plot.renderers.append(graph)
    layout = column(slider, plot)
    show(layout)
