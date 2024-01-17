import networkx as nx
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from utils import nxutils
from training import models


def __draw_nx(G, layout_G=None):
    if layout_G is None:
        layout_G = G

    pos = nx.multipartite_layout(layout_G, 'layer')
    edge_colors = [nxutils.__colormap(data.get('state')) for *_, data in G.edges(data=True)]
    node_colors = [nxutils.__colormap(data.get('state')) for _, data in G.nodes(data=True)]

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=node_colors)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        width=6,
        alpha=0.7,
        edge_color=edge_colors,
    )


def main():
    model = models.MultiTaskBinaryMLP(((4,6,6,2)), seed=2)
    G = nxutils.build_nx(model)

    params = (
        [(module, 'weight') for module in model.layers]
    # + [(module, 'bias') for module in model.layers]
    )
    prune.global_unstructured(params, prune.Identity)
    G = nxutils.build_nx(model)

    out_features = nxutils.out_features(G, len(model.layers))
    in_features  = nxutils.in_features(G)

    # prune
    prune.global_unstructured(
        params, prune.L1Unstructured, amount=.6
    )
    nxutils.tag_params(G, model, in_features, out_features)
    nxutils.__draw_nx(nxutils.subgraph_by_state(G, exclude=[nxutils.ParamState.pruned]), G)
    plt.savefig(f"{__file__}.svg")

if __name__ == '__main__':
    main()