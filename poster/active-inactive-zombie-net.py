
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from common import nxutils
from common import models
from common import constants

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
    nxutils.__draw_nx(nxutils.subgraph_by_state(G, exclude=[constants.ParamState.pruned]), G)
    plt.savefig(f"{__file__}.svg")

if __name__ == '__main__':
    main()