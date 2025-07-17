"""
Utilities to measure oversquashing via Jacobian norms.

Given:
    f(x)      : node‑level embeddings after layer ℓ, shape [N, d]
    x.requires_grad = True
we compute   J = ∂f / ∂x   (shape [N, d, N, d_in])
and return   ‖J‖_F averaged over nodes.

For memory we:
  * sample 'num_nodes' random nodes per graph
  * compute jacobians with functorch.jacrev + vmap
"""

from __future__ import annotations
from typing import Callable, List, Dict

import torch
from torch import Tensor
from torch.func import jacrev, vmap   # PyTorch 2.0+ alias for functorch

# ---------- core helper ---------------------------------------------------- #

@torch.no_grad()
def layer_jacobian_frobenius(
    layer_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    num_samples: int | None = None,
) -> float:
    """
    Parameters
    ----------
    layer_fn : Callable that maps 'x' -> h (shape [N, d])
               MUST close over the network & graph structure.
    x        : input node features [N, d_in] with requires_grad=True
    num_samples : if given, randomly subsample that many nodes to
                  estimate ‖J‖_F.  Defaults to all nodes.

    Returns
    -------
    float  average Frobenius norm over selected nodes
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)

    N = x.size(0)
    idx = torch.randperm(N)[: (num_samples or N)]

    # functorch magic: jacrev w.r.t *inputs* and vmap over node index
    def node_embed(i: torch.Tensor, _x: Tensor) -> Tensor:      # i shape [], dtype long
    # turn i into a 1‑D tensor of indices and gather along dim 0
        return layer_fn(_x).index_select(0, i.unsqueeze(0)).squeeze(0)

    jac = vmap(jacrev(node_embed, argnums=1), in_dims=(0, None))(
        idx, x
    )  # shape [|idx|, d, N, d_in]

    frob = torch.linalg.norm(jac.reshape(len(idx), -1), dim=1)  # per‑node
    return float(frob.mean().item())


# ---------- convenience wrapper for a whole model -------------------------- #

def model_layer_norms(
    model: torch.nn.Module,
    data,
    layer_outputs: List[Tensor],
    sample_nodes: int | None = None,
) -> Dict[str, float]:
    """
    Compute ‖∂hᶫ/∂x‖_F for every hidden layer ℓ in `layer_outputs`.
    Assumes:
        * model has already run forward on `data`
        * layer_outputs is the List[Tensor] with shape [N, d] for each layer

    Returns mapping {layer_idx: norm}
    """
    norms = {}
    x0 = data.x.detach().requires_grad_(True)  # original features

    for li, h in enumerate(layer_outputs):
        fn = lambda _x: model.forward_until(data, stop_layer=li, x_override=_x)
        norms[f"layer_{li}"] = layer_jacobian_frobenius(
            fn, x0, num_samples=sample_nodes
        )

    return norms
