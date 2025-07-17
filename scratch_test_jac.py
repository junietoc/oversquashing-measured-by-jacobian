import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from jacobian import layer_jacobian_frobenius

N = 10
x = torch.randn(N, 5, requires_grad=True)
edge_index = torch.randint(0, N, (2, 40))
layer = GCNConv(5, 7)

def f(inp):           # closure capturing layer & edge_index
    return layer(inp, edge_index)

print("‖J‖_F =", layer_jacobian_frobenius(f, x, num_samples=5))
