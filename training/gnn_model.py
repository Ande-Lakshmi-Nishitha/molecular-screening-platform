"""The GNN used by the DL tier (MSP-25): message passing in plain PyTorch.

Owner: Nishitha. Requirements: FR-016, FR-017, FR-018.

Each layer is a GINE-style update (sum of neighbour messages that include the bond
features, then a small MLP, layer norm and a residual connection). The graph is pooled
by mean and sum and a small head predicts one number. The target's mean and standard
deviation are stored inside the model, so the exported file answers in the target's own
unit (pIC50 or pKi) and the serving code does not need to know them.

No torch_geometric: it is hard to install on Windows and not needed for graphs this
small. The module is TorchScript-compatible, so the exported file holds the whole
model and loads without this source file (SEC-008: no pickle of Python objects).
Training only. The worker (MSP-26) loads the exported TorchScript files.
"""
from typing import List

import torch
from torch import nn

from backend.workers.dl import graph

HIDDEN = 128
LAYERS = 3
DROPOUT = 0.1


class MessageLayer(nn.Module):
    """One message-passing step with bond features and a residual connection."""

    def __init__(self, hidden: int, n_bond: int):
        super().__init__()
        self.bond = nn.Linear(n_bond, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Update every atom from its neighbours."""
        messages = torch.relu(h[edge_index[0]] + self.bond(edge_attr))
        summed = torch.zeros_like(h).index_add(0, edge_index[1], messages)
        update = self.mlp((1.0 + self.eps) * h + summed)
        return h + torch.relu(self.norm(update))


class GnnRegressor(nn.Module):
    """Graph in, one activity value out (in the target's unit)."""

    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS, dropout: float = DROPOUT,
                 y_mean: float = 0.0, y_std: float = 1.0):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        self.embed = nn.Linear(graph.N_ATOM_FEATURES, hidden)
        self.layers = nn.ModuleList([MessageLayer(hidden, graph.N_BOND_FEATURES) for _ in range(layers)])
        self.norm = nn.LayerNorm(2 * hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std", torch.tensor(float(y_std)))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """One predicted value per molecule in the batch, shape (n_molecules,)."""
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        n_molecules = int(batch.max().item()) + 1
        summed = torch.zeros(n_molecules, h.size(1), dtype=h.dtype).index_add(0, batch, h)
        counts = torch.zeros(n_molecules, dtype=h.dtype).index_add(0, batch, torch.ones(h.size(0), dtype=h.dtype))
        pooled = torch.cat([summed / counts.clamp(min=1.0).unsqueeze(1), summed], dim=1)
        out = self.head(self.norm(pooled)).squeeze(1)
        return out * self.y_std + self.y_mean


def to_tensors(graphs: List[graph.MolGraph]):
    """Batch graphs (shared numpy function) and convert to the tensors the model takes."""
    x, edge_index, edge_attr, batch = graph.batch_graphs(graphs)
    return (
        torch.from_numpy(x),
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_attr),
        torch.from_numpy(batch),
    )
