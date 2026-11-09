"""GraphSAGE multi-taches (12 sorties) pour Tox21.

Meme interface que GAT / GATTrainer (GNN.py) :
    SAGETrainer(data).pipeline(seed_list=range(3))

Ce qui change par rapport au GAT : la couche de message passing.
SAGEConv de PyG ignore les features de liaison ; on utilise donc une couche
"SAGE-E" maison : message = ReLU(x_j + e_ij) (facon GINE), agregation
mean/max/add, puis  h_i' = W_l * agg_i + W_r * h_i  (la signature GraphSAGE :
voisinage et noeud central ont des poids separes).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from GNN import ATOM_DIMS, BOND_DIMS, N_TASKS, CatEncoder, GATTrainer


# ====================================================================== couche
class SAGEEConv(MessagePassing):
    """GraphSAGE avec features de liaison.
    h_i' = W_l * AGG_j( ReLU(h_j + e_ij) ) + W_r * h_i"""

    def __init__(self, hidden: int, aggr: str = "mean", use_edge_attr: bool = True):
        super().__init__(aggr=aggr)
        self.use_edge_attr = use_edge_attr
        self.lin_l = torch.nn.Linear(hidden, hidden)              # voisinage
        self.lin_r = torch.nn.Linear(hidden, hidden, bias=False)  # noeud central

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin_l(agg) + self.lin_r(x)

    def message(self, x_j, edge_attr):
        if self.use_edge_attr:
            return F.relu(x_j + edge_attr)
        return x_j


# ====================================================================== modele
class GraphSAGE(torch.nn.Module):
    """GraphSAGE multi-taches. Meme forward / memes sorties que GAT :
    forward(x, edge_index, edge_attr, batch) -> (batch_size, 12) logits.

    - CatEncoder atomes / liaisons,
    - n_layers SAGEEConv : residu + LayerNorm + ReLU + dropout,
    - pooling mean || add || max, tete MLP partagee.
    """

    def __init__(self, hidden: int = 256, n_layers: int = 3, n_tasks: int = N_TASKS,
                 dropout: float = 0.2, aggr: str = "mean", use_edge_attr: bool = True):
        super().__init__()
        self.dropout = dropout
        self.atom_enc = CatEncoder(ATOM_DIMS, hidden)
        self.bond_enc = CatEncoder(BOND_DIMS, hidden)

        self.convs = torch.nn.ModuleList(
            SAGEEConv(hidden, aggr=aggr, use_edge_attr=use_edge_attr)
            for _ in range(n_layers)
        )
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(hidden) for _ in range(n_layers))

        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, n_tasks),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_enc(x)
        e = self.bond_enc(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h, edge_index, e)   # residu
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        mol = torch.cat(
            [global_mean_pool(h, batch), global_add_pool(h, batch), global_max_pool(h, batch)],
            dim=1,
        )
        return self.head(mol)


# ==================================================================== entraineur
class SAGETrainer(GATTrainer):
    """Herite de tout GATTrainer (splits, loss masquee, early stopping, plots,
    sauvegarde). Seuls le modele et le nom des fichiers changent."""

    MODEL_NAME = "GraphSAGE"

    def __init__(self, data, params=None):
        defaults = {"aggr": "mean", "use_edge_attr": True}  # 'max' et 'add' possibles
        super().__init__(data, {**defaults, **(params or {})})

    def build_model(self):
        p = self.params
        return GraphSAGE(
            hidden=p["hidden"], n_layers=p["n_layers"], dropout=p["dropout"],
            aggr=p["aggr"], use_edge_attr=p["use_edge_attr"],
        )