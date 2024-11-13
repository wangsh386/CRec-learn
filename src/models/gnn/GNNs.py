from torch import nn
from torch_geometric.nn import (GlobalAttention, Set2Set, global_add_pool,
                                global_max_pool, global_mean_pool)

from src.models.gnn.GNNConv import GNN_node, GNN_node_Virtualnode


class GNN(nn.Module):
    def __init__(
        self,
        num_layers=4,
        embedding_dim=64,
        gnn_type="gin",
        virtual_node=False,
        residual=False,
        dropout=0.5,
        JK="last",
        graph_pooling="mean",
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """

        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.JK = JK
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layers,
                embedding_dim,
                JK=JK,
                drop_ratio=dropout,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNN_node(
                num_layers,
                embedding_dim,
                JK=JK,
                drop_ratio=dropout,
                residual=residual,
                gnn_type=gnn_type,
            )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(embedding_dim, 2 * embedding_dim),
                    nn.BatchNorm1d(2 * embedding_dim),
                    nn.ReLU(),
                    nn.Linear(2 * embedding_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(embedding_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data["batch"])

        return h_node, h_graph
