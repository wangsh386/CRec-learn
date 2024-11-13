import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pyhealth.datasets import SampleEHRDataset
from torch import nn
from torch_geometric.nn import global_mean_pool

from src.models.crec_base import CRecBase, Predictor

logger = logging.getLogger(__name__)


class Separator(L.LightningModule):
    def __init__(self, args, molecule_graphs, hidden_size) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.molecule_graphs = molecule_graphs
        num_nodes = molecule_graphs.num_nodes
        self.node_weight_layer = nn.Sequential(
            nn.Linear(hidden_size, num_nodes),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 2),
        )

    def forward(self, node_repr, query):
        node_weight = torch.diag_embed(
            self.node_weight_layer(query)
        )  # [b, n_nodes] ->[b, n_nodes, n_nodes]
        node_repr = torch.matmul(node_weight, node_repr)  # [bs, n_nodes, dim]

        n_nodes = node_repr.size(1)
        query = query.unsqueeze(1).expand(-1, n_nodes, -1)
        h = torch.cat([node_repr, query], -1)

        p = self.mlp(h)  # [bs, n_nodes, 2]
        lambda_i = F.gumbel_softmax(p, tau=1, dim=-1)
        lambda_pos = lambda_i[:, :, 0].unsqueeze(dim=-1)
        lambda_neg = lambda_i[:, :, 1].unsqueeze(dim=-1)
        ########
        graph_repr_c, graph_repr_s, graph_repr_org = self.separate(
            node_repr, lambda_pos, lambda_neg
        )

        return graph_repr_c, graph_repr_s, graph_repr_org, lambda_pos

    def separate(self, node_repr, pos_score, neg_score):
        # [bs*n_nodes, dim]
        node_repr_c = (pos_score * node_repr).view(-1, self.hidden_size)
        node_repr_s = (neg_score * node_repr).view(-1, self.hidden_size)

        batch_size = pos_score.size(0)
        self.batch_batch = self.repeat_mol_attr(batch_size)

        _graph_repr_org = global_mean_pool(node_repr, self.molecule_graphs.batch)
        _graph_repr_c = global_mean_pool(node_repr_c, self.batch_batch).view(
            batch_size, -1, self.hidden_size
        )
        _graph_repr_s = global_mean_pool(node_repr_s, self.batch_batch).view(
            batch_size, -1, self.hidden_size
        )

        return _graph_repr_c, _graph_repr_s, _graph_repr_org

    def repeat_mol_attr(self, batch_size):
        batch_batch = torch.zeros(
            [batch_size, self.molecule_graphs["num_nodes"]],
            dtype=torch.long,
            device=self.device,
        )
        num_graphs = self.molecule_graphs["batch"].max() + 1
        for idx in range(batch_size):
            batch_batch[idx] = self.molecule_graphs["batch"] + idx * num_graphs
        batch_batch = batch_batch.contiguous().view(-1)
        return batch_batch


class Intervene(L.LightningModule):
    def __init__(self, args, hidden_size) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.interv_proj = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, graph_repr_c, graph_repr_s):
        batch_size, n_graphs = graph_repr_c.size()[:2]
        if self.training:
            rand_idx = torch.randperm(batch_size)
            graph_repr_s = graph_repr_s[rand_idx]

        if self.args.interv_style == "interpolation":
            lambda_ = self.args.epsilon * torch.rand(
                [batch_size, n_graphs, self.hidden_size], device=self.device
            )
            graph_repr_intervend = (1 - lambda_) * graph_repr_c + lambda_ * graph_repr_s
        elif self.args.interv_style == "concat":
            graph_repr_intervend = self.interv_proj(
                torch.cat([graph_repr_c, graph_repr_s], -1)
            )
        elif self.args.interv_style == "add":
            graph_repr_intervend = graph_repr_c + graph_repr_s
        else:
            raise ValueError
        return graph_repr_intervend


class CRec(CRecBase):
    def __init__(self, args, dataset: SampleEHRDataset, hidden_size: int = 64):
        super().__init__(args=args, dataset=dataset, hidden_size=hidden_size)
        self.args = args
        self.best_monitor_metric = 0.0
        self.hidden_size = hidden_size

        self.separator = Separator(
            args, self.feature_extractor.molecule_graphs, hidden_size
        )
        self.intervene = Intervene(args, hidden_size)

        self.predictor_pos = Predictor(hidden_size, self.feature_extractor.label_size)
        self.predictor_neg = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 2),
        )

        self.average_projection = self.feature_extractor.average_projection

        self.save_hyperparameters()

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        node_repr, query = self.feature_extractor(conditions, procedures)

        _graph_repr_c, _graph_repr_s, _graph_repr_org, pos_score = self.separator(
            node_repr, query
        )

        average_projection = self.average_projection.to(self.device)
        graph_repr_org = torch.matmul(average_projection, _graph_repr_org)  # 181-->193
        graph_repr_c = torch.matmul(average_projection, _graph_repr_c)
        graph_repr_s = torch.matmul(average_projection, _graph_repr_s)

        interved_repr = self.intervene(graph_repr_c, graph_repr_s)

        logits = self.predictor(interved_repr, query)
        logits_pos = self.predictor_pos(graph_repr_c, query)
        logits_neg = self.predictor_neg(graph_repr_s)
        return {
            "score": pos_score,
            "logits": logits,
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "graph_repr_c": graph_repr_c,
            "graph_repr_s": graph_repr_s,
            "graph_repr_org": graph_repr_org,
            "interved_repr": interved_repr,
        }

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        bce_labels, multi_labels = self.prepare_labels(drugs)

        output = self.forward(conditions, procedures)

        loss = 0
        loss_task = self.calc_loss(output["logits"], bce_labels, multi_labels)
        loss += loss_task

        loss_pos = self.calc_loss(output["logits_pos"], bce_labels, multi_labels)
        loss += self.args.w_pos * loss_pos

        random_label = (
            torch.ones_like(
                output["logits_neg"],
                dtype=torch.float,
                device=self.device,
            )
            / 2
        )
        loss_neg = F.kl_div(
            F.log_softmax(output["logits_neg"], dim=-1),
            random_label,
            reduction="batchmean",
        )
        loss += self.args.w_neg * loss_neg

        loss_reg = torch.mean(output["score"])  # L1 norm
        loss += self.args.w_reg * loss_reg

        self.log("train/loss", loss)
        self.log("train/loss_task", loss_task)
        self.log("train/loss_pos", loss_pos)
        self.log("train/loss_neg", loss_neg)
        self.log("train/loss_reg", loss_reg)
        return loss
