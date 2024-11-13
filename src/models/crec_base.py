import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from lightning.pytorch.utilities.types import STEP_OUTPUT
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils import smiles2graph
from pyhealth.datasets import SampleEHRDataset
from pyhealth.medcode import ATC
from pyhealth.models.utils import batch_to_multihot, get_last_visit
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool

from src.models.base import BaseModel
from src.utils import multi_label_metric

logger = logging.getLogger(__name__)


def graph_batch_from_smiles(smiles_list, device=torch.device("cpu")):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]

    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph["edge_index"] + lstnode)
        edge_feats.append(graph["edge_feat"])
        node_feats.append(graph["node_feat"])
        lstnode += graph["num_nodes"]
        batch.append(np.ones(graph["num_nodes"], dtype=np.int64) * idx)

    result = {
        "edge_index": np.concatenate(edge_idxes, axis=-1),
        "edge_attr": np.concatenate(edge_feats, axis=0),
        "batch": np.concatenate(batch, axis=0),
        "x": np.concatenate(node_feats, axis=0),
    }
    result = {k: torch.from_numpy(v).to(device) for k, v in result.items()}
    result["num_nodes"] = lstnode
    result["num_edges"] = result["edge_index"].shape[1]
    return Data(**result)


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr="add")

        self.mlp = nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINGraph(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, embedding_dim: int = 64, dropout: float = 0.7
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim=embedding_dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout_fun = torch.nn.Dropout(dropout)
        for layer in range(self.num_layers):
            self.convs.append(GINConv(embedding_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(embedding_dim))

    def forward(self, graph: Dict[str, Union[int, torch.Tensor]]) -> torch.Tensor:
        h_list = [self.atom_encoder(graph["x"])]
        for layer in range(self.num_layers):
            h = self.batch_norms[layer](
                self.convs[layer](
                    h_list[layer], graph["edge_index"], graph["edge_attr"]
                )
            )
            if layer != self.num_layers - 1:
                h = self.dropout_fun(torch.relu(h))
            else:
                h = self.dropout_fun(h)
            h_list.append(h)
        node_repr = h_list[-1]
        # mean pooling
        batch_size, dim = graph["batch"].max().item() + 1, h_list[-1].shape[-1]
        out_feat = torch.zeros(batch_size, dim).to(h_list[-1])
        cnt = torch.zeros_like(out_feat).to(out_feat)
        index = graph["batch"].unsqueeze(-1).repeat(1, dim)

        out_feat.scatter_add_(dim=0, index=index, src=h_list[-1])
        cnt.scatter_add_(
            dim=0, index=index, src=torch.ones_like(h_list[-1]).to(h_list[-1])
        )
        graph_repr = out_feat / (cnt + 1e-9)
        return node_repr, graph_repr


class MAB(torch.nn.Module):
    def __init__(
        self, Qdim: int, Kdim: int, Vdim: int, number_heads: int, use_ln: bool = False
    ):
        super().__init__()
        self.Vdim = Vdim
        self.number_heads = number_heads

        assert (
            self.Vdim % self.number_heads == 0
        ), "the dim of features should be divisible by number_heads"

        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)

        self.use_ln = use_ln
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)
            self.ln2 = torch.nn.LayerNorm(self.Vdim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads

        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)
        O = Q_split + torch.matmul(Attn, V_split)
        O = torch.cat(O.split(batch_size, 0), 2)

        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)
        O = O if not self.use_ln else self.ln2(O)

        return O


class SAB(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, number_heads: int, use_ln: bool = False
    ):
        super().__init__()
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X, X)


class FeatureExtractor(BaseModel):
    def __init__(
        self,
        args,
        dataset: SampleEHRDataset,
        hidden_size: int = 64,
        num_rnn_layers: int = 1,
        num_gnn_layers: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.args = args
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.dropout_fn = torch.nn.Dropout(dropout)

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer(special_tokens=["<unk>"])
        self.embeddings = self.get_embedding_layers(
            self.feat_tokenizers, self.hidden_size
        )
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        self.all_smiles_list = self.generate_smiles_list()

        average_projection, self.all_smiles_flatten = self.generate_average_projection()
        self.average_projection = torch.nn.Parameter(
            average_projection, requires_grad=False
        )
        self.molecule_graphs = graph_batch_from_smiles(self.all_smiles_flatten)

        self.ddi_adj = self.generate_ddi_adj()

        self.rnns = torch.nn.ModuleDict(
            {
                x: torch.nn.GRU(
                    hidden_size,
                    hidden_size,
                    num_layers=num_rnn_layers,
                    dropout=dropout if num_rnn_layers > 1 else 0,
                    batch_first=True,
                )
                for x in ["conditions", "procedures"]
            }
        )
        self.query_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.molecule_encoder = GINGraph(num_gnn_layers, hidden_size, dropout)

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs,
    ):
        query = self.get_patient_query(conditions, procedures)
        node_repr, _ = self.molecule_encoder(self.molecule_graphs.to(self.device))
        return node_repr, query

    def encode_patient(
        self, feature_key: str, raw_values: List[List[List[str]]]
    ) -> torch.Tensor:
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(
            raw_values, truncation=(False, False)
        )
        tensor_codes = torch.tensor(
            codes, dtype=torch.long, device=self.device
        )  # [bs, v_len, code_len]

        embeddings = self.embeddings[feature_key](
            tensor_codes
        )  # [bs, v_len, code_len, dim]
        embeddings = torch.sum(self.dropout_fn(embeddings), dim=2)  # [bs, v_len, dim]

        mask = torch.sum(embeddings, dim=2) != 0
        lengths = torch.sum(mask.int(), dim=-1).cpu()
        embeddings_packed = rnn_utils.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        outputs_packed, _ = self.rnns[feature_key](embeddings_packed)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs_packed, batch_first=True)
        return outputs, mask

    def get_patient_query(
        self, conditions: List[List[List[str]]], procedures: List[List[List[str]]]
    ):
        # encoding procs and diags
        condition_emb, mask = self.encode_patient("conditions", conditions)
        procedure_emb, mask = self.encode_patient("procedures", procedures)

        patient_emb = torch.cat([condition_emb, procedure_emb], dim=-1)
        queries = self.query_layer(patient_emb)
        query = get_last_visit(queries, mask)
        return query

    def generate_smiles_list(self) -> List[List[str]]:
        """Generates the list of SMILES strings."""
        atc3_to_smiles = {}
        atc = ATC()
        for code in atc.graph.nodes:
            if len(code) != 7:
                continue
            code_atc3 = ATC.convert(code, level=3)
            smiles = atc.graph.nodes[code]["smiles"]
            if smiles != smiles:
                continue
            atc3_to_smiles[code_atc3] = atc3_to_smiles.get(code_atc3, []) + [smiles]
        # just take first one for computational efficiency
        atc3_to_smiles = {
            k: v[: self.args.topk_smiles] for k, v in atc3_to_smiles.items()
        }
        all_smiles_list = [[] for _ in range(self.label_size)]
        vocab_to_index = self.label_tokenizer.vocabulary
        for atc3, smiles_list in atc3_to_smiles.items():
            if atc3 in vocab_to_index:
                index = vocab_to_index(atc3)
                all_smiles_list[index] += smiles_list
        return all_smiles_list

    def generate_average_projection(self) -> Tuple[torch.Tensor, List[str]]:
        molecule_set, average_index = [], []
        for smiles_list in self.all_smiles_list:
            """Create each data with the above defined functions."""
            counter = 0  # counter how many drugs are under that ATC-3
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                molecule_set.append(smiles)
                counter += 1
            average_index.append(counter)
        average_projection = np.zeros((len(average_index), sum(average_index)))
        col_counter = 0
        for i, item in enumerate(average_index):
            if item <= 0:
                continue
            average_projection[i, col_counter : col_counter + item] = 1 / item
            col_counter += item
        average_projection = torch.FloatTensor(average_projection)
        return average_projection, molecule_set

    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj


class FusionModule(nn.Module):
    def __init__(self, hidden_size, label_size) -> None:
        super().__init__()
        self.query_graph_rela = nn.Sequential(
            nn.Linear(hidden_size, label_size), nn.Sigmoid()
        )

    def forward(self, graph_repr, query):
        graph_weight = torch.diag_embed(self.query_graph_rela(query))
        graph_repr = torch.matmul(graph_weight, graph_repr)

        n_graphs = graph_repr.size(1)
        h = torch.cat([graph_repr, query.unsqueeze(1).expand(-1, n_graphs, -1)], -1)
        return h


class Predictor(nn.Module):
    def __init__(self, hidden_size, label_size) -> None:
        super().__init__()
        self.fusion_module = FusionModule(hidden_size, label_size)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, graph_repr, query):
        h = self.fusion_module(graph_repr, query)
        logits = self.predictor(h).squeeze(-1)
        return logits


class CRecBase(BaseModel):
    def __init__(self, args, dataset: SampleEHRDataset, hidden_size: int = 64):
        super().__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.args = args
        self.best_monitor_metric = 0.0
        self.hidden_size = hidden_size

        self.feature_extractor = FeatureExtractor(args, dataset)

        self.predictor = Predictor(hidden_size, self.feature_extractor.label_size)

        self.average_projection = self.feature_extractor.average_projection

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        node_repr, query = self.feature_extractor(conditions, procedures)
        _graph_repr_org = global_mean_pool(
            node_repr, self.feature_extractor.molecule_graphs.batch
        )
        average_projection = self.average_projection.to(self.device)
        graph_repr_org = torch.matmul(average_projection, _graph_repr_org)

        logits = self.predictor(graph_repr_org, query)
        return {"logits": logits}

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        bce_labels, multi_labels = self.prepare_labels(drugs)

        output = self.forward(conditions, procedures)

        loss = self.calc_loss(output["logits"], bce_labels, multi_labels)

        self.log("train/loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.y_true_all = []
        self.y_prob_all = []

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        # prepare labels
        labels_index = self.feature_extractor.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # convert to multihot
        labels = batch_to_multihot(labels_index, self.feature_extractor.label_size)

        output = self.forward(conditions, procedures)
        y_prob = torch.sigmoid(output["logits"])

        y_true = labels.cpu().numpy()
        y_prob = y_prob.cpu().numpy()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_validation_epoch_end(self) -> None:
        y_true_all = np.concatenate(self.y_true_all, axis=0)
        y_prob_all = np.concatenate(self.y_prob_all, axis=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        scores = multi_label_metric(
            y_prob_all, y_true_all, self.feature_extractor.ddi_adj
        )

        for key in scores.keys():
            self.log(f"val/{key}", scores[key])

        monitor_metric = scores["ja"]
        if monitor_metric > self.best_monitor_metric:
            self.best_monitor_metric = monitor_metric
            logger.info(
                f"New best ja: {self.best_monitor_metric:.4f} in epoch {self.trainer.current_epoch}"
            )

        self.log("val/best_ja", self.best_monitor_metric)

    def on_test_epoch_start(self) -> None:
        self.y_true_all = []
        self.y_prob_all = []

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        # prepare labels
        labels_index = self.feature_extractor.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # convert to multihot
        labels = batch_to_multihot(labels_index, self.feature_extractor.label_size)

        output = self.forward(conditions, procedures)
        y_prob = torch.sigmoid(output["logits"])

        y_true = labels.cpu().numpy()
        y_prob = y_prob.cpu().numpy()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_test_epoch_end(self) -> Any:
        y_true_all = np.concatenate(self.y_true_all, axis=0)
        y_prob_all = np.concatenate(self.y_prob_all, axis=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        scores = multi_label_metric(
            y_prob_all, y_true_all, self.feature_extractor.ddi_adj
        )

        for key in scores.keys():
            logger.info(f"test/{key}: {scores[key]}")
            self.log(f"test/{key}", scores[key])
        return scores

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-5)
        return [opt]

    def prepare_labels(self, drugs):
        # prepare labels
        labels_index = self.feature_extractor.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # convert to multihot
        labels = batch_to_multihot(labels_index, self.feature_extractor.label_size)

        multi_labels = -np.ones(
            (len(labels), self.feature_extractor.label_size), dtype=np.int64
        )
        for idx, cont in enumerate(labels_index):
            # remove redundant labels
            cont = list(set(cont))
            multi_labels[idx, : len(cont)] = cont
        multi_labels = torch.from_numpy(multi_labels)
        return labels.to(self.device), multi_labels.to(self.device)

    def calc_loss(self, logits, bce_labels, multi_labels):
        loss_bce = F.binary_cross_entropy_with_logits(logits, bce_labels)
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), multi_labels)
        loss_task = 0.95 * loss_bce + 0.05 * loss_multi
        return loss_task
