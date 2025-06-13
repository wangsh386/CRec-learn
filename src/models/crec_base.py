import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from lightning.pytorch.utilities.types import STEP_OUTPUT
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder  # OGB 中的原子和键编码器
from ogb.utils import smiles2graph  # 将 SMILES 字符串转换为图结构的工具函数
from pyhealth.datasets import SampleEHRDataset  # PyHealth 提供的 EHR 数据集类
from pyhealth.medcode import ATC  # ATC 药物分类编码工具
from pyhealth.models.utils import batch_to_multihot, get_last_visit  # 工具函数：将标签转为多热编码、获取最后一次访问
from rdkit import Chem  # RDKit 化学信息学工具包
from torch import nn
from torch_geometric.data import Data  # PyG 中的数据结构
from torch_geometric.nn import MessagePassing, global_mean_pool  # PyG 中的消息传递机制和全局池化

from src.models.base import BaseModel  # 自定义的模型基类
from src.utils import multi_label_metric  # 多标签评估指标计算工具

# 创建一个 logger，用于记录日志信息
logger = logging.getLogger(__name__)


# 定义一个函数，将 SMILES 字符串列表转换为图数据对象 (PyG 的 Data 对象)
def graph_batch_from_smiles(smiles_list, device=torch.device("cpu")):
    """
    将 SMILES 字符串列表转换为 PyG 的 Data 对象，包含边索引、边特征、节点特征和 batch 信息。

    Args:
        smiles_list: SMILES 字符串列表
        device: 数据存放的设备，默认为 CPU

    Returns:
        PyG 的 Data 对象，包含图结构信息
    """
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    # 遍历 SMILES 列表，将每个 SMILES 转换为图结构
    graphs = [smiles2graph(x) for x in smiles_list]

    for idx, graph in enumerate(graphs):
        # 将边索引偏移，确保不同图的边不重叠
        edge_idxes.append(graph["edge_index"] + lstnode)
        edge_feats.append(graph["edge_feat"])
        node_feats.append(graph["node_feat"])
        lstnode += graph["num_nodes"]  # 更新节点计数器
        batch.append(np.ones(graph["num_nodes"], dtype=np.int64) * idx)  # 记录每个节点属于哪个图

    # 将所有图的边索引、边特征、节点特征和 batch 信息拼接起来
    result = {
        "edge_index": np.concatenate(edge_idxes, axis=-1),
        "edge_attr": np.concatenate(edge_feats, axis=0),
        "batch": np.concatenate(batch, axis=0),
        "x": np.concatenate(node_feats, axis=0),
    }
    # 将 numpy 数组转换为 PyTorch 张量，并移动到指定设备
    result = {k: torch.from_numpy(v).to(device) for k, v in result.items()}
    result["num_nodes"] = lstnode  # 记录总节点数
    result["num_edges"] = result["edge_index"].shape[1]  # 记录总边数
    return Data(**result)  # 返回 PyG 的 Data 对象


# 定义 GIN 卷积层，继承自 MessagePassing，实现图神经网络的消息传递机制
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        """
        初始化 GIN 卷积层。

        Args:
            emb_dim: 嵌入维度，即节点特征的维度
        """
        super().__init__(aggr="add")  # 使用加法聚合方式

        # 定义一个多层感知机 (MLP)，用于更新节点特征
        self.mlp = nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),  # 输入 emb_dim，输出 2*emb_dim
            torch.nn.BatchNorm1d(2 * emb_dim),  # 批归一化
            torch.nn.ReLU(),  # 激活函数 ReLU
            torch.nn.Linear(2 * emb_dim, emb_dim),  # 输出 emb_dim
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))  # 可学习的参数，用于控制自环的权重
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)  # 键编码器，将边特征编码为 emb_dim 维度

    def forward(self, x, edge_index, edge_attr):
        """
        前向传播函数，计算节点更新后的特征。

        Args:
            x: 节点特征，形状为 [num_nodes, emb_dim]
            edge_index: 边索引，形状为 [2, num_edges]
            edge_attr: 边特征，形状为 [num_edges, emb_dim]

        Returns:
            更新后的节点特征
        """
        edge_embedding = self.bond_encoder(edge_attr)  # 将边特征编码为 emb_dim 维度
        # 调用父类的 propagate 方法，执行消息传递
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        """
        消息传递函数，计算消息。

        Args:
            x_j: 邻居节点的特征
            edge_attr: 边特征

        Returns:
            消息，形状为 [num_edges, emb_dim]
        """
        return F.relu(x_j + edge_attr)  # 将邻居节点特征和边特征相加后通过 ReLU 激活

    def update(self, aggr_out):
        """
        更新函数，更新节点特征。

        Args:
            aggr_out: 聚合后的消息，形状为 [num_nodes, emb_dim]

        Returns:
            更新后的节点特征
        """
        return aggr_out


# 定义 GIN 图神经网络模型，包含多个 GIN 卷积层
class GINGraph(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, embedding_dim: int = 64, dropout: float = 0.7
    ):
        """
        初始化 GIN 图神经网络模型。

        Args:
            num_layers: GNN 层数，默认为 4
            embedding_dim: 嵌入维度，默认为 64
            dropout: Dropout 概率，默认为 0.7
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")  # GNN 层数必须大于 1

        # 初始化原子编码器，将节点特征编码为 embedding_dim 维度
        self.atom_encoder = AtomEncoder(emb_dim=embedding_dim)
        self.convs = torch.nn.ModuleList()  # 存储 GIN 卷积层的列表
        self.batch_norms = torch.nn.ModuleList()  # 存储批归一化层的列表
        self.num_layers = num_layers  # GNN 层数
        self.dropout_fun = torch.nn.Dropout(dropout)  # Dropout 层

        # 创建多个 GIN 卷积层和批归一化层
        for layer in range(self.num_layers):
            self.convs.append(GINConv(embedding_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(embedding_dim))

    def forward(self, graph: Dict[str, Union[int, torch.Tensor]]) -> torch.Tensor:
        """
        前向传播函数，计算图的节点表示和图表示。

        Args:
            graph: 图数据，包含节点特征、边索引、边特征和 batch 信息

        Returns:
            node_repr: 节点表示，形状为 [num_nodes, emb_dim]
            graph_repr: 图表示，形状为 [batch_size, emb_dim]
        """
        # 使用原子编码器对节点特征进行编码
        h_list = [self.atom_encoder(graph["x"])]
        for layer in range(self.num_layers):
            # 通过 GIN 卷积层和批归一化层更新节点特征
            h = self.batch_norms[layer](
                self.convs[layer](
                    h_list[layer], graph["edge_index"], graph["edge_attr"]
                )
            )
            if layer != self.num_layers - 1:
                h = self.dropout_fun(torch.relu(h))  # 如果不是最后一层，使用 ReLU 激活和 Dropout
            else:
                h = self.dropout_fun(h)  # 最后一层只使用 Dropout
            h_list.append(h)

        node_repr = h_list[-1]  # 获取最后一层的节点表示

        # 使用全局平均池化计算图表示
        batch_size, dim = graph["batch"].max().item() + 1, h_list[-1].shape[-1]
        out_feat = torch.zeros(batch_size, dim).to(h_list[-1])  # 初始化图表示张量
        cnt = torch.zeros_like(out_feat).to(out_feat)  # 初始化计数器张量
        index = graph["batch"].unsqueeze(-1).repeat(1, dim)  # 扩展 batch 索引

        # 使用 scatter_add_ 对节点特征进行聚合，计算图表示
        out_feat.scatter_add_(dim=0, index=index, src=h_list[-1])
        cnt.scatter_add_(
            dim=0, index=index, src=torch.ones_like(h_list[-1]).to(h_list[-1])
        )
        graph_repr = out_feat / (cnt + 1e-9)  # 避免除以零
        return node_repr, graph_repr


class MAB(torch.nn.Module):
    def __init__(
        self, Qdim: int, Kdim: int, Vdim: int, number_heads: int, use_ln: bool = False
    ):
        """
        初始化多头注意力块 (Multi-head Attention Block, MAB)。

        Args:
            Qdim: 查询 (Query) 的特征维度
            Kdim: 键 (Key) 的特征维度
            Vdim: 值 (Value) 的特征维度
            number_heads: 多头注意力的头数
            use_ln: 是否使用 Layer Normalization，默认为 False
        """
        super().__init__()
        self.Vdim = Vdim  # 值的维度
        self.number_heads = number_heads  # 注意力头数

        # 确保值维度能被头数整除
        assert (
            self.Vdim % self.number_heads == 0
        ), "the dim of features should be divisible by number_heads"

        # 定义四个线性层，分别用于生成 Q (Query)、K (Key)、V (Value) 和最终输出 O
        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)  # Q = Linear(Qdim -> Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)  # K = Linear(Kdim -> Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)  # V = Linear(Kdim -> Vdim)
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)  # O = Linear(Vdim -> Vdim)

        self.use_ln = use_ln  # 是否使用 Layer Normalization
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)  # 第一个 LayerNorm
            self.ln2 = torch.nn.LayerNorm(self.Vdim)  # 第二个 LayerNorm

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，计算多头注意力机制的输出。

        Args:
            X: 查询 (Query) 张量，形状为 [batch_size, seq_len, Qdim]
            Y: 键和值 (Key & Value) 张量，形状为 [batch_size, seq_len, Kdim]

        Returns:
            O: 多头注意力机制的输出张量，形状为 [batch_size, seq_len, Vdim]
        """
        # 通过线性层分别生成 Q、K、V
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads  # 计算每个头的维度

        # 将 Q、K、V 按头数拆分，并在 batch 维度上拼接，便于并行计算
        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        # 计算注意力分数矩阵
        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)  # 缩放点积注意力

        # 使用注意力分数对 V 进行加权求和
        O = Q_split + torch.matmul(Attn, V_split)  # 残差连接

        # 将多头注意力的结果重新拼接回原始 batch 维度
        O = torch.cat(O.split(batch_size, 0), 2)

        # 如果使用 LayerNorm，对结果进行归一化
        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)  # 通过线性层生成最终输出
        O = O if not self.use_ln else self.ln2(O)  # 再次归一化（如果启用）

        return O


class SAB(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, number_heads: int, use_ln: bool = False
    ):
        """
        初始化自注意力块 (Self-Attention Block, SAB)。

        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            number_heads: 多头注意力的头数
            use_ln: 是否使用 Layer Normalization，默认为 False
        """
        super().__init__()
        # SAB 本质上是 MAB 的一种特殊情况，其中 Q、K、V 来自同一个输入
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，计算自注意力机制的输出。

        Args:
            X: 输入张量，形状为 [batch_size, seq_len, in_dim]

        Returns:
            输出张量，形状为 [batch_size, seq_len, out_dim]
        """
        return self.net(X, X)  # Q 和 K、V 都来自 X


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
        """
        初始化特征提取器模块。

        Args:
            args: 配置参数对象
            dataset: EHR 数据集对象
            hidden_size: 隐藏层维度，默认为 64
            num_rnn_layers: RNN 层数，默认为 1
            num_gnn_layers: GNN 层数，默认为 4
            dropout: Dropout 概率，默认为 0.5
        """
        super().__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],  # 使用的条件和操作特征键
            label_key="drugs",  # 标签键
            mode="multilabel",  # 多标签分类任务
        )
        self.args = args
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.dropout_fn = torch.nn.Dropout(dropout)  # Dropout 层

        # 初始化特征分词器、标签分词器和嵌入层
        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer(special_tokens=["<unk>"])
        self.embeddings = self.get_embedding_layers(
            self.feat_tokenizers, self.hidden_size
        )
        self.label_size = self.label_tokenizer.get_vocabulary_size()  # 标签数量

        # 生成 SMILES 列表，用于构建分子图
        self.all_smiles_list = self.generate_smiles_list()

        # 生成平均投影矩阵，用于将分子图表示映射到目标维度
        average_projection, self.all_smiles_flatten = self.generate_average_projection()
        self.average_projection = torch.nn.Parameter(
            average_projection, requires_grad=False
        )  # 设置为不可训练参数

        # 使用 SMILES 列表构建分子图数据集
        self.molecule_graphs = graph_batch_from_smiles(self.all_smiles_flatten)

        # 生成 DDI (药物-药物相互作用) 邻接矩阵
        self.ddi_adj = self.generate_ddi_adj()

        # 初始化 RNN 模块，用于编码条件和操作序列
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
        # 定义查询层，用于将条件和操作序列的表示融合为查询向量
        self.query_layer = nn.Linear(2 * hidden_size, hidden_size)
        # 初始化分子图编码器 (基于 GIN 的图神经网络)
        self.molecule_encoder = GINGraph(num_gnn_layers, hidden_size, dropout)

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs,
    ):
        """
        前向传播函数，提取条件和操作序列的特征，并生成分子图的节点表示和图表示。

        Args:
            conditions: 条件序列，格式为 List[List[List[str]]]
            procedures: 操作序列，格式为 List[List[List[str]]]

        Returns:
            node_repr: 分子图的节点表示，形状为 [num_nodes, emb_dim]
            query: 查询向量，形状为 [batch_size, hidden_size]
        """
        # 获取患者查询向量 (融合条件和操作序列的表示)
        query = self.get_patient_query(conditions, procedures)
        # 使用分子图编码器提取分子图的节点表示
        node_repr, _ = self.molecule_encoder(self.molecule_graphs.to(self.device))
        return node_repr, query

    def encode_patient(
        self, feature_key: str, raw_values: List[List[List[str]]]
    ) -> torch.Tensor:
        """
        对患者的条件和操作序列进行编码，生成序列的嵌入表示。

        Args:
            feature_key: 特征键，如 "conditions" 或 "procedures"
            raw_values: 原始特征值，格式为 List[List[List[str]]]

        Returns:
            outputs: 编码后的序列表示，形状为 [batch_size, seq_len, emb_dim]
            mask: 序列掩码，形状为 [batch_size, seq_len]
        """
        # 使用分词器对原始特征值进行编码
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(
            raw_values, truncation=(False, False)
        )
        # 将编码后的 token 转换为张量
        tensor_codes = torch.tensor(
            codes, dtype=torch.long, device=self.device
        )  # [batch_size, seq_len, code_len]

        # 通过嵌入层将 token 转换为嵌入表示
        embeddings = self.embeddings[feature_key](
            tensor_codes
        )  # [batch_size, seq_len, code_len, emb_dim]
        # 对 code 维度进行求和，并应用 Dropout
        embeddings = torch.sum(self.dropout_fn(embeddings), dim=2)  # [batch_size, seq_len, emb_dim]

        # 计算序列掩码 (非零位置为有效 token)
        mask = torch.sum(embeddings, dim=2) != 0
        # 计算序列长度
        lengths = torch.sum(mask.int(), dim=-1).cpu()
        # 使用 pack_padded_sequence 对变长序列进行打包，以提高 RNN 计算效率
        embeddings_packed = rnn_utils.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        # 通过 RNN 对序列进行编码
        outputs_packed, _ = self.rnns[feature_key](embeddings_packed)
        # 将打包后的序列解包
        outputs, _ = rnn_utils.pad_packed_sequence(outputs_packed, batch_first=True)
        return outputs, mask

    def get_patient_query(
        self, conditions: List[List[List[str]]], procedures: List[List[List[str]]]
    ):
        """
        获取患者的查询向量，通过融合条件和操作序列的表示生成。

        Args:
            conditions: 条件序列
            procedures: 操作序列

        Returns:
            query: 查询向量，形状为 [batch_size, hidden_size]
        """
        # 对条件和操作序列进行编码
        condition_emb, mask = self.encode_patient("conditions", conditions)
        procedure_emb, mask = self.encode_patient("procedures", procedures)

        # 将条件和操作序列的表示拼接起来
        patient_emb = torch.cat([condition_emb, procedure_emb], dim=-1)
        # 通过线性层生成查询向量
        queries = self.query_layer(patient_emb)
        # 获取每个患者的最后一次访问的查询向量
        query = get_last_visit(queries, mask)
        return query

    def generate_smiles_list(self) -> List[List[str]]:
        """
        生成 SMILES 列表，用于构建分子图。

        Returns:
            all_smiles_list: SMILES 列表，每个元素对应一个 ATC-3 分类的 SMILES 列表
        """
        atc3_to_smiles = {}
        atc = ATC()
        # 遍历 ATC 图中的节点，提取 ATC-3 分类和对应的 SMILES
        for code in atc.graph.nodes:
            if len(code) != 7:
                continue
            code_atc3 = ATC.convert(code, level=3)
            smiles = atc.graph.nodes[code]["smiles"]
            if smiles != smiles:
                continue
            atc3_to_smiles[code_atc3] = atc3_to_smiles.get(code_atc3, []) + [smiles]
        # 为了计算效率，每个 ATC-3 分类只取前 topk_smiles 个 SMILES
        atc3_to_smiles = {
            k: v[: self.args.topk_smiles] for k, v in atc3_to_smiles.items()
        }
        all_smiles_list = [[] for _ in range(self.label_size)]
        vocab_to_index = self.label_tokenizer.vocabulary
        # 将 ATC-3 分类映射到标签索引，并将对应的 SMILES 添加到列表中
        for atc3, smiles_list in atc3_to_smiles.items():
            if atc3 in vocab_to_index:
                index = vocab_to_index(atc3)
                all_smiles_list[index] += smiles_list
        return all_smiles_list

    def generate_average_projection(self) -> Tuple[torch.Tensor, List[str]]:
        """
        生成平均投影矩阵，用于将分子图表示映射到目标维度。

        Returns:
            average_projection: 平均投影矩阵，形状为 [num_classes, max_smiles_per_class]
            molecule_set: 所有 SMILES 的集合
        """
        molecule_set, average_index = [], []
        # 遍历每个 ATC-3 分类的 SMILES 列表，统计每个分类下的 SMILES 数量
        for smiles_list in self.all_smiles_list:
            counter = 0
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                molecule_set.append(smiles)
                counter += 1
            average_index.append(counter)
        # 初始化平均投影矩阵
        average_projection = np.zeros((len(average_index), sum(average_index)))
        col_counter = 0
        # 填充平均投影矩阵，使得每个分类的 SMILES 均匀分布
        for i, item in enumerate(average_index):
            if item <= 0:
                continue
            average_projection[i, col_counter : col_counter + item] = 1 / item
            col_counter += item
        average_projection = torch.FloatTensor(average_projection)
        return average_projection, molecule_set

    def generate_ddi_adj(self) -> torch.FloatTensor:
        """
        生成药物-药物相互作用 (DDI) 邻接矩阵。

        Returns:
            ddi_adj: DDI 邻接矩阵，形状为 [label_size, label_size]
        """
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        # 将 DDI 列表中的药物转换为 ATC-3 分类
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        # 填充邻接矩阵
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

class FusionModule(nn.Module):
    def __init__(self, hidden_size, label_size) -> None:
        """
        初始化融合模块，用于将分子图表示和查询向量融合。

        Args:
            hidden_size: 隐藏层维度
            label_size: 标签数量
        """
        super().__init__()
        # 定义一个线性层，用于计算图权重
        self.query_graph_rela = nn.Sequential(
            nn.Linear(hidden_size, label_size), nn.Sigmoid()
        )

    def forward(self, graph_repr, query):
        """
        前向传播函数，计算融合后的表示。

        Args:
            graph_repr: 分子图表示，形状为 [batch_size, num_nodes, hidden_size]
            query: 查询向量，形状为 [batch_size, hidden_size]

        Returns:
            h: 融合后的表示，形状为 [batch_size, num_nodes, 2*hidden_size]
        """
        # 计算图权重矩阵
        graph_weight = torch.diag_embed(self.query_graph_rela(query))
        # 使用图权重矩阵对分子图表示进行加权
        graph_repr = torch.matmul(graph_weight, graph_repr)

        n_graphs = graph_repr.size(1)
        # 将查询向量扩展并与分子图表示拼接
        h = torch.cat([graph_repr, query.unsqueeze(1).expand(-1, n_graphs, -1)], -1)
        return h

class Predictor(nn.Module):
    def __init__(self, hidden_size, label_size) -> None:
        """
        初始化预测器模块，用于预测多标签分类结果。

        Args:
            hidden_size: 隐藏层维度
            label_size: 标签数量
        """
        super().__init__()
        # 定义融合模块，用于将分子图表示和查询向量融合
        self.fusion_module = FusionModule(hidden_size, label_size)
        # 定义预测器，包含线性层、LayerNorm、ReLU 和 Dropout
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, graph_repr, query):
        """
        前向传播函数，计算预测 logits。

        Args:
            graph_repr: 分子图表示，形状为 [batch_size, num_nodes, hidden_size]
            query: 查询向量，形状为 [batch_size, hidden_size]

        Returns:
            logits: 预测 logits，形状为 [batch_size, label_size]
        """
        # 通过融合模块计算融合后的表示
        h = self.fusion_module(graph_repr, query)
        # 通过预测器计算 logits
        logits = self.predictor(h).squeeze(-1)
        return logits

class CRecBase(BaseModel):
    def __init__(self, args, dataset: SampleEHRDataset, hidden_size: int = 64):
        """
        初始化基础模型 CRecBase。

        Args:
            args: 配置参数对象
            dataset: EHR 数据集对象
            hidden_size: 隐藏层维度，默认为 64
        """
        super().__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],  # 使用的条件和操作特征键
            label_key="drugs",  # 标签键
            mode="multilabel",  # 多标签分类任务
        )
        self.args = args
        self.best_monitor_metric = 0.0  # 记录最佳监控指标
        self.hidden_size = hidden_size

        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(args, dataset)

        # 初始化预测器
        self.predictor = Predictor(hidden_size, self.feature_extractor.label_size)

        # 初始化平均投影矩阵
        self.average_projection = self.feature_extractor.average_projection

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播函数，提取特征并计算预测 logits。

        Args:
            conditions: 条件序列
            procedures: 操作序列

        Returns:
            包含预测 logits 的字典
        """
        # 使用特征提取器提取分子图的节点表示和查询向量
        node_repr, query = self.feature_extractor(conditions, procedures)
        # 使用全局平均池化计算图表示
        _graph_repr_org = global_mean_pool(
            node_repr, self.feature_extractor.molecule_graphs.batch
        )
        # 使用平均投影矩阵将图表示映射到目标维度
        average_projection = self.average_projection.to(self.device)
        graph_repr_org = torch.matmul(average_projection, _graph_repr_org)

        # 通过预测器计算 logits
        logits = self.predictor(graph_repr_org, query)
        return {"logits": logits}

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        训练步骤函数，定义单个 batch 的训练过程。

        Args:
            batch: 当前 batch 的数据
            batch_idx: 当前 batch 的索引

        Returns:
            loss: 当前 batch 的损失值
        """
        # 从 batch 中提取条件和操作序列以及标签
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        # 准备标签数据，包括二分类标签和多分类标签
        bce_labels, multi_labels = self.prepare_labels(drugs)

        # 调用 forward 函数，得到模型的输出结果
        output = self.forward(conditions, procedures)

        # 计算任务损失（如交叉熵损失），并累加到总损失中
        loss = self.calc_loss(output["logits"], bce_labels, multi_labels)

        # 使用 logger 记录损失值，便于监控训练过程
        self.log("train/loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        """
        在验证 epoch 开始时调用，初始化记录变量。
        """
        self.y_true_all = []
        self.y_prob_all = []

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        验证步骤函数，定义单个 batch 的验证过程。

        Args:
            batch: 当前 batch 的数据
            batch_idx: 当前 batch 的索引

        Returns:
            None
        """
        # 从 batch 中提取条件和操作序列以及标签
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        # 准备标签数据，包括索引和多热编码
        labels_index = self.feature_extractor.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # 将标签索引转换为多热编码
        labels = batch_to_multihot(labels_index, self.feature_extractor.label_size)

        # 调用 forward 函数，得到模型的输出结果
        output = self.forward(conditions, procedures)
        # 计算预测概率
        y_prob = torch.sigmoid(output["logits"])

        # 将真实标签和预测概率转换为 numpy 数组，并记录下来
        y_true = labels.cpu().numpy()
        y_prob = y_prob.cpu().numpy()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_validation_epoch_end(self) -> None:
        """
        在验证 epoch 结束时调用，计算评估指标并记录日志。
        """
        # 将所有 batch 的真实标签和预测概率拼接起来
        y_true_all = np.concatenate(self.y_true_all, axis=0)
        y_prob_all = np.concatenate(self.y_prob_all, axis=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        # 计算多标签评估指标
        scores = multi_label_metric(
            y_prob_all, y_true_all, self.feature_extractor.ddi_adj
        )

        # 记录各项评估指标
        for key in scores.keys():
            self.log(f"val/{key}", scores[key])

        # 获取当前 epoch 的 Jaccard 相似度 (JA) 指标
        monitor_metric = scores["ja"]
        # 如果当前 JA 指标优于历史最佳指标，则更新最佳指标并记录日志
        if monitor_metric > self.best_monitor_metric:
            self.best_monitor_metric = monitor_metric
            logger.info(
                f"New best ja: {self.best_monitor_metric:.4f} in epoch {self.trainer.current_epoch}"
            )

        # 记录最佳 JA 指标
        self.log("val/best_ja", self.best_monitor_metric)

    def on_test_epoch_start(self) -> None:
        """
        在测试 epoch 开始时调用，初始化记录变量。
        """
        self.y_true_all = []
        self.y_prob_all = []

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        测试步骤函数，定义单个 batch 的测试过程。

        Args:
            batch: 当前 batch 的数据
            batch_idx: 当前 batch 的索引

        Returns:
            None
        """
        # 从 batch 中提取条件和操作序列以及标签
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        # 准备标签数据，包括索引和多热编码
        labels_index = self.feature_extractor.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # 将标签索引转换为多热编码
        labels = batch_to_multihot(labels_index, self.feature_extractor.label_size)

        # 调用 forward 函数，得到模型的输出结果
        output = self.forward(conditions, procedures)
        # 计算预测概率
        y_prob = torch.sigmoid(output["logits"])

        # 将真实标签和预测概率转换为 numpy 数组，并记录下来
        y_true = labels.cpu().numpy()
        y_prob = y_prob.cpu().numpy()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_test_epoch_end(self) -> Any:
        """
        在测试 epoch 结束时调用，计算评估指标并记录日志。
        """
        # 将所有 batch 的真实标签和预测概率拼接起来
        y_true_all = np.concatenate(self.y_true_all, axis=0)
        y_prob_all = np.concatenate(self.y_prob_all, axis=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        # 计算多标签评估指标
        scores = multi_label_metric(
            y_prob_all, y_true_all, self.feature_extractor.ddi_adj
        )

        # 记录各项评估指标
        for key in scores.keys():
            logger.info(f"test/{key}: {scores[key]}")
            self.log(f"test/{key}", scores[key])
        return scores

    def configure_optimizers(self):
        """
        配置优化器。

        Returns:
            optimizer: 配置好的优化器
        """
        # 使用 Adam 优化器，学习率由配置参数指定，权重衰减为 1e-5
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-5)
        return [opt]

    def prepare_labels(self, drugs):
        """
        准备标签数据，包括二分类标签和多分类标签。

        Args:
            drugs: 药物标签列表

        Returns:
            labels: 二分类标签，形状为 [batch_size, label_size]
            multi_labels: 多分类标签，形状为 [batch_size, label_size]
        """
        # 使用标签分词器将药物标签转换为索引
        labels_index = self.feature_extractor.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # 将标签索引转换为多热编码
        labels = batch_to_multihot(labels_index, self.feature_extractor.label_size)

        # 初始化多分类标签矩阵
        multi_labels = -np.ones(
            (len(labels), self.feature_extractor.label_size), dtype=np.int64
        )
        # 遍历每个样本的标签索引，去除冗余标签并填充多分类标签矩阵
        for idx, cont in enumerate(labels_index):
            # 去除重复标签
            cont = list(set(cont))
            # 填充多分类标签矩阵
            multi_labels[idx, : len(cont)] = cont
        # 将多分类标签矩阵转换为 PyTorch 张量
        multi_labels = torch.from_numpy(multi_labels)
        return labels.to(self.device), multi_labels.to(self.device)

    def calc_loss(self, logits, bce_labels, multi_labels):
        """
        计算损失函数，包括二分类交叉熵损失和多标签边缘损失。

        Args:
            logits: 预测 logits，形状为 [batch_size, label_size]
            bce_labels: 二分类标签，形状为 [batch_size, label_size]
            multi_labels: 多分类标签，形状为 [batch_size, label_size]

        Returns:
            loss_task: 总任务损失
        """
        # 计算二分类交叉熵损失
        loss_bce = F.binary_cross_entropy_with_logits(logits, bce_labels)
        # 计算多标签边缘损失
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), multi_labels)
        # 加权求和得到总任务损失
        loss_task = 0.95 * loss_bce + 0.05 * loss_multi
        return loss_task
