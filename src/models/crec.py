import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pyhealth.datasets import SampleEHRDataset  # 导入用于处理 EHR（电子健康记录）数据的 Dataset 类
from torch import nn
from torch_geometric.nn import global_mean_pool  # 用于图数据的池化操作

from src.models.crec_base import CRecBase, Predictor  # 导入基类和预测器模块

# 创建一个 logger，用于记录日志信息
logger = logging.getLogger(__name__)


# 定义 Separator 类，继承自 LightningModule，用于将分子图表示分离为两部分（如因果部分和混淆部分）
class Separator(L.LightningModule):
    def __init__(self, args, molecule_graphs, hidden_size) -> None:
        """
        初始化 Separator 模块。

        Args:
            args: 配置参数对象，包含训练相关的超参数等。
            molecule_graphs: 分子图数据结构，包含节点、边等信息。
            hidden_size: 隐藏层的维度大小。
        """
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.molecule_graphs = molecule_graphs
        num_nodes = molecule_graphs.num_nodes  # 获取分子图中的节点数量

        # 定义一个神经网络层，用于计算每个节点的权重
        # 输出是一个 [batch_size, num_nodes] 的张量，表示每个节点的权重
        self.node_weight_layer = nn.Sequential(
            nn.Linear(hidden_size, num_nodes),  # 输入 hidden_size，输出 num_nodes
            nn.Sigmoid(),  # 使用 Sigmoid 将权重限制在 [0, 1] 范围内
        )

        # 定义一个多层感知机 (MLP)，用于根据节点表示和查询向量计算分离得分
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size * 2),  # 输入 2*hidden_size，输出 hidden_size*2
            nn.LayerNorm(hidden_size * 2),  # 对 hidden_size*2 维度进行 Layer Normalization
            nn.ReLU(),  # 激活函数 ReLU
            nn.Linear(hidden_size * 2, 2),  # 最终输出 2 维，表示正负两部分的分离得分
        )

    def forward(self, node_repr, query):
        """
        前向传播函数，计算节点权重并对节点表示进行加权求和，最终分离出两部分表示。

        Args:
            node_repr: 节点表示，形状为 [batch_size, num_nodes, hidden_size]
            query: 查询向量，形状为 [batch_size, hidden_size]

        Returns:
            graph_repr_c: 分离出的正部分图表示
            graph_repr_s: 分离出的负部分图表示
            graph_repr_org: 原始图表示
            lambda_pos: 正部分的分离得分
        """
        # 计算节点权重矩阵，形状从 [batch_size, num_nodes] 扩展为 [batch_size, num_nodes, num_nodes]
        # 这里使用 diag_embed 将一维权重扩展为对角矩阵，用于加权节点表示
        node_weight = torch.diag_embed(
            self.node_weight_layer(query)
        )  # [b, n_nodes] -> [b, n_nodes, n_nodes]

        # 使用权重矩阵对节点表示进行加权求和，得到加权后的节点表示
        node_repr = torch.matmul(node_weight, node_repr)  # [bs, n_nodes, dim]

        n_nodes = node_repr.size(1)  # 获取节点数量
        # 将查询向量扩展为与节点表示相同的形状，便于拼接
        query = query.unsqueeze(1).expand(-1, n_nodes, -1)
        # 拼接节点表示和查询向量，形成新的特征表示 h
        h = torch.cat([node_repr, query], -1)

        # 将拼接后的特征 h 输入 MLP，计算分离得分 p
        p = self.mlp(h)  # [bs, n_nodes, 2]

        # 使用 Gumbel Softmax 对得分 p 进行采样，得到分离概率分布 lambda_i
        # tau 是温度参数，dim=-1 表示在最后一个维度上进行 Softmax
        lambda_i = F.gumbel_softmax(p, tau=1, dim=-1)
        # 分离出正部分得分 lambda_pos 和负部分得分 lambda_neg
        lambda_pos = lambda_i[:, :, 0].unsqueeze(dim=-1)  # 正部分得分
        lambda_neg = lambda_i[:, :, 1].unsqueeze(dim=-1)  # 负部分得分

        ########
        # 调用 separate 方法，根据得分将节点表示分离为两部分图表示
        graph_repr_c, graph_repr_s, graph_repr_org = self.separate(
            node_repr, lambda_pos, lambda_neg
        )

        return graph_repr_c, graph_repr_s, graph_repr_org, lambda_pos

    def separate(self, node_repr, pos_score, neg_score):
        """
        根据正负得分将节点表示分离为两部分图表示。

        Args:
            node_repr: 节点表示，形状为 [batch_size, num_nodes, hidden_size]
            pos_score: 正部分得分，形状为 [batch_size, num_nodes, 1]
            neg_score: 负部分得分，形状为 [batch_size, num_nodes, 1]

        Returns:
            _graph_repr_c: 正部分图表示
            _graph_repr_s: 负部分图表示
            _graph_repr_org: 原始图表示
        """
        # 将节点表示与正负得分相乘，分别得到正部分和负部分的节点表示
        # 最终形状为 [batch_size * num_nodes, hidden_size]
        node_repr_c = (pos_score * node_repr).view(-1, self.hidden_size)
        node_repr_s = (neg_score * node_repr).view(-1, self.hidden_size)

        batch_size = pos_score.size(0)  # 获取 batch 大小
        # 调用 repeat_mol_attr 方法，生成用于图池化的 batch 索引
        self.batch_batch = self.repeat_mol_attr(batch_size)

        # 对原始节点表示进行图池化，得到原始图表示
        _graph_repr_org = global_mean_pool(node_repr, self.molecule_graphs.batch)
        # 对正部分节点表示进行图池化，得到正部分图表示
        _graph_repr_c = global_mean_pool(node_repr_c, self.batch_batch).view(
            batch_size, -1, self.hidden_size
        )
        # 对负部分节点表示进行图池化，得到负部分图表示
        _graph_repr_s = global_mean_pool(node_repr_s, self.batch_batch).view(
            batch_size, -1, self.hidden_size
        )

        return _graph_repr_c, _graph_repr_s, _graph_repr_org

    def repeat_mol_attr(self, batch_size):
        """
        生成用于图池化的 batch 索引。

        Args:
            batch_size: 当前 batch 的大小

        Returns:
            batch_batch: 用于图池化的 batch 索引张量
        """
        # 初始化一个全零张量，形状为 [batch_size * num_nodes]，数据类型为 long
        batch_batch = torch.zeros(
            [batch_size, self.molecule_graphs["num_nodes"]],
            dtype=torch.long,
            device=self.device,
        )
        # 获取分子图中的图数量（即不同的分子数量）
        num_graphs = self.molecule_graphs["batch"].max() + 1
        # 遍历每个 batch，为每个分子分配唯一的 batch 索引
        for idx in range(batch_size):
            batch_batch[idx] = self.molecule_graphs["batch"] + idx * num_graphs
        # 将 batch_batch 张量展平为一维
        batch_batch = batch_batch.contiguous().view(-1)
        return batch_batch


# 定义 Intervene 类，继承自 LightningModule，用于对两部分图表示进行干预操作
class Intervene(L.LightningModule):
    def __init__(self, args, hidden_size) -> None:
        """
        初始化 Intervene 模块。

        Args:
            args: 配置参数对象，包含训练相关的超参数等。
            hidden_size: 隐藏层的维度大小。
        """
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size
        # 定义一个干预投影网络，用于将两部分图表示进行融合
        self.interv_proj = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),  # 输入 2*hidden_size，输出 hidden_size
            nn.LayerNorm(hidden_size),  # 对 hidden_size 维度进行 Layer Normalization
            nn.ReLU(),  # 激活函数 ReLU
            nn.Dropout(),  # Dropout 层，防止过拟合
            nn.Linear(hidden_size, hidden_size),  # 最终输出 hidden_size
        )

    def forward(self, graph_repr_c, graph_repr_s):
        """
        前向传播函数，根据训练模式和干预风格对两部分图表示进行干预操作。

        Args:
            graph_repr_c: 正部分图表示，形状为 [batch_size, num_graphs, hidden_size]
            graph_repr_s: 负部分图表示，形状为 [batch_size, num_graphs, hidden_size]

        Returns:
            graph_repr_intervend: 干预后的图表示
        """
        batch_size, n_graphs = graph_repr_c.size()[:2]  # 获取 batch 大小和图数量
        # 如果处于训练模式，对负部分图表示进行随机排列，增加数据的多样性
        if self.training:
            rand_idx = torch.randperm(batch_size)
            graph_repr_s = graph_repr_s[rand_idx]

        # 根据配置的干预风格选择不同的干预方式
        if self.args.interv_style == "interpolation":  # 插值干预
            # 随机生成一个插值系数 lambda_，形状为 [batch_size, n_graphs, hidden_size]
            lambda_ = self.args.epsilon * torch.rand(
                [batch_size, n_graphs, self.hidden_size], device=self.device
            )
            # 使用插值公式计算干预后的图表示
            graph_repr_intervend = (1 - lambda_) * graph_repr_c + lambda_ * graph_repr_s
        elif self.args.interv_style == "concat":  # 拼接干预
            # 将两部分图表示拼接后输入干预投影网络，得到干预后的图表示
            graph_repr_intervend = self.interv_proj(
                torch.cat([graph_repr_c, graph_repr_s], -1)
            )
        elif self.args.interv_style == "add":  # 直接相加干预
            # 直接将两部分图表示相加，得到干预后的图表示
            graph_repr_intervend = graph_repr_c + graph_repr_s
        else:
            raise ValueError("未知的干预风格！")  # 如果干预风格未定义，抛出错误
        return graph_repr_intervend


# 定义 CRec 类，继承自 CRecBase，是整个模型的主类
class CRec(CRecBase):
    def __init__(self, args, dataset: SampleEHRDataset, hidden_size: int = 64):
        """
        初始化 CRec 模型。

        Args:
            args: 配置参数对象，包含训练相关的超参数等。
            dataset: EHR 数据集对象，用于提取特征。
            hidden_size: 隐藏层的维度大小，默认为 64。
        """
        super().__init__(args=args, dataset=dataset, hidden_size=hidden_size)
        self.args = args
        self.best_monitor_metric = 0.0  # 用于记录最佳监控指标
        self.hidden_size = hidden_size

        # 初始化 Separator 模块，用于分离图表示
        self.separator = Separator(
            args, self.feature_extractor.molecule_graphs, hidden_size
        )
        # 初始化 Intervene 模块，用于对图表示进行干预操作
        self.intervene = Intervene(args, hidden_size)

        # 初始化两个预测器，分别用于正部分和负部分的预测
        self.predictor_pos = Predictor(hidden_size, self.feature_extractor.label_size)
        self.predictor_neg = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 2),
        )

        # 初始化平均投影矩阵，用于将图表示映射到目标维度
        self.average_projection = self.feature_extractor.average_projection

        # 保存超参数，便于后续加载模型时恢复配置
        self.save_hyperparameters()

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播函数，接收条件和操作序列，输出模型的预测结果。

        Args:
            conditions: 条件序列，格式为 List[List[List[str]]]
            procedures: 操作序列，格式为 List[List[List[str]]]
            **kwargs: 其他可选参数

        Returns:
            一个字典，包含模型的输出结果，如得分、logits 等
        """
        # 使用特征提取器对条件和操作序列进行编码，得到节点表示和查询向量
        node_repr, query = self.feature_extractor(conditions, procedures)

        # 使用 Separator 模块对节点表示进行分离，得到正负部分图表示及原始图表示
        _graph_repr_c, _graph_repr_s, _graph_repr_org, pos_score = self.separator(
            node_repr, query
        )

        # 使用平均投影矩阵将图表示映射到目标维度
        average_projection = self.average_projection.to(self.device)
        graph_repr_org = torch.matmul(average_projection, _graph_repr_org)  # [181] --> [193]
        graph_repr_c = torch.matmul(average_projection, _graph_repr_c)
        graph_repr_s = torch.matmul(average_projection, _graph_repr_s)

        # 使用 Intervene 模块对正负部分图表示进行干预操作，得到干预后的图表示
        interved_repr = self.intervene(graph_repr_c, graph_repr_s)

        # 使用预测器对干预后的图表示进行预测，得到最终的 logits
        logits = self.predictor(interved_repr, query)
        # 使用预测器对正部分图表示进行预测，得到正部分的 logits
        logits_pos = self.predictor_pos(graph_repr_c, query)
        # 使用预测器对负部分图表示进行预测，得到负部分的 logits
        logits_neg = self.predictor_neg(graph_repr_s)
        # 返回所有输出结果
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
        """
        训练步骤函数，定义单个 batch 的训练过程。

        Args:
            batch: 当前 batch 的数据
            batch_idx: 当前 batch 的索引

        Returns:
            loss: 当前 batch 的损失值
        """
        # 从 batch 中提取条件和操作序列
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs = batch["drugs"]

        # 准备标签数据，包括二分类标签和多分类标签
        bce_labels, multi_labels = self.prepare_labels(drugs)

        # 调用 forward 函数，得到模型的输出结果
        output = self.forward(conditions, procedures)

        # 初始化损失值为 0
        loss = 0
        # 计算任务损失（如交叉熵损失），并累加到总损失中
        loss_task = self.calc_loss(output["logits"], bce_labels, multi_labels)
        loss += loss_task

        # 计算正部分的损失，并乘以权重系数 w_pos 后累加到总损失中
        loss_pos = self.calc_loss(output["logits_pos"], bce_labels, multi_labels)
        loss += self.args.w_pos * loss_pos

        # 生成随机标签，用于计算负部分的 KL 散度损失
        random_label = (
            torch.ones_like(
                output["logits_neg"],
                dtype=torch.float,
                device=self.device,
            )
            / 2
        )
        # 计算负部分的 KL 散度损失，并乘以权重系数 w_neg 后累加到总损失中
        loss_neg = F.kl_div(
            F.log_softmax(output["logits_neg"], dim=-1),
            random_label,
            reduction="batchmean",
        )
        loss += self.args.w_neg * loss_neg

        # 计算正部分得分的均值作为正则化损失，并乘以权重系数 w_reg 后累加到总损失中
        loss_reg = torch.mean(output["score"])  # L1 norm
        loss += self.args.w_reg * loss_reg

        # 使用 logger 记录各项损失值，便于监控训练过程
        self.log("train/loss", loss)
        self.log("train/loss_task", loss_task)
        self.log("train/loss_pos", loss_pos)
        self.log("train/loss_neg", loss_neg)
        self.log("train/loss_reg", loss_reg)
        # 返回总损失值
        return loss
