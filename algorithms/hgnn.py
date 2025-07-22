import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, Linear, NNConv, SAGEConv

from algorithms.models import RobotMLP, VacancyMLP, backward_hook
from common_args import env_attr
from utils import dict_to_batch_tensor


class HGNNModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        super().setup()
    # def __init__(self, node_type_names, edge_type_tuples, hidden_dim, out_dim, num_layers=2):
        # super().__init__()
        self.hidden_dim = self.model_config.get('hidden_dim')
        node_type_names = self.model_config.get('node_type_names')
        num_layers = self.model_config.get('num_layers')
        hidden_dim = self.model_config.get('hidden_dim')
        # 1. 输入层: 为每种节点类型创建独立的线性映射层
        self.in_proj = nn.ModuleDict()
        for node_type in node_type_names:
            # 假设 data['node_type'].x 存在
            # 这里的 in_channels 需要根据实际数据动态获取
            # 我们在主脚本中传入
            # in_channels = data[node_type].x.shape[1]
            # self.in_proj[node_type] = Linear(in_channels, hidden_dim)
            pass # 我们将在主脚本中动态创建

        # 2. 消息传递层:
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # HeteroConv 允许为每种边类型定义不同的卷积层
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_dim) # (-1, -1) 表示源和目标节点特征维度可以不同
                # --- 如何使用 edge_attr ---
                # 要使用边属性，需要替换成支持边属性的层，例如 NNConv
                # edge_type: NNConv(
                #     in_channels=(-1, -1),
                #     out_channels=hidden_dim,
                #     nn=nn.Sequential(nn.Linear(5, hidden_dim * hidden_dim)) # 5是edge_attr的维度
                # )
                # -------------------------
                for edge_type in edge_type_tuples
            }, aggr='sum') # 'sum', 'mean', 'max' or 'min'
            self.convs.append(conv)

        # 3. 输出层: 只为需要预测的 'type_a' 节点创建分类器
        self.out_proj = Linear(hidden_dim, out_dim)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
    # def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # 1. 初始化一个空的 HeteroData 对象
        data = HeteroData()

        # 2. 定义节点信息
        # 节点类型 'type_a'
        num_nodes_a = 100
        data['type_a'].x = torch.randn(num_nodes_a, 32) # 32维特征
        # 我们假设要对 type_a 节点进行分类，假设有 5 个类别
        data['type_a'].y = torch.randint(0, 5, (num_nodes_a,))

        # 节点类型 'type_b'
        num_nodes_b = 200
        data['type_b'].x = torch.randn(num_nodes_b, 64) # 64维特征

        # 节点类型 'type_c'
        num_nodes_c = 150
        data['type_c'].x = torch.randn(num_nodes_c, 16) # 16维特征

        print("节点信息设置完毕:")
        print(data)
        print("-" * 30)

        # 3. 定义边和边属性信息
        # PyG 的边索引格式为 [2, num_edges]

        # 边: ('type_a', 'link_aa', 'type_a')
        num_edges_aa = 300
        edge_index_aa = torch.randint(0, num_nodes_a, (2, num_edges_aa), dtype=torch.long)
        # 边属性: 距离 (1维) + 离散关系 (4维 one-hot)
        dist_aa = torch.rand(num_edges_aa, 1)
        rel_aa = torch.nn.functional.one_hot(torch.randint(0, 4, (num_edges_aa,)), num_classes=4).float()
        edge_attr_aa = torch.cat([dist_aa, rel_aa], dim=-1) # 拼接
        data['type_a', 'link_aa', 'type_a'].edge_index = edge_index_aa
        data['type_a', 'link_aa', 'type_a'].edge_attr = edge_attr_aa

        # 边: ('type_a', 'link_ab', 'type_b')
        num_edges_ab = 500
        edge_index_ab = torch.stack([
            torch.randint(0, num_nodes_a, (num_edges_ab,)),
            torch.randint(0, num_nodes_b, (num_edges_ab,))
        ], dim=0).long()
        dist_ab = torch.rand(num_edges_ab, 1)
        rel_ab = torch.nn.functional.one_hot(torch.randint(0, 4, (num_edges_ab,)), num_classes=4).float()
        edge_attr_ab = torch.cat([dist_ab, rel_ab], dim=-1)
        data['type_a', 'link_ab', 'type_b'].edge_index = edge_index_ab
        data['type_a', 'link_ab', 'type_b'].edge_attr = edge_attr_ab

        # 边: ('type_b', 'link_bc', 'type_c')
        num_edges_bc = 400
        edge_index_bc = torch.stack([
            torch.randint(0, num_nodes_b, (num_edges_bc,)),
            torch.randint(0, num_nodes_c, (num_edges_bc,))
        ], dim=0).long()
        dist_bc = torch.rand(num_edges_bc, 1)
        rel_bc = torch.nn.functional.one_hot(torch.randint(0, 4, (num_edges_bc,)), num_classes=4).float()
        edge_attr_bc = torch.cat([dist_bc, rel_bc], dim=-1)
        data['type_b', 'link_bc', 'type_c'].edge_index = edge_index_bc
        data['type_b', 'link_bc', 'type_c'].edge_attr = edge_attr_bc

        print("边信息设置完毕:")
        print(data)
        print("-" * 30)

        # 验证数据
        # PyG 会自动添加反向边，但这里我们只定义了单向的
        # 如果需要，可以手动添加或使用 T.ToUndirected() 转换
        # data = T.ToUndirected()(data) # 这是一个常用的预处理步骤

        # 检查数据是否有效
        data.validate()
        print("HeteroData 对象创建成功!")
        # 1. 应用输入映射
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = self.in_proj[node_type](x).relu()
        
        # 2. 消息传递
        for conv in self.convs:
            # 对于 SAGEConv, 我们不需要传入 edge_attr_dict
            x_dict_proj = conv(x_dict_proj, edge_index_dict)
            
            # --- 如果使用 NNConv ---
            # x_dict_proj = conv(x_dict_proj, edge_index_dict, edge_attr_dict)
            # ---------------------
            
            # 应用激活函数
            x_dict_proj = {key: x.relu() for key, x in x_dict_proj.items()}
            
        # 3. 获取 'type_a' 节点的输出并进行分类
        x_a = x_dict_proj['type_a']
        out = self.out_proj(x_a)
        
        return out
    
    @override(ValueFunctionAPI)
    def compute_values(self, batch, **kwargs):
        phase_emb, robot_emb, vacancy_emb, ws_emb = self._encode_entities(batch)
        seq_list = []
        seq_list.append(vacancy_emb)                    # [B,N_s,D]
        seq_list.append(ws_emb)                   # [B,N_ws,D]
        seq_list.append(robot_emb)                    # [B,1,D]
        if self.use_phase_embedding:
            seq_list.append(phase_emb)
        seq = torch.cat(seq_list, dim=1)          # [B, L, D]
        B = seq.shape[0]
        seq = seq.view(B, -1)
        # pdb.set_trace()
        value = self.value_head(seq)  # [B,1]
        return value