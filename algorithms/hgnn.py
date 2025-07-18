import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, NNConv

class HeteroGNN(nn.Module):
    def __init__(self, node_type_names, edge_type_tuples, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
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