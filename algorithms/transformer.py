import pdb

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

from algorithms.models import RobotMLP, VacancyMLP, backward_hook
from common_args import env_attr
from utils import dict_to_batch_tensor


class TransformerModule(TorchRLModule, ValueFunctionAPI):
    """
    输出：
      _forward: {'logits': List of tensors, each [B, action_dim_i]}  对应 MultiDiscrete action heads
      compute_values: {'value': Tensor [B,1]}
    """
    @override(TorchRLModule)
    def setup(self):
        super().setup()
        
        # 嵌入维度
        embed_dim = self.model_config.get('embed_dim', 512)

        self.use_phase_embedding = self.model_config.get('use_phase_embedding', True)
        self.phase_embedding = nn.Embedding(env_attr.n_robot_state, embed_dim)
        
        shelf_dim = env_attr.n_sku_types
        self.robot_mlp = RobotMLP(shelf_dim, 2, embed_dim) # [x, y]
        self.vacancy_mlp = VacancyMLP(shelf_dim, 3, embed_dim) # [x, y, dis]

        # print('phase_embedding')
        # print(list(self.phase_embedding.parameters()))
        # print('robot_mlp')
        # print(self.robot_mlp.placeholder)
        
        ws_dim = 1 + 2 + env_attr.n_sku_types
        self.ws_mlp = nn.Sequential(
            nn.Linear(ws_dim, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
        )

        # 6) Transformer Encoder: 接收实体序列长度 L, 每个 embedding 大小 embed_dim
        nhead = self.model_config.get('nhead', 8)
        num_layers = self.model_config.get('num_layers', 6)
        ff_dim = embed_dim * nhead
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(embed_dim, 1)
        seq_len = 1 + env_attr.n_shelves + env_attr.n_workstations
        if self.use_phase_embedding:
            seq_len += 1
        self.value_head = nn.Linear(embed_dim * seq_len, 1)
        
        # self.apply(lambda module: module.register_full_backward_hook(backward_hook))

    def _encode_entities(self, batch):
        """
        将各实体编码到同一 embedding 空间，输出各 [B, embed_dim]
        """

        B = batch['obs']['robots']['state'].shape[0]
        if isinstance(batch['obs']['robots']['state'], np.ndarray): # compute_values 传进来的是 np.ndarray 且没有 batch
            batch['obs'] = dict_to_batch_tensor(batch['obs'])
            B = 1
            
        robots = batch['obs']['robots']
        next_robot_state = robots['state'][torch.arange(B), batch['obs']['global']['next_robot'].squeeze(-1)]
        phase_emb = self.phase_embedding(next_robot_state).unsqueeze(1)  # [B, 1, embed_dim]
        
        shelves = batch['obs']['shelves']
        vacancies = batch['obs']['vacancies']
        feat_s = torch.cat([
            vacancies['distance'].float().unsqueeze(-1),      
            vacancies['coord'].float(),    
            shelves['inventory'].float()
        ], dim=-1).view(B, env_attr.n_shelves, -1)                # [B, N_s, shelf_feat_dim]
        vacancy_emb = self.vacancy_mlp(vacancies['state'].view(B, env_attr.n_shelves, -1), feat_s)            # [B, N_s, embed_dim]

        # 机器人特征编码: [B, N_r, feat] -> MLP -> [B, N_r, embed_dim]
        
        # pdb.set_trace()
        next_robot_shelf = robots['shelf'][torch.arange(B), batch['obs']['global']['next_robot'].squeeze(-1)]
        next_robot_coord = robots['coord'][torch.arange(B), batch['obs']['global']['next_robot'].squeeze(-1)].float()
        robot_emb = self.robot_mlp(next_robot_shelf, next_robot_coord, shelves['inventory'].float())            # MLP applies over last dim: [B, N_r, embed_dim]

        ws = batch['obs']['workstations']
        feat_ws = torch.cat([
            ws['distance'].float().unsqueeze(-1),
            ws['coord'].float(),
            ws['demand'].float()
        ], dim=-1).view(B, env_attr.n_workstations, -1).type(torch.float32)               # [B, N_ws, ws_feat_dim]
        ws_emb = self.ws_mlp(feat_ws)             # [B, N_ws, embed_dim]

        return phase_emb, robot_emb, vacancy_emb, ws_emb

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        """
        推理步骤：编码实体 -> Transformer -> 提取全局 token -> logits
        """
        phase_emb, robot_emb, vacancy_emb, ws_emb = self._encode_entities(batch)

        # 构建 Transformer 输入序列: 全局 token + N_r robots + N_s shelves + N_ws workstations + map token
        seq_list = []
        seq_list.append(vacancy_emb)                    # [B,N_s,D]
        seq_list.append(ws_emb)                   # [B,N_ws,D]
        seq_list.append(robot_emb)                    # [B,1,D]
        if self.use_phase_embedding:
            seq_list.append(phase_emb)
        seq = torch.cat(seq_list, dim=1)          # [B, L, D]

        # pdb.set_trace()
        enc = self.transformer(seq)             # [B, L, D]
        logits = enc[:, :env_attr.n_shelves + env_attr.n_workstations, :]            # [B, L, D]
        logits = self.policy_head(logits).squeeze(-1)

        # 各动作维度 logits
        # logits = enc.transpose(0, 1)[:, env_attr.n_robots:, :]
        # logits = self.policy_heads(logits).squeeze(-1)
        action_mask = batch['obs']['global']["action_mask"]
        if action_mask is not None:
            logits = torch.where(
                action_mask.bool(),
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            )
        return {
            Columns.ACTION_DIST_INPUTS: logits
        }

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
