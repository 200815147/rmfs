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


class HierarchicalModule(TorchRLModule, ValueFunctionAPI):
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
        self.embed_dim = embed_dim
        shelf_dim = env_attr.n_sku_types
        ws_dim = 1 + 2 + env_attr.n_sku_types
        # Transformer Encoder: 接收实体序列长度 L, 每个 embedding 大小 embed_dim
        nhead = self.model_config.get('nhead', 8)
        num_layers = self.model_config.get('num_layers', 6)
        ff_dim = embed_dim * nhead

        self.num_stages = 3
        self.robot_mlp = nn.ModuleList()
        self.vacancy_mlp = nn.ModuleList()
        self.ws_mlp = nn.ModuleList()
        self.transformer = nn.ModuleList()
        self.policy_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.robot_mlp.append(RobotMLP(shelf_dim, 2, embed_dim)) # [x, y]
            self.vacancy_mlp.append(VacancyMLP(shelf_dim, 3, embed_dim)) # [x, y, dis]
            self.ws_mlp.append(nn.Sequential(
                nn.Linear(ws_dim, embed_dim), nn.LeakyReLU(),
                nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
            ))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim, batch_first=True
            )
            self.transformer.append(nn.TransformerEncoder(encoder_layer, num_layers=num_layers))
            self.policy_head.append(nn.Linear(embed_dim, 1))

        seq_len = 1 + env_attr.n_shelves + env_attr.n_workstations
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
        # pdb.set_trace()
        
        shelves = batch['obs']['shelves']
        vacancies = batch['obs']['vacancies']
        ws = batch['obs']['workstations']
        device = next_robot_state.device
        vacancy_emb = torch.zeros((B, env_attr.n_shelves, self.embed_dim), device=device)
        ws_emb = torch.zeros((B, env_attr.n_workstations, self.embed_dim), device=device)
        robot_emb = torch.zeros((B, 1, self.embed_dim), device=device)
        for i in range(self.num_stages):
            idx = torch.where(next_robot_state == i)[0]
            if len(idx) == 0:
                continue
            try:
                feat_s = torch.cat([
                    vacancies['distance'][idx].float().unsqueeze(-1),      
                    vacancies['coord'][idx].float(),    
                    shelves['inventory'][idx].float()
                ], dim=-1).view(len(idx), env_attr.n_shelves, -1)                # [B, N_s, shelf_feat_dim]
            except:
                pdb.set_trace()
            try:
                vacancy_emb[idx] = self.vacancy_mlp[i](vacancies['state'][idx].view(len(idx), env_attr.n_shelves, -1), feat_s)            # [B, N_s, embed_dim]
            except:
                pdb.set_trace()
            # 机器人特征编码: [B, N_r, feat] -> MLP -> [B, N_r, embed_dim]
            
            try:
                next_robot_shelf = robots['shelf'][idx, batch['obs']['global']['next_robot'][idx].squeeze(-1)]
            except:
                pdb.set_trace()
            next_robot_coord = robots['coord'][idx, batch['obs']['global']['next_robot'][idx].squeeze(-1)].float()
            robot_emb[idx] = self.robot_mlp[i](next_robot_shelf, next_robot_coord, shelves['inventory'][idx].float())            # MLP applies over last dim: [B, N_r, embed_dim]

            feat_ws = torch.cat([
                ws['distance'][idx].float().unsqueeze(-1),
                ws['coord'][idx].float(),
                ws['demand'][idx].float()
            ], dim=-1).view(len(idx), env_attr.n_workstations, -1).type(torch.float32)               # [B, N_ws, ws_feat_dim]
            ws_emb[idx] = self.ws_mlp[i](feat_ws)             # [B, N_ws, embed_dim]

        return robot_emb, vacancy_emb, ws_emb

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        """
        推理步骤：编码实体 -> Transformer -> 提取全局 token -> logits
        """
        robot_emb, vacancy_emb, ws_emb = self._encode_entities(batch)

        # 构建 Transformer 输入序列: 全局 token + N_r robots + N_s shelves + N_ws workstations + map token
        seq_list = []
        seq_list.append(vacancy_emb)                    # [B,N_s,D]
        seq_list.append(ws_emb)                   # [B,N_ws,D]
        seq_list.append(robot_emb)                    # [B,1,D]

        seq = torch.cat(seq_list, dim=1)          # [B, L, D]

        B = seq.shape[0]
        robots = batch['obs']['robots']
        next_robot_state = robots['state'][torch.arange(B), batch['obs']['global']['next_robot'].squeeze(-1)]
        device = next_robot_state.device
        logits = torch.zeros((B, env_attr.n_shelves + env_attr.n_workstations), device=device)
        for i in range(self.num_stages):
            idx = torch.where(next_robot_state == i)[0]
            if len(idx) == 0:
                continue
            enc = self.transformer[i](seq[idx])             # [B, L, D]
            try:
                tmp_emb = enc[:, :env_attr.n_shelves + env_attr.n_workstations, :]            # [B, L, D]
            except:
                pdb.set_trace()
            try:
                logits[idx] = self.policy_head[i](tmp_emb).squeeze(-1)
            except:
                pdb.set_trace()

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
        robot_emb, vacancy_emb, ws_emb = self._encode_entities(batch)
        seq_list = []
        seq_list.append(vacancy_emb)                    # [B,N_s,D]
        seq_list.append(ws_emb)                   # [B,N_ws,D]
        seq_list.append(robot_emb)                    # [B,1,D]
        
        seq = torch.cat(seq_list, dim=1)          # [B, L, D]
        B = seq.shape[0]
        seq = seq.view(B, -1)
        # pdb.set_trace()
        value = self.value_head(seq)  # [B,1]
        return value
