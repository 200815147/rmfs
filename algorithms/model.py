import pdb

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

from common_args import env_attr
from utils import dict_to_batch_tensor, flatten_obs


class VacancyMLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.vacancy_mlp = nn.Sequential(
            nn.Linear(1, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
        )
        self.shelf_mlp = nn.Sequential(
            nn.Linear(1 + input_dim, embed_dim), nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim), nn.LeakyReLU()
        )
        self.embed_dim = embed_dim

    def forward(self, state, x):
        # state [B, Nv, 1]
        # x [B, Nv, dim] 2(vacancy coord) + 1(shelf state) + env.n_sku_types
        # x [B, Nv, dim] 1(distance) + 1(shelf state) + env.n_sku_types
        # pdb.set_trace()
        state = state.squeeze(-1)
        B = state.shape[0]
        Nv = state.shape[1]
        v_batch_idx, v_idx = torch.where(state == env_attr.n_shelves)
        v_feat = x[v_batch_idx, v_idx, :1]
        v_out = self.vacancy_mlp(v_feat)
        s_batch_idx, s_idx = torch.where(state != env_attr.n_shelves)
        s_feat = x[s_batch_idx, s_idx]
        s_out = self.shelf_mlp(s_feat)
        out = torch.zeros((B, Nv, self.embed_dim), device=x.device).float()
        out[v_batch_idx, v_idx] = v_out
        out[s_batch_idx, s_idx] = s_out
        return out

class TransformerModule(TorchRLModule, ValueFunctionAPI):
    """
    使用 Transformer 的分支编码模型：
    输入：
      batch['obs'] 是一个 dict，包含多个子空间：
        - map['id']: Tensor of shape [B, num_cells]
        - robots fields: state [B, N_robots], coord [B, N_robots, 2], target [B, N_robots,2], shelf [B, N_robots]
        - shelves fields: state [B, N_shelves], coord [B, N_shelves,2], inventory [B, N_shelves, N_sku]
        - workstations fields: coord [B, N_ws,2], demand [B, N_ws, N_sku]
        - global['next_robot']: Tensor of shape [B]
    输出：
      forward_inference: {'logits': List of tensors, each [B, action_dim_i]}  对应 MultiDiscrete action heads
      forward_value: {'value': Tensor [B,1]}
    """
    @override(TorchRLModule)
    def setup(self):
        super().setup()

        # 嵌入维度
        emb_dim = self.model_config.get('emb_dim', 64)

        # 1) 地图 Embedding: 将每个格子ID嵌入，并对所有格子求和得到 [B, emb_dim]
        # map_space = self.observation_space['map']['id']
        # num_cells = map_space.nvec[0]  # 格子总数
        # pdb.set_trace()
        # self.map_emb = nn.Embedding(num_cells, emb_dim)

        # 2) 机器人特征 MLP: 把 state, coord, target, shelf 拼接后 MLP -> [B, emb_dim]
        robots = self.observation_space['robots']
        # pdb.set_trace()
        # robot_dim = (
        #     robots['state'].nvec.shape[0]
        #     + 2 * robots['coord'].shape[1]
        #     + 2 * robots['target'].shape[1]
        #     + robots['shelf'].nvec.shape[0]
        # )
        robot_dim = 1 + 2 + 2 + 1
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_dim, emb_dim), nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim), nn.LeakyReLU()
        )

        # 3) 货架特征 MLP: state, coord, inventory -> [B, emb_dim]
        shelves = self.observation_space['shelves']
        # shelf_dim = (
        #     shelves['state'].nvec.shape[0]
        #     + 2 * shelves['coord'].shape[1]
        #     + shelves['inventory'].shape[0] * shelves['inventory'].shape[1]
        # )
        shelf_dim = 1 + env_attr.n_sku_types
        self.vacancy_mlp = VacancyMLP(shelf_dim, emb_dim)

        # 4) 工作站特征 MLP: coord, demand -> [B, emb_dim]
        ws = self.observation_space['workstations']
        ws_dim = 1 + env_attr.n_sku_types
        self.ws_mlp = nn.Sequential(
            nn.Linear(ws_dim, emb_dim), nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim), nn.LeakyReLU()
        )

        print(f'robot dim: {robot_dim} vacancy dim: {shelf_dim} workstation dim: {ws_dim}')

        # 5) 全局 Embedding: next_robot -> [B, emb_dim]
        global_space = self.observation_space['global']['next_robot']
        self.global_emb = nn.Embedding(global_space.nvec[0], emb_dim)

        # 6) Transformer Encoder: 接收实体序列长度 L, 每个 embedding 大小 emb_dim
        nhead = self.model_config.get('nhead', 4)
        num_layers = self.model_config.get('num_layers', 2)
        ff_dim = self.model_config.get('ff_dim', emb_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=ff_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 7) Policy heads: 针对 MultiDiscrete 每个动作维度一个线性层
        self.policy_heads = nn.Linear(emb_dim, 1)

        # 8) Value head: 聚合特征 -> 单 scalar
        # self.value_head = nn.Linear(emb_dim * (env_attr.n_robots + env_attr.n_shelves + env_attr.n_workstations), 1)
        self.value_head = nn.Linear(1, 1)
        # self.value_head = nn.Sequential(
        #     nn.Linear(emb_dim * (env_attr.n_robots + env_attr.n_shelves + env_attr.n_workstations), emb_dim), nn.LeakyReLU(),
        #     nn.Linear(emb_dim, 1)
        # )

    def _encode_entities(self, batch):
        """
        将各实体编码到同一 embedding 空间，输出各 [B, emb_dim]
        """
        # 地图
        # map_ids = batch['obs']['map']['id']  # [B, num_cells]
        # map_emb = self.map_emb(map_ids).sum(dim=1)  # [B, emb_dim]

        B = batch['obs']['robots']['state'].shape[0]
        if isinstance(batch['obs']['robots']['state'], np.ndarray): # compute_values 传进来的是 np.ndarray 且没有 batch
            batch['obs'] = dict_to_batch_tensor(batch['obs'])
            B = 1
        
        # 机器人特征编码: [B, N_r, feat] -> MLP -> [B, N_r, emb_dim]
        robots = batch['obs']['robots']
        feat_r = torch.cat([
            robots['state'].float().unsqueeze(-1),                 # [B, N_r]
            robots['coord'].float(),          # [B, 2*N_r]
            robots['target'].float(),         # [B, 2*N_r]
            robots['shelf'].float().unsqueeze(-1)                  # [B, N_r]
        ], dim=-1).view(B, env_attr.n_robots, -1)                # [B, N_r, robot_feat_dim]
        robot_emb = self.robot_mlp(feat_r)            # MLP applies over last dim: [B, N_r, emb_dim]

        # 货架特征编码: [B, N_s, feat] -> MLP -> [B, N_s, emb_dim]
        shelves = batch['obs']['shelves']
        vacancies = batch['obs']['vacancies']
        feat_s = torch.cat([
            vacancies['distance'].float().unsqueeze(-1),  
            # vacancies['coord'].float(),  
            shelves['state'].float().unsqueeze(-1),                 # [B, N_s]
            shelves['inventory'].float()
        ], dim=-1).view(B, env_attr.n_shelves, -1)                # [B, N_s, shelf_feat_dim]
        vacancy_emb = self.vacancy_mlp(vacancies['state'].view(B, env_attr.n_shelves, -1), feat_s)            # [B, N_s, emb_dim]
        # pdb.set_trace()
        # 工作站特征编码: [B, N_ws, feat] -> MLP -> [B, N_ws, emb_dim]
        ws = batch['obs']['workstations']
        feat_ws = torch.cat([
            ws['distance'].float().unsqueeze(-1),
            # ws['coord'].float(),
            ws['demand'].float()
        ], dim=-1).view(B, env_attr.n_workstations, -1).type(torch.float32)               # [B, N_ws, ws_feat_dim]
        ws_emb = self.ws_mlp(feat_ws)             # [B, N_ws, emb_dim]

        # 全局
        # gr = batch['obs']['global']['next_robot']  # [B]
        # global_emb = self.global_emb(torch.as_tensor(gr, dtype=torch.long))  # [B, emb_dim]

        map_emb = global_emb = None # TODO 暂时不用
        return map_emb, robot_emb, vacancy_emb, ws_emb, global_emb

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        """
        推理步骤：编码实体 -> Transformer -> 提取全局 token -> logits
        """
        map_emb, robot_emb, vacancy_emb, ws_emb, global_emb = self._encode_entities(batch)

        # 构建 Transformer 输入序列: 全局 token + N_r robots + N_s shelves + N_ws workstations + map token
        seq_list = []
        # seq_list.append(emb_g.unsqueeze(1))       # [B,1,D]
        seq_list.append(robot_emb)                    # [B,N_r,D]
        seq_list.append(vacancy_emb)                    # [B,N_s,D]
        seq_list.append(ws_emb)                   # [B,N_ws,D]
        # seq_list.append(map_emb.unsqueeze(1))     # [B,1,D]
        seq = torch.cat(seq_list, dim=1)          # [B, L, D]
        # pdb.set_trace()
        # Transformer expects [L, B, D]
        seq_t = seq.transpose(0, 1)                # [L, B, D]
        enc = self.transformer(seq_t)             # [L, B, D]
        # 各动作维度 logits
        logits = enc.transpose(0, 1)[:, env_attr.n_robots:, :]
        logits = self.policy_heads(logits).squeeze(-1)
        action_mask = batch['obs']['global']["action_mask"]
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
            logits = torch.where(
                action_mask,
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            )
        # print('fuck')
        # print(batch['obs']['robots']['state'])
        # print(logits)
        # print(batch['obs']['robots']['state'].shape)
        # print(logits.shape)
        # pdb.set_trace()
        return {
            Columns.ACTION_DIST_INPUTS: logits
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, **kwargs):
        feat = batch['obs']['robots']['state'].float() # [B, N_r]
        # pdb.set_trace()
        return self.value_head(feat) 

        map_emb, robot_emb, vacancy_emb, ws_emb, global_emb = self._encode_entities(batch)

        seq_list = []
        # seq_list.append(emb_g.unsqueeze(1))       # [B,1,D]
        seq_list.append(robot_emb)                    # [B,N_r,D]
        seq_list.append(vacancy_emb)                    # [B,N_s,D]
        seq_list.append(ws_emb)                   # [B,N_ws,D]
        # seq_list.append(map_emb.unsqueeze(1))     # [B,1,D]
        seq = torch.cat(seq_list, dim=1)          # [B, L, D]
        B = seq.shape[0]
        seq = seq.view(B, -1)
        value = self.value_head(seq)  # [B,1]
        # print(value)
        # pdb.set_trace()
        return value


class MyRLModule(TorchRLModule, ValueFunctionAPI): # fuck 这个 ValueFunctionAPI 必须要加
    @override(TorchRLModule)
    def setup(self):
        super().setup()
        # flat_space = flatten_space(self.observation_space)
        # self.input_dim = flat_space.shape[0]
        
        self.input_dim = 257
        hidden_layers = self.model_config.get("fcnet_hiddens", [256, 256])
        layers = []
        last = self.input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last, h))
            layers.append(nn.LeakyReLU())
            last = h
        self.shared_net = nn.Sequential(*layers)

        self.policy_head = nn.Linear(last, env_attr.x_max * env_attr.y_max)
        self.value_head = nn.Linear(last, 1)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        action_mask = obs['global']["action_mask"]
        flat = flatten_obs(obs)
        x = torch.as_tensor(flat, dtype=torch.float32)
        shared = self.shared_net(x)
        logits = self.policy_head(shared)
        
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.float32)
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
        obs = batch[Columns.OBS]
        flat = flatten_obs(obs)
        x = torch.as_tensor(flat, dtype=torch.float32)
        shared = self.shared_net(x)
        value = self.value_head(shared).squeeze(-1)
        return value

