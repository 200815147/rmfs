import argparse
import json
import pdb
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import rl_utils
from common_args import env_attr
from rmfs_env import RMFSEnv
from utils import dict_to_batch_tensor


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

class PolicyNet(nn.Module):
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
    def __init__(self):
        super().__init__()

        # 嵌入维度
        emb_dim = 32

        robot_dim = 1 + 2 + 2 + 1
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_dim, emb_dim), nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim), nn.LeakyReLU()
        )
        
        shelf_dim = 1 + env_attr.n_sku_types
        self.vacancy_mlp = VacancyMLP(shelf_dim, emb_dim)

        ws_dim = 1 + env_attr.n_sku_types
        self.ws_mlp = nn.Sequential(
            nn.Linear(ws_dim, emb_dim), nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim), nn.LeakyReLU()
        )

        print(f'robot dim: {robot_dim} vacancy dim: {shelf_dim} workstation dim: {ws_dim}')

        # 6) Transformer Encoder: 接收实体序列长度 L, 每个 embedding 大小 emb_dim
        nhead = 4
        num_layers = 2
        ff_dim = emb_dim * nhead
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=ff_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decision_phase_embedding = nn.Embedding(env_attr.n_robot_state, emb_dim)

        # 7) Policy heads: 针对 MultiDiscrete 每个动作维度一个线性层
        self.policy_head = nn.Linear(emb_dim, 1)

        # 动作预测的输出头
        self.action_head = nn.Linear(emb_dim, env_attr.n_shelves + env_attr.n_workstations)

        # 8) Value head: 聚合特征 -> 单 scalar
        # self.value_head = nn.Linear(emb_dim * (env_attr.n_robots + env_attr.n_shelves + env_attr.n_workstations), 1)
        self.value_head = nn.Linear(1, 1)
        # self.value_head = nn.Sequential(
        #     nn.Linear(emb_dim * (env_attr.n_robots + env_attr.n_shelves + env_attr.n_workstations), emb_dim), nn.LeakyReLU(),
        #     nn.Linear(emb_dim, 1)
        # )

    def _encode_entities(self, batch):
        
        B = batch['robots']['state'].shape[0]
        if isinstance(batch['robots']['state'], np.ndarray): # compute_values 传进来的是 np.ndarray 且没有 batch
            batch = dict_to_batch_tensor(batch, device='cuda')
            B = 1
        # pdb.set_trace()
        # 机器人特征编码: [B, N_r, feat] -> MLP -> [B, N_r, emb_dim]
        robots = batch['robots']
        feat_r = torch.cat([
            robots['state'].float().unsqueeze(-1),                 # [B, N_r]
            robots['coord'].float(),          # [B, 2*N_r]
            robots['target'].float(),         # [B, 2*N_r]
            robots['shelf'].float().unsqueeze(-1)                  # [B, N_r]
        ], dim=-1).view(B, env_attr.n_robots, -1)                # [B, N_r, robot_feat_dim]
        robot_emb = self.robot_mlp(feat_r)            # MLP applies over last dim: [B, N_r, emb_dim]

        # 货架特征编码: [B, N_s, feat] -> MLP -> [B, N_s, emb_dim]
        shelves = batch['shelves']
        vacancies = batch['vacancies']
        feat_s = torch.cat([
            vacancies['distance'].float().unsqueeze(-1),  
            # vacancies['coord'].float(),  
            shelves['state'].float().unsqueeze(-1),                 # [B, N_s]
            shelves['inventory'].float()
        ], dim=-1).view(B, env_attr.n_shelves, -1)                # [B, N_s, shelf_feat_dim]
        vacancy_emb = self.vacancy_mlp(vacancies['state'].view(B, env_attr.n_shelves, -1), feat_s)            # [B, N_s, emb_dim]
        # pdb.set_trace()
        # 工作站特征编码: [B, N_ws, feat] -> MLP -> [B, N_ws, emb_dim]
        ws = batch['workstations']
        feat_ws = torch.cat([
            ws['distance'].float().unsqueeze(-1),
            # ws['coord'].float(),
            ws['demand'].float()
        ], dim=-1).view(B, env_attr.n_workstations, -1).type(torch.float32)               # [B, N_ws, ws_feat_dim]
        ws_emb = self.ws_mlp(feat_ws)             # [B, N_ws, emb_dim]

        map_emb = global_emb = None # TODO 暂时不用
        return map_emb, robot_emb, vacancy_emb, ws_emb, global_emb

    def forward(self, batch, **kwargs):
        """
        推理步骤：编码实体 -> Transformer -> 提取全局 token -> logits
        """
        map_emb, robot_emb, vacancy_emb, ws_emb, global_emb = self._encode_entities(batch)

        # 构建 Transformer 输入序列: 全局 token + N_r robots + N_s shelves + N_ws workstations + map token
        seq_list = []
        # seq_list.append(emb_g.unsqueeze(1))       # [B,1,D]
        # seq_list.append(robot_emb)                    # [B,N_r,D]
        seq_list.append(vacancy_emb)                    # [B,N_s,D]
        seq_list.append(ws_emb)                   # [B,N_ws,D]
        # seq_list.append(self.decision_phase_embedding)
        # seq_list.append(map_emb.unsqueeze(1))     # [B,1,D]
        seq = torch.cat(seq_list, dim=1)          # [B, L, D]
        # pdb.set_trace()
        # Transformer expects [L, B, D]
        seq_t = seq.transpose(0, 1)                # [L, B, D]
        enc = self.transformer(seq_t)             # [L, B, D]

        # logits = self.action_head(enc.transpose(0, 1)[:, -1, :])
        # 各动作维度 logits
        logits = enc.transpose(0, 1)
        # pdb.set_trace()
        # print(logits.shape)
        # logits = enc.transpose(0, 1)[:, env_attr.n_robots:, :]
        try:
            logits = self.policy_head(logits).squeeze(-1)
        except:
            pdb.set_trace()
        action_mask = batch['global']["action_mask"]
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
            logits = torch.where(
                action_mask,
                logits,
                torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            )
        # print(logits)
        # print(F.softmax(logits, dim=1))
        # pdb.set_trace()
        return F.softmax(logits, dim=1)
    
class REINFORCE:
    def __init__(self, learning_rate, gamma, device):
        self.policy_net = PolicyNet().to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        if isinstance(state['robots']['state'], np.ndarray): # compute_values 传进来的是 np.ndarray 且没有 batch
            state = dict_to_batch_tensor(state, device='cuda')
            
        probs = self.policy_net(state)
        try:
            action_dist = torch.distributions.Categorical(probs)
        except:
            pdb.set_trace()
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = dict_to_batch_tensor(state_list[i], device='cuda')
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            # pdb.set_trace()
            # print(loss.item())
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ray rllib
    parser.add_argument('--disable_env_info', action='store_true')
    parser.add_argument('--log_to_driver', action='store_true')
    parser.add_argument('--disable_local_mode', action='store_true')
    # environment
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_l', type=int, default=0)
    parser.add_argument('--seed_r', type=int, default=0)
    # config
    parser.add_argument('--model', type=str, default='heuristic')
    parser.add_argument('--layout', type=str, default='layout')
    # train
    parser.add_argument('--training_iteration', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--num_env_runners', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument("--disable_gae", action="store_true")
    parser.add_argument("--disable_critic", action="store_true")
    parser.add_argument('--num_sgd_iter', type=int, default=5)
    # heuristic
    parser.add_argument('--pick_heuristic', choices=['naive', 'nearest'], default='naive')
    parser.add_argument('--deliver_heuristic', choices=['naive', 'max_satify', 'nearest'], default='naive')
    parser.add_argument('--return_heuristic', choices=['origin', 'nearest'], default='origin')
    opts = parser.parse_args()

    env_config={
                "print_env_info": not opts.disable_env_info,
                "seed_l": opts.seed_l,
                "seed_r": opts.seed_r,
            }
    with open(f'{opts.layout}.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    env_config['shelves'] = json_data['shelves']
    env_config['workstations'] = json_data['workstations']

    learning_rate = 1e-4
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = RMFSEnv(config=env_config)
    torch.manual_seed(0)
    agent = REINFORCE(learning_rate, gamma, device)

    return_list = []
    for i in range(5):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                obs, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(obs)
                    next_obs, reward, done, _, _ = env.step(action)
                    transition_dict['states'].append(obs)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_obs)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    obs = next_obs
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 50 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 50 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-50:])
                    })
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    env_name = 'RMFS'
    from datetime import datetime

    # 获取当前时间并格式化为字符串（精确到秒）
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.savefig(f'output/{current_time}_fig1.png')
    # plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.savefig(f'output/{current_time}_fig2.png')
    # plt.show()

    rewards = []
    for i in range(opts.seed_l, opts.seed_r + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.take_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            obs = next_obs
            total_reward += reward
        rewards.append(total_reward)
    print(np.mean(rewards))
    pdb.set_trace()