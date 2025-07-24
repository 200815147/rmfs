import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. 常量和超参数定义 ---
# 环境维度
GRID_WIDTH = 20
GRID_HEIGHT = 20
NUM_ROBOTS = 5
NUM_PODS = 15
NUM_WORKSTATIONS = 3
NUM_SKU_TYPES = 10
MAX_ORDER_LINES = 5  # 每个订单最多包含的SKU种类
MAX_SKU_PER_LINE = 10 # 每个订单行中单个SKU的最大数量

# 模型维度
EMBED_DIM = 128      # 所有嵌入向量的维度
N_HEADS = 8          # Transformer中多头注意力的头数
N_LAYERS = 3         # Transformer编码器层数
FF_DIM = 512         # Transformer中前馈网络的隐藏层维度
DROPOUT = 0.1        # Dropout比率
CONTEXT_LENGTH = 10  # Decision Transformer的上下文长度 K，即考虑过去K个 (R, S, A) 元组

# 动作空间 (简化表示)
# 5个基本移动动作 (上, 下, 左, 右, 停留)
# + NUM_PODS (选择一个货架ID)
# + NUM_WORKSTATIONS (选择一个工作站ID)
# 实际应用中，动作空间可能更复杂，例如使用Pointer Networks [2, 3, 4, 5]
# 来动态指向有效实体。
ACTION_SPACE_SIZE = 5 + NUM_PODS + NUM_WORKSTATIONS

# 决策阶段 (在报告中定义)
DECISION_PHASES = {
    "SHELF_SELECTION": 0,
    "PATH_TO_WORKSTATION": 1,
    "PATH_TO_RETURN_LOCATION": 2,
    "WAIT": 3,
    "IDLE": 4 # 机器人空闲时的初始阶段
}
NUM_DECISION_PHASES = len(DECISION_PHASES)

# --- 2. 环境设置 (简化的 RMFS_Env) ---
# 这是一个用于演示的模拟环境。实际的RMFS环境会复杂得多。
class RMFS_Env:
    def __init__(self, grid_width, grid_height, num_robots, num_pods, num_workstations, num_sku_types):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_robots = num_robots
        self.num_pods = num_pods
        self.num_workstations = num_workstations
        self.num_sku_types = num_sku_types

        # 0: 空地, 1: 货架, 2: 工作站, 3: 机器人 (仅用于可视化，实际机器人位置在self.robots中)
        self.grid = np.zeros((grid_height, grid_width), dtype=int)
        self.robots = {}        # {robot_id: {'pos': (x,y), 'carrying_pod': None, 'task': None, 'decision_phase': 'IDLE'}}
        self.pods = {}          # {pod_id: {'pos': (x,y), 'original_pos': (x,y), 'skus': {sku_type: quantity}}}
        self.workstations = {}  # {ws_id: {'pos': (x,y), 'order_queue': deque()}}
        self.orders = {}        # {order_id: {'skus_needed': {sku_type: quantity}, 'target_ws': ws_id, 'priority': float, 'deadline': int, 'fulfilled_skus': {}}}
        self.current_order_id = 0
        self.current_time = 0

        self._initialize_environment()

    def _initialize_environment(self):
        # 放置工作站
        for i in range(self.num_workstations):
            while True:
                x, y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
                if self.grid[y, x] == 0:
                    self.grid[y, x] = 2 # 工作站
                    self.workstations[i] = {'pos': (x, y), 'order_queue': deque()}
                    break
        # 放置货架
        for i in range(self.num_pods):
            while True:
                x, y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
                if self.grid[y, x] == 0:
                    self.grid[y, x] = 1 # 货架
                    skus = {sku_type: random.randint(1, 50) for sku_type in range(self.num_sku_types)}
                    self.pods[i] = {'pos': (x, y), 'original_pos': (x, y), 'skus': skus}
                    break
        # 放置机器人
        for i in range(self.num_robots):
            while True:
                x, y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
                if self.grid[y, x] == 0:
                    self.robots[i] = {'pos': (x, y), 'carrying_pod': None, 'task': None, 'decision_phase': 'IDLE'}
                    break

    def reset(self):
        # 重置环境状态
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.robots = {}
        self.pods = {}
        self.workstations = {}
        self.orders = {}
        self.current_order_id = 0
        self.current_time = 0
        self._initialize_environment()
        # 返回机器人0在IDLE阶段的初始观测
        return self._get_observation(0, 'IDLE')

    def _generate_new_order(self):
        order_id = self.current_order_id
        self.current_order_id += 1
        num_lines = random.randint(1, MAX_ORDER_LINES)
        skus_needed = {}
        for _ in range(num_lines):
            sku_type = random.randint(0, self.num_sku_types - 1)
            quantity = random.randint(1, MAX_SKU_PER_LINE)
            skus_needed[sku_type] = skus_needed.get(sku_type, 0) + quantity
        target_ws_id = random.choice(list(self.workstations.keys()))
        priority = random.random() # 优先级，值越大越紧急
        deadline = self.current_time + random.randint(50, 200) # 截止时间
        self.orders[order_id] = {
            'skus_needed': skus_needed,
            'target_ws': target_ws_id,
            'priority': priority,
            'deadline': deadline,
            'fulfilled_skus': {sku_type: 0 for sku_type in skus_needed}
        }
        self.workstations[target_ws_id]['order_queue'].append(order_id)
        return order_id

    def _get_observation(self, robot_id, decision_phase):
        """
        生成当前环境的观测状态。
        包括全局网格信息、动态实体特征、当前机器人ID、决策阶段和目标回报。
        """
        # 全局网格地图观测 (多通道图像式)
        # 通道0: 空地, 通道1: 货架, 通道2: 工作站, 通道3: 机器人位置
        grid_obs = np.zeros((4, self.grid_height, self.grid_width), dtype=np.float32)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] == 0: grid_obs[0, y, x] = 1.0
                elif self.grid[y, x] == 1: grid_obs[1, y, x] = 1.0
                elif self.grid[y, x] == 2: grid_obs[2, y, x] = 1.0
        for r_id, r_info in self.robots.items():
            grid_obs[3, r_info['pos'][1], r_info['pos']] = 1.0

        # 动态实体特征 (用于GNN)
        # 节点类型: 机器人, 货架, 工作站, 订单
        robot_features =
        for r_id, r_info in self.robots.items():
            # (x, y, 是否搬运货架, 是否是当前决策机器人)
            robot_features.append([r_info['pos'], r_info['pos'][1], 1 if r_info['carrying_pod'] is not None else 0, 1 if r_id == robot_id else 0])

        pod_features =
        for p_id, p_info in self.pods.items():
            # (x, y, 是否被机器人搬运, SKU总数)
            pod_features.append([p_info['pos'], p_info['pos'][1], 1 if any(r['carrying_pod'] == p_id for r in self.robots.values()) else 0, sum(p_info['skus'].values())])

        workstation_features =
        for ws_id, ws_info in self.workstations.items():
            # (x, y, 订单队列长度)
            workstation_features.append([ws_info['pos'], ws_info['pos'][1], len(ws_info['order_queue'])])

        order_features =
        for o_id, o_info in self.orders.items():
            # (目标工作站ID, 优先级, 剩余截止时间, 总SKU需求, 已履行SKU总数)
            order_features.append([o_info['target_ws'], o_info['priority'], max(0, o_info['deadline'] - self.current_time), sum(o_info['skus_needed'].values()), sum(o_info['fulfilled_skus'].values())])

        # 邻接矩阵 (简化: 暂时全连接，实际应基于物理邻近或任务分配)
        # 在真实的GNN中，这会更复杂，例如基于物理邻近或任务分配。
        num_nodes_total = len(robot_features) + len(pod_features) + len(workstation_features) + len(order_features)
        adj_matrix = np.ones((num_nodes_total, num_nodes_total), dtype=np.float32) # 为简单起见，全连接

        # 目标回报 (Return-to-Go) (目前为占位符，应基于期望的未来奖励)
        # 在真实的Decision Transformer中，这会是任务或回合的目标回报。
        return_to_go = 1.0 # 占位符 [6]

        # 将所有观测转换为PyTorch张量并添加批次维度
        return {
            'grid_observation': torch.tensor(grid_obs, dtype=torch.float32).unsqueeze(0),
            'robot_features': torch.tensor(robot_features, dtype=torch.float32).unsqueeze(0),
            'pod_features': torch.tensor(pod_features, dtype=torch.float32).unsqueeze(0),
            'workstation_features': torch.tensor(workstation_features, dtype=torch.float32).unsqueeze(0),
            'order_features': torch.tensor(order_features, dtype=torch.float32).unsqueeze(0),
            'adjacency_matrix': torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0),
            'current_robot_id': torch.tensor([robot_id], dtype=torch.long),
            'decision_phase': torch.tensor([], dtype=torch.long),
            'return_to_go': torch.tensor([return_to_go], dtype=torch.float32)
        }

    def step(self, robot_id, action):
        """
        执行一个动作并更新环境状态。
        这是一个高度简化的step函数。在真实的事件驱动系统中，动作的执行可能需要多个时间步，
        并且机器人只有在完成当前子任务（例如，到达目的地）后才会触发新的决策事件。
        """
        robot_info = self.robots[robot_id]
        prev_pos = robot_info['pos']
        reward = 0
        done = False
        info = {}

        # 模拟动作执行
        if action < 5: # 基本移动 (上, 下, 左, 右, 停留)
            dx, dy = 0, 0
            if action == 0: dy = -1 # 上
            elif action == 1: dy = 1  # 下
            elif action == 2: dx = -1 # 左
            elif action == 3: dx = 1  # 右
            # action == 4: 停留 (dx=0, dy=0)

            new_x, new_y = prev_pos + dx, prev_pos[1] + dy

            # 检查边界和障碍物 (简化: 只有网格类型0可通行)
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height and self.grid[new_y, new_x] == 0:
                robot_info['pos'] = (new_x, new_y)
                if robot_info['carrying_pod'] is not None:
                    # 如果搬运货架，货架位置随机器人移动
                    self.pods[robot_info['carrying_pod']]['pos'] = (new_x, new_y)
                reward += self._calculate_reward('move', distance=1) # 移动惩罚 [7, 8]
            else:
                reward += self._calculate_reward('collision') # 碰撞惩罚 [7]
                info['collision'] = True
            
            # 简化阶段转换：如果机器人有任务，继续执行任务阶段；否则回到IDLE
            if robot_info['task']:
                if robot_info['task']['type'] == 'pickup':
                    robot_info['decision_phase'] = 'SHELF_SELECTION' # 还在前往货架的路上
                elif robot_info['task']['type'] == 'deliver':
                    robot_info['decision_phase'] = 'PATH_TO_WORKSTATION' # 还在前往工作站的路上
                elif robot_info['task']['type'] == 'return':
                    robot_info['decision_phase'] = 'PATH_TO_RETURN_LOCATION' # 还在返回货架的路上
            else:
                robot_info['decision_phase'] = 'IDLE' # 移动后空闲，等待新任务

        elif action < 5 + self.num_pods: # 选择一个货架
            pod_id = action - 5
            if robot_info['decision_phase'] == 'SHELF_SELECTION':
                # 假设机器人已到达货架位置（简化）
                # 实际应先导航到货架位置，然后执行拿起动作
                robot_info['carrying_pod'] = pod_id
                robot_info['task'] = {'type': 'deliver', 'ws_id': random.choice(list(self.workstations.keys())), 'pod_id': pod_id} # 简化：直接分配到随机工作站
                reward += self._calculate_reward('sku_picked') # 拣选SKU奖励 [1]
                robot_info['decision_phase'] = 'PATH_TO_WORKSTATION' # 准备运送到工作站
            else:
                reward += self._calculate_reward('invalid_action') # 错误阶段选择货架惩罚

        elif action < 5 + self.num_pods + self.num_workstations: # 选择一个工作站
            ws_id = action - (5 + self.num_pods)
            if robot_info['decision_phase'] == 'PATH_TO_WORKSTATION' and robot_info['carrying_pod'] is not None:
                # 假设机器人已到达工作站位置（简化）
                # 实际应先导航到工作站位置，然后执行放下动作
                order_id = self.workstations[ws_id]['order_queue'] if self.workstations[ws_id]['order_queue'] else None
                if order_id is not None:
                    order = self.orders[order_id]
                    pod_skus = self.pods[robot_info['carrying_pod']]['skus']
                    # 履行订单 (简化)
                    for sku_type, quantity_needed in order['skus_needed'].items():
                        if sku_type in pod_skus and pod_skus[sku_type] > 0:
                            fulfilled = min(quantity_needed - order['fulfilled_skus'].get(sku_type, 0), pod_skus[sku_type])
                            order['fulfilled_skus'][sku_type] = order['fulfilled_skus'].get(sku_type, 0) + fulfilled
                            pod_skus[sku_type] -= fulfilled
                    
                    if all(order['fulfilled_skus'].get(sku, 0) >= order['skus_needed'][sku] for sku in order['skus_needed']):
                        reward += self._calculate_reward('order_complete') # 订单完成奖励 [1]
                        self.workstations[ws_id]['order_queue'].popleft()
                        del self.orders[order_id]
                        info['order_completed'] = order_id
                        if not self.orders: # 所有订单都已履行
                            done = True
                
                robot_info['carrying_pod'] = None
                robot_info['task'] = {'type': 'return', 'pod_id': action - 5} # 简化：假设返回之前搬运的货架
                reward += self._calculate_reward('delivery_target_selected') # 运送目标选择奖励
                robot_info['decision_phase'] = 'PATH_TO_RETURN_LOCATION' # 准备返回货架
            else:
                reward += self._calculate_reward('invalid_action') # 错误阶段选择工作站惩罚

        # 模拟货架返回 (简化)
        if robot_info['task'] and robot_info['task']['type'] == 'return' and robot_info['pos'] == self.pods[robot_info['task']['pod_id']]['original_pos']:
            robot_info['task'] = None
            robot_info['decision_phase'] = 'IDLE' # 货架返回后空闲
            reward += self._calculate_reward('pod_returned') # 货架送回奖励

        # 检查机器人是否空闲并需要新任务
        if robot_info['decision_phase'] == 'IDLE' and not robot_info['task'] and not robot_info['carrying_pod']:
            if self.workstations['order_queue']: # 如果有待处理订单，分配新任务
                order_id = self.workstations['order_queue']
                target_pod_id = None
                for p_id, p_info in self.pods.items():
                    if any(sku in p_info['skus'] for sku in self.orders[order_id]['skus_needed']):
                        target_pod_id = p_id
                        break
                if target_pod_id is not None:
                    robot_info['task'] = {'type': 'pickup', 'pod_id': target_pod_id}
                    robot_info['decision_phase'] = 'SHELF_SELECTION'
                else:
                    robot_info['decision_phase'] = 'WAIT' # 没有找到合适的货架
            else:
                self._generate_new_order() # 如果没有待处理订单，生成新订单
                robot_info['decision_phase'] = 'WAIT' # 等待新订单被处理

        # 更新全局时间
        self.current_time += 1
        reward += self._calculate_reward('time_penalty') # 时间惩罚 [7]

        # 检查订单延迟
        for order_id, order_info in list(self.orders.items()):
            if self.current_time > order_info['deadline'] and not order_info.get('delayed', False):
                reward += self._calculate_reward('order_delay') # 订单延迟惩罚
                self.orders[order_id]['delayed'] = True # 标记为已延迟，避免重复惩罚

        next_observation = self._get_observation(robot_id, robot_info['decision_phase'])
        return next_observation, reward, done, info

    def _calculate_reward(self, event_type, **kwargs):
        """
        奖励函数组件 [1, 7, 8]
        """
        rewards = {
            'order_complete': 100.0,
            'sku_picked': 10.0,
            'pod_returned': 5.0,
            'time_penalty': -0.1,
            'move': -0.05 * kwargs.get('distance', 1),
            'collision': -50.0,
            'waiting': -0.5, # 处于'WAIT'阶段的惩罚
            'order_delay': -75.0,
            'invalid_action': -10.0,
            'pod_selected': 1.0,
            'delivery_target_selected': 1.0
        }
        return rewards.get(event_type, 0.0)

# --- 3. 状态表示和编码模块 ---

class GridMapEncoder(nn.Module):
    """
    使用CNN编码2D网格地图观测。
    输入: (Batch, Channels, Height, Width)
    输出: (Batch, Flattened_Features)
    参考:
    """
    def __init__(self, in_channels, height, width, embed_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 计算卷积层输出的尺寸
        self._out_height = height // 4
        self._out_width = width // 4
        self.fc = nn.Linear(64 * self._out_height * self._out_width, embed_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # 展平
        x = self.fc(x)
        return x

class GNN_Encoder(nn.Module):
    """
    使用简化的图神经网络编码动态实体关系。
    这是一个占位符。完整的实现将使用torch_geometric库。
    参考:
    """
    def __init__(self, robot_feature_dim, pod_feature_dim, ws_feature_dim, order_feature_dim, embed_dim):
        super().__init__()
        # 线性层将原始特征投影到公共嵌入维度
        self.robot_proj = nn.Linear(robot_feature_dim, embed_dim)
        self.pod_proj = nn.Linear(pod_feature_dim, embed_dim)
        self.ws_proj = nn.Linear(ws_feature_dim, embed_dim)
        self.order_proj = nn.Linear(order_feature_dim, embed_dim)

        # 简化的“聚合” - 在真实的GNN中，这将是消息传递
        self.aggregator = nn.Linear(embed_dim * 4, embed_dim) # 假设聚合特征的拼接

        # 如果安装了torch_geometric，你可以使用类似以下的代码:
        # import torch_geometric.nn as gnn_nn
        # self.conv1 = gnn_nn.GATConv(embed_dim, embed_dim)
        # self.conv2 = gnn_nn.GATConv(embed_dim, embed_dim)

    def forward(self, robot_features, pod_features, workstation_features, order_features, adjacency_matrix):
        # 投影特征
        robot_embeds = self.robot_proj(robot_features)
        pod_embeds = self.pod_proj(pod_features)
        ws_embeds = self.ws_proj(workstation_features)
        order_embeds = self.order_proj(order_features)

        # 简化的聚合 (例如，平均池化或拼接)
        # 在真实的GNN中，你会根据adjacency_matrix执行消息传递
        batch_size = robot_embeds.size(0)
        
        # 处理空实体列表，创建零张量
        avg_robot_embed = robot_embeds.mean(dim=1) if robot_embeds.size(1) > 0 else torch.zeros(batch_size, robot_embeds.size(2), device=robot_embeds.device)
        avg_pod_embed = pod_embeds.mean(dim=1) if pod_embeds.size(1) > 0 else torch.zeros(batch_size, pod_embeds.size(2), device=pod_embeds.device)
        avg_ws_embed = ws_embeds.mean(dim=1) if ws_embeds.size(1) > 0 else torch.zeros(batch_size, ws_embeds.size(2), device=pod_embeds.device)
        avg_order_embed = order_embeds.mean(dim=1) if order_embeds.size(1) > 0 else torch.zeros(batch_size, order_embeds.size(2), device=pod_embeds.device)

        # 拼接并聚合
        combined_embeds = torch.cat([avg_robot_embed, avg_pod_embed, avg_ws_embed, avg_order_embed], dim=-1)
        global_entity_embed = self.aggregator(combined_embeds)

        # 如果使用torch_geometric，这里会是GNN的实际前向传播逻辑
        # 例如:
        # graph_data = Data(x=torch.cat([robot_embeds.squeeze(0), pod_embeds.squeeze(0),...]), edge_index=adj_matrix.squeeze(0).nonzero().t().contiguous())
        # x = self.conv1(graph_data.x, graph_data.edge_index)
        # x = self.conv2(x, graph_data.edge_index)
        # 然后从x中提取相关的节点嵌入作为Transformer的输入。

        return global_entity_embed

class PositionalEncoding2D(nn.Module):
    """
    用于网格坐标的2D正弦位置编码。
    参考:
    """
    def __init__(self, embed_dim, max_x=GRID_WIDTH, max_y=GRID_HEIGHT):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_x = max_x
        self.max_y = max_y

        pe = torch.zeros(max_y, max_x, embed_dim)
        position_x = torch.arange(0, max_x, dtype=torch.float).unsqueeze(1)
        position_y = torch.arange(0, max_y, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # 将维度的一半用于X，一半用于Y，然后组合
        pe[:, :, 0::2] = torch.sin(position_x * div_term).unsqueeze(0).repeat(max_y, 1, 1)
        pe[:, :, 1::2] = torch.cos(position_x * div_term).unsqueeze(0).repeat(max_y, 1, 1)
        
        pe_y = torch.zeros(max_y, max_x, embed_dim)
        pe_y[:, :, 0::2] = torch.sin(position_y * div_term).unsqueeze(1).repeat(1, max_x, 1)
        pe_y[:, :, 1::2] = torch.cos(position_y * div_term).unsqueeze(1).repeat(1, max_x, 1)

        self.register_buffer('pe', pe + pe_y) # 组合X和Y的位置编码

    def forward(self, x_coords, y_coords):
        # x_coords, y_coords 是形状为 (batch_size, ) 的张量
        # 返回这些特定坐标的位置编码
        return self.pe[y_coords, x_coords]


class EntityEmbeddings(nn.Module):
    """
    为机器人ID和决策阶段提供可学习的嵌入。
    参考: [9, 10, 11, 12, 13, 14]
    """
    def __init__(self, num_robots, num_decision_phases, embed_dim):
        super().__init__()
        self.robot_id_embedding = nn.Embedding(num_robots, embed_dim)
        self.decision_phase_embedding = nn.Embedding(num_decision_phases, embed_dim)

    def forward(self, robot_id, decision_phase):
        robot_embed = self.robot_id_embedding(robot_id)
        phase_embed = self.decision_phase_embedding(decision_phase)
        return robot_embed, phase_embed

# --- 4. 主模型架构: RMFS_DecisionTransformer ---

class RMFS_DecisionTransformer(nn.Module):
    """
    用于RMFS机器人调度的Decision Transformer模型。
    输入: (Return-to-Go, 全局状态, 机器人特定状态, 动作, 机器人ID, 决策阶段) 序列。
    输出: 当前机器人去往每个地点的概率分布。
    参考:
    """
    def __init__(self, grid_width, grid_height, num_robots, num_pods, num_workstations, num_sku_types,
                 embed_dim, n_heads, n_layers, ff_dim, dropout, context_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.action_space_size = 5 + num_pods + num_workstations # 与ACTION_SPACE_SIZE常量匹配

        # 不同状态模态的编码器
        self.grid_encoder = GridMapEncoder(in_channels=4, height=grid_height, width=grid_width, embed_dim=embed_dim)
        
        # GNN_Encoder的虚拟维度 (如果未安装PyG)
        # 在实际设置中，这些维度应从实际实体特征大小中导出。
        robot_feature_dim_dummy = 4 # (x, y, carrying_pod_bool, is_current_robot_bool)
        pod_feature_dim_dummy = 4   # (x, y, is_carried_bool, total_sku_count)
        ws_feature_dim_dummy = 3    # (x, y, num_orders_in_queue)
        order_feature_dim_dummy = 5 # (target_ws_id, priority, deadline_remaining, total_skus_needed, total_skus_fulfilled)

        self.gnn_encoder = GNN_Encoder(robot_feature_dim_dummy, pod_feature_dim_dummy, ws_feature_dim_dummy, order_feature_dim_dummy, embed_dim)
        
        self.entity_embeddings = EntityEmbeddings(num_robots, NUM_DECISION_PHASES, embed_dim)
        self.pos_encoder_2d = PositionalEncoding2D(embed_dim, grid_width, grid_height)

        # 线性层用于嵌入目标回报和动作
        self.return_embed = nn.Linear(1, embed_dim) # 目标回报是标量
        self.action_embed = nn.Embedding(self.action_space_size, embed_dim) # 动作是离散的

        # 序列本身的位置嵌入 (时间步)
        # 每个时间步通常有多个token (例如: R, S, A)，所以序列长度是 context_length * tokens_per_step
        self.sequence_pos_embed = nn.Embedding(context_length * 3, embed_dim) # 3个token/时间步 (R, S, A)

        # Transformer编码器 (前向传播中将应用因果掩码)
        # 为简单起见，使用nn.TransformerEncoder，但完整的Decision Transformer通常使用
        # 仅解码器架构并带有因果掩码 [6]。
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 动作预测的输出头
        self.action_head = nn.Linear(embed_dim, self.action_space_size)

    def forward(self, observations, actions=None):
        """
        前向传播函数。
        observations 是一个字典，包含:
        'grid_observation': (Batch, Channels, Height, Width)
        'robot_features': (Batch, Num_Robots, Robot_Feature_Dim)
        'pod_features': (Batch, Num_Pods, Pod_Feature_Dim)
        'workstation_features': (Batch, Num_Workstations, WS_Feature_Dim)
        'order_features': (Batch, Num_Orders, Order_Feature_Dim)
        'adjacency_matrix': (Batch, Num_Nodes, Num_Nodes)
        'current_robot_id': (Batch, 1)
        'decision_phase': (Batch, 1)
        'return_to_go': (Batch, 1)
        actions (可选): 过去动作的序列 (Batch, Context_Length - 1)
        """
        batch_size = observations['grid_observation'].size(0)

        # 1. 编码不同状态模态
        grid_embed = self.grid_encoder(observations['grid_observation']) # (Batch, embed_dim)
        
        # GNN编码动态实体
        gnn_embed = self.gnn_encoder(
            observations['robot_features'],
            observations['pod_features'],
            observations['workstation_features'],
            observations['order_features'],
            observations['adjacency_matrix']
        ) # (Batch, embed_dim)

        # 组合全局状态嵌入
        global_state_embed = grid_embed + gnn_embed # (Batch, embed_dim)

        # 机器人ID和决策阶段嵌入
        robot_id_embed, decision_phase_embed = self.entity_embeddings(
            observations['current_robot_id'].squeeze(1), # 移除最后一个维度
            observations['decision_phase'].squeeze(1)    # 移除最后一个维度
        ) # 均为 (Batch, embed_dim)

        # 目标回报嵌入
        return_embed = self.return_embed(observations['return_to_go']) # (Batch, embed_dim)

        # 构建Transformer的输入序列
        # 格式: (R_t, S_t, A_t-1, R_t-1, S_t-1, A_t-2,...)
        # 为了简化，我们假设 `observations` 包含当前状态，并且 `actions` 包含序列中的 *先前* 动作。
        # 在一个完整的Decision Transformer中，你会传递一个 `context_length` 步的历史记录。
        
        # 对于单步预测，我们使用当前时间步的 (目标回报, 全局状态, 机器人ID, 决策阶段) 作为序列。
        # 堆叠当前token
        # 每个token的形状: (Batch, 1, Embed_Dim)
        current_tokens = torch.stack([return_embed, global_state_embed, robot_id_embed, decision_phase_embed], dim=1)
        
        # 添加序列位置编码 (针对这个短序列进行简化)
        # 在一个完整的DT中，这将是 `self.sequence_pos_embed(torch.arange(seq_len, device=tokens.device))`
        positions = torch.arange(current_tokens.size(1), device=current_tokens.device).unsqueeze(0).repeat(batch_size, 1)
        current_tokens = current_tokens + self.sequence_pos_embed(positions)

        # 创建Transformer的因果掩码
        # 这确保了动作的预测只依赖于前面的token。
        # 对于单次预测，如果输出头只关注最后一个token，掩码可能不是严格必要的，但对于序列建模至关重要。
        # 在这里，我们基于所有当前输入token预测动作，因此对于这种简化设置不需要因果掩码。
        # 如果我们以自回归方式预测动作序列，我们将使用:
        # causal_mask = nn.Transformer.generate_square_subsequent_mask(current_tokens.size(1), device=current_tokens.device)
        # transformer_output = self.transformer_encoder(current_tokens, mask=causal_mask)
        
        transformer_output = self.transformer_encoder(current_tokens)

        # Transformer的输出形状为 (Batch, Sequence_Length, Embed_Dim)
        # 我们希望基于最后一个token的表示 (或池化表示) 来预测动作。
        # 为了简单起见，我们取 `decision_phase_embed` token对应的表示，假设它是我们 `current_tokens` 序列中的最后一个。
        action_logits = self.action_head(transformer_output[:, -1, :]) # (Batch, Action_Space_Size)

        # 应用Softmax以获得概率分布 [15, 16]
        action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs

# --- 5. 训练循环 (高级伪代码) ---

# 用于存储经验