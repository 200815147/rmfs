import heapq
import pdb
from itertools import cycle

import gymnasium as gym
import numpy as np
import torch
from cprint import *
from gymnasium import spaces
import json
from common_args import LOGLEVEL, MapState, RobotState, env_attr


class RMFSEnv(gym.Env):
    """
    Robotic Mobile Fulfillment System 环境。
    事件驱动：每当机器人空闲时触发决策。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        layout = config['layout']
        with open(f'/home/tangyibang/rmfs/{layout}.json', 'r', encoding='utf-8') as f: # TODO
            json_data = json.load(f)
        self.config = json_data
        # self.config['shelves'] = json_data['shelves']
        # self.config['workstations'] = json_data['workstations']
        # self.config['robots'] = json_data['robots']
        env_attr.n_robots = len(json_data['robots'])
        env_attr.n_workstations = len(json_data['workstations'])
        env_attr.n_shelves = len(json_data['shelves'])
        env_attr.n_sku_types = json_data['n_sku_types']
        env_attr.x_max = json_data['x_max']
        env_attr.y_max = json_data['y_max']
        
        self.print_env_info = config['print_env_info']
        self.extra_reward = config['extra_reward']
        self.seed_pool = cycle(range(config["seed_l"], config["seed_r"] + 1))
        self.seed_iter = iter(self.seed_pool)
        self.instance_idx = -1

        # 定义空间
        self.observation_space = spaces.Dict({
            'map': spaces.Dict({
                'id': spaces.MultiDiscrete([[env_attr.x_max * env_attr.y_max + 1] * env_attr.y_max] * env_attr.x_max) # 每个格子 id
            }),
            'robots': spaces.Dict({
                'state': spaces.MultiDiscrete([env_attr.n_robot_state] * env_attr.n_robots), # 每个机器人状态 定义见 env_attr
                "coord": spaces.Box(0, max(env_attr.x_max, env_attr.y_max), shape=(env_attr.n_robots, 2), dtype=np.int32), # 当前坐标
                "target": spaces.Box(0, max(env_attr.x_max, env_attr.y_max), shape=(env_attr.n_robots, 2), dtype=np.int32), # 目标
                'shelf': spaces.MultiDiscrete([env_attr.n_shelves + 1] * env_attr.n_robots) # 运送货架 +1 表示没送
            }),
            'workstations': spaces.Dict({
                "coord": spaces.Box(0, max(env_attr.x_max, env_attr.y_max), shape=(env_attr.n_workstations, 2), dtype=np.int32), # 坐标
                'demand': spaces.Box(0, env_attr.inf, shape=(env_attr.n_workstations, env_attr.n_sku_types), dtype=np.int32), # 订单需求
                'distance': spaces.Box(0, env_attr.x_max + env_attr.y_max, shape=(env_attr.n_workstations,), dtype=np.int32) # 与决策机器人距离
            }),
            'shelves': spaces.Dict({
                'state': spaces.MultiDiscrete([env_attr.n_robots + env_attr.n_shelves] * env_attr.n_shelves), # 货架状态 表示现在被哪个机器人运送或者在哪个空地
                'inventory': spaces.Box(0, env_attr.inf, shape=(env_attr.n_shelves, env_attr.n_sku_types), dtype=np.int32) # 库存
            }),
            'vacancies': spaces.Dict({
                'state': spaces.MultiDiscrete([env_attr.n_shelves + 1] * env_attr.n_shelves), # 货架状态 表示现在放了哪个货架
                "coord": spaces.Box(0, max(env_attr.x_max, env_attr.y_max), shape=(env_attr.n_shelves, 2), dtype=np.int32), # 坐标
                'distance': spaces.Box(0, env_attr.x_max + env_attr.y_max, shape=(env_attr.n_shelves,), dtype=np.int32) # 与决策机器人距离
            }),
            'global': spaces.Dict({
                'next_robot': spaces.MultiDiscrete([env_attr.n_robots]), # 下一个需要决策的机器人 如果没了应该也没问题 一轮结束了
                'action_mask': spaces.MultiBinary(env_attr.n_shelves + env_attr.n_workstations)
            })
        })
        self.action_space = spaces.Discrete(env_attr.n_shelves + env_attr.n_workstations)

        self.robot_offset = 1 # 0 空地
        self.workstation_offset = self.robot_offset + env_attr.n_robots
        self.vacancy_offset = self.workstation_offset + env_attr.n_workstations

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = next(self.seed_iter)
        self.seed(seed)
        self.instance_idx += 1
        self.log(f'Reset environment. idx={self.instance_idx}, seed={seed}')
        self.current_frame = 0
        self.current_time = 0
        self.makespan = 0
        self.event_queue = []
        self.map_id = np.zeros((env_attr.x_max, env_attr.y_max), dtype=np.int32)
        
        # 初始化机器人
        self.robots = [{"coord": (0, 0), "state": RobotState.PICK, "target": (0, 0), "shelf": env_attr.n_shelves} for r in range(env_attr.n_robots)] # TODO 机器人初始位置
        self.end_robots = 0
        robots = self.config.get('robots', None)
        assert len(robots) == env_attr.n_robots, 'n_robots wrong.'
        for i in range(env_attr.n_robots):
            heapq.heappush(self.event_queue, (0, 1, i))
            self.robots[i]['coord'] = tuple(robots[i])
            self.map_id[self.robots[i]['coord'][0]][self.robots[i]['coord'][1]] = i + self.robot_offset

        total_skus = np.zeros((env_attr.n_sku_types), dtype=np.int32)
        shelves = self.config.get('shelves', None)
        assert len(shelves) == env_attr.n_shelves, 'n_shelves wrong.'
        self.shelves = []
        self.vacancies = []
        for i in range(env_attr.n_shelves):
            inventory = np.zeros((env_attr.n_sku_types), dtype=np.int32)
            for k, v in shelves[i]['inventory']:
                inventory[k] = v
            total_skus += inventory
            self.shelves.append({'inventory': inventory, 'state': env_attr.n_robots + i})
            self.vacancies.append({"coord": shelves[i]['coord'], 'state': i})
            self.map_id[shelves[i]['coord'][0]][shelves[i]['coord'][1]] = i + self.vacancy_offset
        
        workstations = self.config.get('workstations', None)
        assert len(workstations) == env_attr.n_workstations, 'n_workstations wrong.'
        self.workstations = []
        self.future_demand = np.zeros((env_attr.n_sku_types), dtype=np.int32)
        for i in range(env_attr.n_workstations):
            demand = np.zeros((env_attr.n_sku_types), dtype=np.int32)
            self.workstations.append({"coord": workstations[i], 'demand': demand})
            self.map_id[workstations[i][0]][workstations[i][1]] = i + self.workstation_offset
        
        # time sku_type num workstation
        self.orders = []
        num_l = self.config.get('order_num_l', 1)
        num_r = self.config.get('order_num_r', 2)
        total_orders_num = self.np_random.integers(num_l, num_r)
        time_l = self.config.get('order_time_l', 0)
        time_r = self.config.get('order_time_r', 1)
        for i in range(total_orders_num):
            appear_time = self.np_random.integers(time_l, time_r)
            nonzero_idx = np.nonzero(total_skus)[0]
            if len(nonzero_idx) == 0:
                break
            sku_type = self.np_random.choice(nonzero_idx)
            # sku_type = 0 # debug
            num = self.np_random.integers(1, total_skus[sku_type] + 1)
            # num = 1 # debug
            total_skus[sku_type] -= num
            workstation = self.np_random.integers(0, env_attr.n_workstations)
            self.orders.append((appear_time, sku_type, num, workstation))
            self.log(f'Generate order: time={appear_time} sku={sku_type} num={num} assigned to {workstation}', LOGLEVEL.INFO)
        self.orders_cnt = 0
        for i, (time, _, _, _) in enumerate(self.orders):
            heapq.heappush(self.event_queue, (time, 0, i))
        
        return self._get_obs(), {}
    
    def decode_action(self, action):
        if action < env_attr.n_shelves:
            return self.vacancies[action]['coord']
        else:
            return self.workstations[action - env_attr.n_shelves]['coord']
    
    def step(self, action):
        time, priority, robot_id = heapq.heappop(self.event_queue)
        assert priority == 1, 'event must be robot.' # 保证是机器人事件
        self.current_time = time
        self.current_frame += 1
        assert self.current_frame < env_attr.max_frame, 'endless.'
        self.log(f"frame={self.current_frame}, time={self.current_time}", LOGLEVEL.INFO)
        robot = self.robots[robot_id]
        target_x, target_y = self.decode_action(action)
        reward = 0
        if robot['state'] == RobotState.PICK:
            vacancy_id = self.map_id[target_x][target_y] - self.vacancy_offset
            shelf_id = self.vacancies[vacancy_id]['state']
            assert 0 <= vacancy_id < env_attr.n_shelves, f'Robot {robot_id} target invalid (stage: 0).'
            assert shelf_id != env_attr.n_shelves, f'No shelf on ({target_x}, {target_y}) (stage: 0).'
            shelf = self.shelves[shelf_id]
            vacancy = self.vacancies[vacancy_id]
            shelf['state'] = robot_id
            robot['state'] = RobotState.DELIVER
            robot['shelf'] = shelf_id
            vacancy['state'] = env_attr.n_shelves
            self.log(f"Robot {robot_id} pick shelf {shelf_id}.", LOGLEVEL.INFO)
            if self.extra_reward:
                reward += env_attr.pick_reward

        elif robot['state'] == RobotState.DELIVER:
            workstation_id = self.map_id[target_x][target_y] - self.workstation_offset
            assert 0 <= workstation_id < env_attr.n_workstations, f'Robot {robot_id} target invalid (stage: 1).'
            shelf_id = robot['shelf']
            shelf = self.shelves[shelf_id]
            workstation = self.workstations[workstation_id]
            take_sku = np.minimum(shelf['inventory'], workstation['demand'])
            if np.sum(take_sku) == 0:
                self.log(f'Shelf {shelf_id} is useless for workstation {workstation_id}.', LOGLEVEL.WARN)
            shelf['inventory'] -= take_sku
            workstation['demand'] -= take_sku
            self.future_demand -= take_sku
            robot['state'] = RobotState.RETURN
            self.log(f"Robot {robot_id} deliver shelf {shelf_id} to workstation {workstation_id}.", LOGLEVEL.INFO)
            if self.extra_reward:
                reward += env_attr.deliver_reward

        elif robot['state'] == RobotState.RETURN:
            shelf_id = robot['shelf']
            vacancy_id = self.map_id[target_x][target_y] - self.vacancy_offset
            assert 0 <= shelf_id < env_attr.n_shelves, f'Robot {robot_id} carry invalid shelf (stage: 2).'
            assert 0 <= vacancy_id < env_attr.n_shelves, f'Robot {robot_id} target invalid (stage: 2).'
            shelf = self.shelves[shelf_id]
            vacancy = self.vacancies[vacancy_id]
            shelf['state'] = env_attr.n_robots + vacancy_id
            robot['state'] = RobotState.PICK
            vacancy['state'] = shelf_id
            self.log(f"Robot {robot_id} return shelf {shelf_id} to ({target_x}, {target_y}).", LOGLEVEL.INFO)
            if self.extra_reward:
                reward += env_attr.return_reward

        elif robot['state'] == RobotState.END:
            pass
        else:
            raise ValueError(f'Robot {robot_id} state error.')

        # 计算时间
        robot_x, robot_y = robot["coord"]
        dis = abs(robot_x - target_x) + abs(robot_y - target_y)

        finish_time = self.current_time + dis
        robot["coord"] = (target_x, target_y)
        if robot['state'] == RobotState.PICK and not np.any(self.get_workstations_obs()['demand']) and self.orders_cnt == len(self.orders):
            robot['sate'] = RobotState.END
            self.end_robots += 1
            self.makespan = max(self.makespan, finish_time)
        else:
            heapq.heappush(self.event_queue, (finish_time, 1, robot_id))

        reward -= dis

        done = self.end_robots == env_attr.n_robots
        obs = self._get_obs()
        info = {'distance': dis}
        if done:
            info['makespan'] = self.makespan
        
        self.log(f"Robot {robot_id} take action ({robot_x}, {robot_y}) -> ({target_x}, {target_y}), reward={reward}", LOGLEVEL.INFO)

        # if self.current_frame % 5 == 0:
        #     self.log('workstation')
        #     for i in range(env_attr.n_workstations):
        #         self.log(self.workstations[i]['demand'])
        #     self.log('shelves')
        #     for i in range(env_attr.n_shelves):
        #         self.log(f'shelf {i}')
        #         self.log(self.shelves[i]['coord'])
        #         self.log(self.shelves[i]['inventory'])
        return obs, reward, done, False, info
    
    def get_next_robot(self):
        while len(self.event_queue) != 0 and self.event_queue[0][1] == 0:
            _, _, order_id = heapq.heappop(self.event_queue)
            _, sku_type, sku_num, workstations_id = self.orders[order_id]
            self.orders_cnt += 1
            self.future_demand[sku_type] += sku_num
            self.log(f"Order {order_id} assigned to workstation {workstations_id}, add sku ({sku_type}) * {sku_num}", LOGLEVEL.INFO)
            self.workstations[workstations_id]['demand'][sku_type] += sku_num
        return self.event_queue[0][2] if len(self.event_queue) != 0 else 0
    
    def get_workstations_obs(self):
        workstations_obs = {
            'coord': np.array([workstation['coord'] for workstation in self.workstations], dtype=np.int32),
            'demand': np.array([workstation['demand'] for workstation in self.workstations], dtype=np.int32)
        }
        return workstations_obs

    def _get_obs(self):
        next_robot = self.get_next_robot()
        robot_x, robot_y = self.robots[next_robot]['coord']
        robots_obs = {
            'state': np.array([robot['state'].value for robot in self.robots], dtype=np.int32),
            'coord': np.array([robot['coord'] for robot in self.robots], dtype=np.int32),
            'target': np.array([robot['target'] for robot in self.robots], dtype=np.int32),
            'shelf': np.array([robot['shelf'] for robot in self.robots], dtype=np.int32)
        }
        workstations_obs = self.get_workstations_obs()
        workstations_obs['distance'] = np.abs(robot_x - workstations_obs['coord'][:, 0]) + np.abs(robot_y - workstations_obs['coord'][:, 1])
        shelves_obs = {
            'state': np.array([shelf['state'] for shelf in self.shelves], dtype=np.int32),
            'inventory': np.array([shelf['inventory'] for shelf in self.shelves], dtype=np.int32)
        }
        vacancies_obs = {
            'state': np.array([vacancy['state'] for vacancy in self.vacancies], dtype=np.int32),
            'coord': np.array([vacancy['coord'] for vacancy in self.vacancies], dtype=np.int32)
        }
        vacancies_obs['distance'] = np.abs(robot_x - vacancies_obs['coord'][:, 0]) + np.abs(robot_y - vacancies_obs['coord'][:, 1])
        action_mask = np.zeros((env_attr.n_shelves + env_attr.n_workstations), dtype=np.int8)
        # TODO mask 向量化
        if self.robots[next_robot]['state'] == RobotState.PICK: # 只能取未取货架
            valid_vacancy_id = None
            for i, vacancy in enumerate(self.vacancies): 
                if vacancy['state'] != env_attr.n_shelves:
                    valid_vacancy_id = i
                    inventory = self.shelves[vacancy['state']]['inventory']
                    if np.any(np.minimum(inventory, self.future_demand)):
                        action_mask[i] = 1
            if not np.any(action_mask):
                self.log(f'No shelf needed. Go to vacancy {valid_vacancy_id}.', log_level=LOGLEVEL.WARN)
                action_mask[valid_vacancy_id] = 1
        elif self.robots[next_robot]['state'] == RobotState.DELIVER: # 只能去工作站
            shelf_id = self.robots[next_robot]['shelf']
            inventory = self.shelves[shelf_id]['inventory']
            for i, workstation in enumerate(self.workstations):
                if np.any(np.minimum(workstation['demand'], inventory)):
                    action_mask[env_attr.n_shelves + i] = 1
            if not np.any(action_mask):
                self.log('No workstation needed. Go to workstation 0.', log_level=LOGLEVEL.WARN)
                action_mask[env_attr.n_shelves] = 1
        elif self.robots[next_robot]['state'] == RobotState.RETURN: # 只能去货架空地
            action_mask = (vacancies_obs['state'] == env_attr.n_shelves).astype(np.int8)
            action_mask = np.concatenate((action_mask, np.zeros((env_attr.n_workstations), dtype=np.int8)))
        obs = {
            'map': {
                'id': self.map_id
            },
            'robots': robots_obs,
            'workstations': workstations_obs,
            'shelves': shelves_obs,
            'vacancies': vacancies_obs,
            'global': {
                'next_robot': np.array([next_robot], dtype=np.int32),
                'action_mask': action_mask
            }
        }
        # pdb.set_trace()
        for key in obs.keys():
            assert self.observation_space[key].contains(obs[key]), f"{key} obs invalid!"
        
        return obs

    def log(self, msg, log_level=LOGLEVEL.INFO):
        if self.print_env_info:
            if log_level == LOGLEVEL.INFO:
                cprint.info(msg)
            elif log_level == LOGLEVEL.WARN:
                cprint.warn(msg)