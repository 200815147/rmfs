import pdb

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

from common_args import MapState, RobotState, env_attr
from utils import encode_action


class HeuristicRLM(RLModule):

    def __init__(self, observation_space, action_space, model_config, inference_only=True, **kwargs):
        super().__init__()
        self.config = kwargs['env_config']
        self.shelves = self.config['shelves']
        self.workstations = self.config['workstations']
        self.model_config = model_config

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = []
        obs = batch[Columns.OBS]
        robot_id = int(obs['global']['next_robot'])
        robots_obs = obs['robots']
        workstations_obs = obs['workstations']
        shelves_obs = obs['shelves']
        map_obs = obs['map']

        def distance(key1, idx1, key2, idx2):
            assert key1 in ['robots', 'workstations', 'shelves']
            assert key2 in ['robots', 'workstations', 'shelves']
            x1, y1 = obs[key1]['coord'][idx1]
            x2, y2 = obs[key2]['coord'][idx2]
            return abs(x1 - x2) + abs(y1 - y2)
        
        # fuck = robots_obs['state'][robot_id]
        # print(f'decide {robot_id} {fuck}')
        if robots_obs['state'][robot_id] == RobotState.PICK.value:
            shelf_id = -1
            if self.model_config['pick'] == 'naive':
                valid_id = 0
                # fuck = shelves_obs['state'][7]
                # print(f'debug state {fuck}')
                for i in range(env_attr.n_shelves):
                    if shelves_obs['state'][i] == env_attr.n_robots:
                        valid_id = i
                        inventory = shelves_obs['inventory'][i]
                        for j in range(env_attr.n_workstations):
                            demand = workstations_obs['demand'][j]
                            tmp = np.minimum(inventory, demand)
                            if np.sum(tmp) > 0:
                                shelf_id = i
                                # print(f'debug {robot_id} {i} {j}')
                                # print(tmp)
                                break
                    if shelf_id != -1:
                        break
                if shelf_id == -1:
                    shelf_id = valid_id
            elif self.model_config['pick'] == 'nearest':
                valid_id = 0
                min_dis = env_attr.inf
                for i in range(env_attr.n_shelves):
                    if shelves_obs['state'][i] == env_attr.n_robots:
                        valid_id = i
                        inventory = shelves_obs['inventory'][i]
                        dis = distance('robots', robot_id, 'shelves', i)
                        for j in range(env_attr.n_workstations):
                            demand = workstations_obs['demand'][j]
                            tmp = np.minimum(inventory, demand)
                            if np.sum(tmp) > 0:
                                if dis < min_dis:
                                    min_dis = dis
                                    shelf_id = i
                                    break
                    if shelf_id != -1:
                        break
                if shelf_id == -1:
                    shelf_id = valid_id
            else:
                raise NotImplementedError
            # print(f'debug {shelf_id}')
            ret.append((shelves_obs['coord'][shelf_id][0], shelves_obs['coord'][shelf_id][1]))
        elif robots_obs['state'][robot_id] == RobotState.DELIVER.value:
            if self.model_config['deliver'] == 'naive':
                shelf_id = robots_obs['shelf'][robot_id]
                inventory = shelves_obs['inventory'][shelf_id]
                workstations_id = 0
                for i in range(env_attr.n_workstations):
                    demand = workstations_obs['demand'][i]
                    tmp = np.minimum(inventory, demand)
                    if np.sum(tmp) > 0:
                        workstations_id = i
                        break
            elif self.model_config['deliver'] == 'max_satify':
                shelf_id = robots_obs['shelf'][robot_id]
                inventory = shelves_obs['inventory'][shelf_id]
                max_demand = -1
                workstations_id = 0
                for i in range(env_attr.n_workstations):
                    demand = workstations_obs['demand'][i]
                    satisfy_demand = np.sum(np.minimum(inventory, demand))
                    if satisfy_demand > max_demand:
                        max_demand = satisfy_demand
                        workstations_id = i
            elif self.model_config['deliver'] == 'nearest':
                shelf_id = robots_obs['shelf'][robot_id]
                inventory = shelves_obs['inventory'][shelf_id]
                min_dis = env_attr.inf
                workstations_id = 0
                for i in range(env_attr.n_workstations):
                    demand = workstations_obs['demand'][i]
                    satisfy_demand = np.sum(np.minimum(inventory, demand))
                    dis = distance('robots', robot_id, 'workstations', i)
                    if satisfy_demand > 0 and dis < min_dis:
                        min_dis = dis
                        workstations_id = i
            else:
                raise NotImplementedError
            # pdb.set_trace()
            # print(inventory)
            # print(workstations_obs['demand'][workstations_id])
            ret.append(self.workstations[workstations_id])
        elif robots_obs['state'][robot_id] == RobotState.RETURN.value:
            if self.model_config['return'] == 'origin':
                shelf_id = robots_obs['shelf'][robot_id]
                tx, ty = self.shelves[shelf_id]['coord']
            elif self.model_config['return'] == 'nearest':
                min_dis = env_attr.inf
                robot_x, robot_y = robots_obs['coord'][robot_id]
                for shelf_x in range(env_attr.x_max):
                    for shelf_y in range(env_attr.y_max):
                        if map_obs['id'][shelf_x][shelf_y] == MapState.SHELF_EMPTY.value:
                            dis = abs(shelf_x - robot_x) + abs(shelf_y - robot_y)
                            if dis < min_dis:
                                min_dis = dis
                                tx, ty = shelf_x, shelf_y
                # print(shelves_obs)
                # print(map_obs['id'])
            else:
                raise NotImplementedError
            # pdb.set_trace()
            # shelf_id = int(shelf_id)
            # print(f'shelf id {shelf_id}')
            ret.append((tx, ty))
        ret = [encode_action(x, y) for x, y in ret]
        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "HeuristicRLM is not trainable! Make sure you do NOT include it "
        )

    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]