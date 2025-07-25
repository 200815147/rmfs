import json
import pdb

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

from common_args import RobotState, env_attr


class HeuristicRLM(RLModule):

    def __init__(self, observation_space, action_space, model_config, inference_only=True, **kwargs):
        super().__init__()
        layout = kwargs['env_config']['layout']
        self.n_robots = len(layout['robots'])
        self.n_workstations = len(layout['workstations'])
        self.n_shelves = len(layout['shelves'])
        self.shelves = layout['shelves']
        self.workstations = layout['workstations']
        self.model_config = model_config

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = []
        obs = batch[Columns.OBS]
        robot_id = int(obs['global']['next_robot'])
        robots_obs = obs['robots']
        workstations_obs = obs['workstations']
        shelves_obs = obs['shelves']
        vacancies_obs = obs['vacancies']

        def distance(key1, idx1, key2, idx2):
            assert key1 in ['robots', 'workstations', 'vacancies']
            assert key2 in ['robots', 'workstations', 'vacancies']
            x1, y1 = obs[key1]['coord'][idx1]
            x2, y2 = obs[key2]['coord'][idx2]
            return abs(x1 - x2) + abs(y1 - y2)
        
        if robots_obs['state'][robot_id] == RobotState.PICK.value:
            vacancy_id = -1
            if self.model_config['pick'] == 'naive':
                valid_id = 0
                for i in range(self.n_shelves):
                    if vacancies_obs['state'][i] != self.n_shelves:
                        valid_id = i
                        shelf_id = vacancies_obs['state'][i]
                        inventory = shelves_obs['inventory'][shelf_id]
                        for j in range(self.n_workstations): # TODO 这里可以通过全局订单信息优化
                            demand = workstations_obs['demand'][j]
                            tmp = np.minimum(inventory, demand)
                            if np.any(tmp):
                                vacancy_id = i
                                break
                    if vacancy_id != -1:
                        break
                if vacancy_id == -1:
                    vacancy_id = valid_id
            elif self.model_config['pick'] == 'max_satify':
                valid_id = 0
                max_demand = 0
                for i in range(self.n_shelves):
                    if vacancies_obs['state'][i] != self.n_shelves:
                        valid_id = i
                        shelf_id = vacancies_obs['state'][i]
                        inventory = shelves_obs['inventory'][shelf_id]
                        for j in range(self.n_workstations):
                            demand = workstations_obs['demand'][j]
                            satisfy_demand = np.sum(np.minimum(inventory, demand))
                            if satisfy_demand > max_demand:
                                max_demand = satisfy_demand
                                vacancy_id = i
                    if vacancy_id != -1:
                        break
                if vacancy_id == -1:
                    vacancy_id = valid_id
            elif self.model_config['pick'] == 'nearest':
                valid_id = 0
                min_dis = env_attr.inf
                for i in range(self.n_shelves):
                    if vacancies_obs['state'][i] != self.n_shelves:
                        valid_id = i
                        shelf_id = vacancies_obs['state'][i]
                        inventory = shelves_obs['inventory'][shelf_id]
                        dis = distance('robots', robot_id, 'vacancies', i)
                        for j in range(self.n_workstations):
                            demand = workstations_obs['demand'][j]
                            tmp = np.minimum(inventory, demand)
                            if np.any(tmp):
                                if dis < min_dis:
                                    min_dis = dis
                                    vacancy_id = i
                                    break
                    if vacancy_id != -1:
                        break
                if vacancy_id == -1:
                    vacancy_id = valid_id
            else:
                raise NotImplementedError
            ret.append(vacancy_id)
        elif robots_obs['state'][robot_id] == RobotState.DELIVER.value:
            if self.model_config['deliver'] == 'naive':
                shelf_id = robots_obs['shelf'][robot_id]
                inventory = shelves_obs['inventory'][shelf_id]
                workstations_id = 0
                for i in range(self.n_workstations):
                    demand = workstations_obs['demand'][i]
                    tmp = np.minimum(inventory, demand)
                    if np.any(tmp):
                        workstations_id = i
                        break
            elif self.model_config['deliver'] == 'max_satify':
                shelf_id = robots_obs['shelf'][robot_id]
                inventory = shelves_obs['inventory'][shelf_id]
                max_demand = -1
                workstations_id = 0
                for i in range(self.n_workstations):
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
                for i in range(self.n_workstations):
                    demand = workstations_obs['demand'][i]
                    dis = distance('robots', robot_id, 'workstations', i)
                    if np.any(np.minimum(inventory, demand)) and dis < min_dis:
                        min_dis = dis
                        workstations_id = i
            else:
                raise NotImplementedError
            # pdb.set_trace()
            # print(inventory)
            # print(workstations_obs['demand'][workstations_id])
            ret.append(workstations_id + self.n_shelves)
        elif robots_obs['state'][robot_id] == RobotState.RETURN.value:
            if self.model_config['return'] == 'origin':
                vacancy_id = robots_obs['shelf'][robot_id]
            elif self.model_config['return'] == 'nearest':
                valid_id = 0
                min_dis = env_attr.inf
                for i in range(self.n_shelves):
                    if vacancies_obs['state'][i] == self.n_shelves:
                        valid_id = i
                        dis = distance('robots', robot_id, 'vacancies', i)
                        if dis < min_dis:
                            min_dis = dis
                            vacancy_id = i
                if vacancy_id == -1:
                    vacancy_id = valid_id
            else:
                raise NotImplementedError
            ret.append(vacancy_id)
        return {
            Columns.ACTIONS: np.array(ret)
        }

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