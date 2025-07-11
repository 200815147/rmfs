import argparse
import pdb
import time
from pathlib import Path

import numpy as np
import ray
import torch
from gymnasium.envs.registration import register
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune import RunConfig, Tuner

from algorithms.heuristic import HeuristicRLM
from algorithms.model import MyRLModule, TransformerModule
from rmfs_env import RMFSEnv


class EpisodeReturn(RLlibCallback):
    def __init__(self):
        super().__init__()
        # Keep some global state in between individual callback events.
        self.overall_sum_of_rewards = 0.0

    def on_episode_end(self, *, episode, **kwargs):
        self.overall_sum_of_rewards += episode.get_return()
        print(f"Episode done. R={episode.get_return()} Global SUM={self.overall_sum_of_rewards}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_env_info', action='store_true')
    parser.add_argument('--model', type=str, default='heuristic')
    parser.add_argument('--log_to_driver', action='store_true')
    parser.add_argument('--disable_local_mode', action='store_true')
    parser.add_argument('--training_iteration', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed_l', type=int, default=0)
    parser.add_argument('--seed_r', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lambda_', type=float, default=0.95)
    opts = parser.parse_args()

    ray.init(log_to_driver=opts.log_to_driver, local_mode=not opts.disable_local_mode)
    # log_to_driver=False 关闭 raylet 警告
    # local_mode=True 可以用 pdb 调试
    # CUDA_VISIBLE_DEVICES 必须要加
    env_config={
                "print_env_info": not opts.disable_env_info,
                "seed_l": opts.seed_l,
                "seed_r": opts.seed_r,
                "shelves": [
                    {"coord": (3, 0), "inventory": [(0, 10)]},
                    {"coord": (4, 0), "inventory": [(1, 10)]},
                    {"coord": (3, 1), "inventory": [(2, 10)]},
                    {"coord": (4, 1), "inventory": [(3, 10)]},
                    {"coord": (3, 3), "inventory": [(4, 10)]},
                    {"coord": (4, 3), "inventory": [(5, 10)]},
                    {"coord": (3, 4), "inventory": [(6, 10)]},
                    {"coord": (4, 4), "inventory": [(7, 10)]},
                ],
                "workstations": [(0, 0), (0, 4), (7, 0), (7, 4)],
                'instances': [
                    [
                        (1, 0, 2, 0),
                        (2, 1, 3, 1),
                        (3, 2, 2, 2),
                        (3, 3, 3, 1),
                        (4, 1, 2, 3),
                        (5, 4, 3, 1),
                        (6, 3, 3, 0),
                        (9, 5, 2, 3),
                        (10, 0, 3, 1),
                    ] # time sku_type num workstation
                ]
            }

    env = RMFSEnv(env_config)
    
    if opts.model in ['test', 'transformer']:
        if opts.model == 'test':
            module_class = MyRLModule
        else:
            module_class = TransformerModule
        config = (
            PPOConfig()
            .environment(
                env=RMFSEnv,
                env_config=env_config
            )
            .framework("torch")
            .env_runners(
                num_env_runners=2
                # Number of EnvRunner actors to create for parallel sampling. 
                # Setting this to 0 forces sampling to be done in the local EnvRunner (main process or the Algorithm's actor when using Tune).
            )
            .training(
                train_batch_size=opts.batch_size,
                gamma=opts.gamma,
                lambda_=opts.lambda_,
                use_gae=True,
                use_critic=True
            )
            .resources(num_gpus=2)
            .rl_module(
            # We need to explicitly specify here RLModule to use and
            # the catalog needed to build it.
                rl_module_spec=RLModuleSpec(
                    module_class=module_class,
                    model_config={}
                ),
            )
            .learners(
                num_gpus_per_learner=1
                # Cannot set both `num_cpus_per_learner` > 1 and  `num_gpus_per_learner` > 0! 
                # Either set `num_cpus_per_learner` > 1 (and `num_gpus_per_learner`=0) OR 
                #   set `num_gpus_per_learner` > 0 (and leave `num_cpus_per_learner` at its default value of 1).
                # This is due to issues with placement group fragmentation. 
                # See https://github.com/ray-project/ray/issues/35409 for more details.
            )
            .callbacks(EpisodeReturn)
        )
        tuner = Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=RunConfig(
                stop={"training_iteration": opts.training_iteration},
                verbose=1
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()
        metrics_df = best_result.metrics_dataframe
        print(metrics_df[['training_iteration', 'env_runners/episode_return_mean']])
        # pdb.set_trace()
        best_ckpt = best_result.checkpoint
        rl_module = RLModule.from_checkpoint(
            Path(best_ckpt.path)
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        )
    elif opts.model == 'heuristic':
        rl_module = HeuristicRLM(
            observation_space=env.observation_space, 
            action_space=env.observation_space,
            model_config={
                'pick': 'nearest', # naive nearest
                'deliver': 'nearest', # naive max_satify nearest
                'return': 'nearest' # origin nearest
            },
            env_config=env_config
        )
    T = 1
    rewards = []
    distances = []
    makespans = []
    st = time.time()
    for _ in range(T):
        print(f'Episode: {_}')
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_distance = 0
        # pdb.set_trace()
        # print("Rendering example episode:")
        with torch.no_grad():
            while not done:
                # Compute the next action from a batch (B=1) of observations.
                # obs_batch = torch.from_numpy(obs).unsqueeze(0)  # add batch B=1 dimension
                # pdb.set_trace()
                model_outputs = rl_module.forward_inference({"obs": obs})
                if isinstance(rl_module, HeuristicRLM):
                    action = model_outputs[Columns.ACTIONS][0]
                else:
                    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
                    action = np.argmax(action_dist_params)
                # pdb.set_trace()
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                total_distance += info['distance']
                # env.render()
        # print("Example episode finished.")
        print(f'total distance = {total_distance}')
        rewards.append(total_reward)
        distances.append(total_distance)
        makespans.append(info['makespan'])
    ed = time.time()
    print(f'time: {ed - st}')
    mean_reward = sum(rewards) / len(rewards)
    print(f'mean reward: {mean_reward}')
    mean_distance = sum(distances) / len(distances)
    print(f'mean distance: {mean_distance}')
    mean_makespan = sum(makespans) / len(makespans)
    print(f'mean makespan: {mean_makespan}')