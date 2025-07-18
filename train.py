import argparse
import pdb
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
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

    ray.init(log_to_driver=opts.log_to_driver, local_mode=not opts.disable_local_mode)
    # log_to_driver=False 关闭 raylet 警告
    # local_mode=True 可以用 pdb 调试
    # CUDA_VISIBLE_DEVICES 必须要加
    env_config={
                "print_env_info": not opts.disable_env_info,
                "seed_l": opts.seed_l,
                "seed_r": opts.seed_r,
            }
    with open(f'{opts.layout}.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    env_config['shelves'] = json_data['shelves']
    env_config['workstations'] = json_data['workstations']
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
                num_env_runners=opts.num_env_runners
                # Number of EnvRunner actors to create for parallel sampling. 
                # Setting this to 0 forces sampling to be done in the local EnvRunner (main process or the Algorithm's actor when using Tune).
            )
            .training(
                train_batch_size=opts.batch_size,
                gamma=opts.gamma,
                lambda_=opts.lambda_,
                use_gae=not opts.disable_gae,
                use_critic=not opts.disable_critic,
                lr=opts.lr,
                num_sgd_iter=opts.num_sgd_iter
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
                verbose=2
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()
        metrics_df = best_result.metrics_dataframe
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        df = metrics_df[['training_iteration', 'env_runners/episode_return_mean', 'learners/default_policy/policy_loss', 'learners/default_policy/vf_loss']]
        print(metrics_df[['training_iteration', 'env_runners/episode_return_mean']])
        df.to_csv('train_metrics.csv', index=False)

        best_ckpt = best_result.checkpoint
        rl_module = RLModule.from_checkpoint(
            Path(best_ckpt.path)
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        )
        # pdb.set_trace()
    elif opts.model == 'heuristic':
        rl_module = HeuristicRLM(
            observation_space=env.observation_space, 
            action_space=env.observation_space,
            model_config={
                'pick': opts.pick_heuristic, # naive nearest
                'deliver': opts.deliver_heuristic, # naive max_satify nearest
                'return': opts.return_heuristic # origin nearest
            },
            env_config=env_config
        )
    T = 100
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
        
        with torch.no_grad():
            while not done:
                model_outputs = rl_module.forward_inference({"obs": obs})
                if isinstance(rl_module, HeuristicRLM):
                    action = model_outputs[Columns.ACTIONS][0]
                else:
                    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
                    # pdb.set_trace()
                    action = np.argmax(action_dist_params)
                # pdb.set_trace()
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                total_distance += info['distance']
                # env.render()

        # print(f'total distance = {total_distance}')
        rewards.append(total_reward)
        distances.append(total_distance)
        makespans.append(info['makespan'])
    ed = time.time()
    print(f'time: {ed - st}')
    mean_reward = sum(rewards) / len(rewards)
    print(f'mean reward: {mean_reward}')
    print(rewards)
    mean_distance = sum(distances) / len(distances)
    print(f'mean distance: {mean_distance}')
    mean_makespan = sum(makespans) / len(makespans)
    print(f'mean makespan: {mean_makespan}')