import argparse
import datetime
import json
import os
import pdb
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
from cprint import *
from gymnasium.envs.registration import register
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.tune import RunConfig, Tuner
from tqdm import tqdm

from algorithms.heuristic import HeuristicRLM
from algorithms.hgnn import HGNNModule
from algorithms.hierarchy import HierarchicalModule
from algorithms.transformer import TransformerModule
from algorithms.distance_aware_transformer import DistanceAwareTransformerModule
from rmfs_env import RMFSEnv


class MyCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.max_return_mean = -99999999
        self.min_distance = 99999999
        self.exp_name = None

    def on_environment_created(
        self,
        *,
        env_runner,
        metrics_logger,
        env,
        env_context,
        **kwargs,
    ):
        # print(env_context)
        # self.exp_name = env_context['exp_name']
        pass

    def on_episode_end(self, *, episode: SingleAgentEpisode, metrics_logger: MetricsLogger, **kwargs):
        # pdb.set_trace()
        makespan = episode.get_infos()[-1]['makespan']
        seed = episode.get_infos()[-1]['seed']
        metrics_logger.log_value('reward', episode.get_return(), reduce='mean', window=1000)
        metrics_logger.log_value('makespan', makespan, reduce='mean', window=1000)
        metrics_logger.log_value('total_steps', episode.env_steps(), reduce='sum')
        metrics_logger.log_value('total_episodes', 1, reduce='sum')
        # import datetime
        cprint.info(f"Episode done. {datetime.datetime.now().isoformat()}")
        total_steps = metrics_logger.peek('total_steps')
        total_episodes = metrics_logger.peek('total_episodes')
        print(f"Total steps={total_steps}, total episodes={total_episodes}.")
        print(f"R={episode.get_return()} makespan={makespan} seed={seed} steps={episode.env_steps()}")
        mean_reward = metrics_logger.peek('reward')
        mean_makespan = metrics_logger.peek('makespan')
        print(f'Mean reward: {mean_reward} Mean makespan: {mean_makespan}')

    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger: MetricsLogger,
        result: dict,
        **kwargs,
    ):
        # print(self.train_iterations)
        # pdb.set_trace()
        # self.train_iterations += 1
        pass

    def on_evaluate_start(
        self,
        *,
        algorithm,
        metrics_logger: MetricsLogger,
        **kwargs,
    ):
        print('Start evaluate.')

    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: MetricsLogger,
        evaluation_metrics: dict,
        **kwargs,
    ):
        exp_name = algorithm.config.exp_name
        return_mean = evaluation_metrics['env_runners']['episode_return_mean']
        mean_distance = -evaluation_metrics['env_runners']['reward']
        mean_makespan = evaluation_metrics['env_runners']['makespan']
        print(f'Evaluation return mean: {return_mean}, mean distance: {mean_distance}, mean makespan: {mean_makespan}.')
        if mean_distance < self.min_distance:
            self.min_distance = mean_distance
            print(f'Save checkpoint to: /home/tangyibang/rmfs/output/checkpoints/{exp_name}.')
            algorithm.save_checkpoint(f'/home/tangyibang/rmfs/output/checkpoints/{exp_name}')

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
    parser.add_argument('--extra_reward', action='store_true')
    # config
    parser.add_argument('--model', type=str, default='heuristic')
    parser.add_argument('--layout', type=str, default='layout')
    parser.add_argument('--exp', type=str)
    # eval config
    parser.add_argument('--eval_seed_l', type=int, default=0)
    parser.add_argument('--eval_seed_r', type=int, default=100)
    parser.add_argument('--evaluation_interval', type=int, default=1)
    parser.add_argument('--evaluation_num_env_runners', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str)
    # model config
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    # train
    parser.add_argument('--training_iteration', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--num_env_runners', type=int, default=1)
    parser.add_argument('--num_envs_per_env_runner', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument("--disable_gae", action="store_true")
    parser.add_argument("--disable_critic", action="store_true")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument('--train_batch_size_per_learner', type=int, default=2048)
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser.add_argument('--num_learners', type=int, default=1)
    # heuristic
    parser.add_argument('--eval_all', action="store_true")
    pick_choices = ['naive', 'max_satify', 'nearest']
    parser.add_argument('--pick_heuristic', choices=pick_choices, default='naive')
    deliver_choices = ['naive', 'max_satify', 'nearest']
    parser.add_argument('--deliver_heuristic', choices=deliver_choices, default='naive')
    return_choices = ['origin', 'nearest']
    parser.add_argument('--return_heuristic', choices=return_choices, default='origin')
    opts = parser.parse_args()

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.cuda.cudnn_enabled = False # 可能牺牲性能？
    torch.backends.cudnn.deterministic = True # 可能牺牲性能？

    ray.init(log_to_driver=opts.log_to_driver, local_mode=not opts.disable_local_mode)
    # log_to_driver=False 关闭 raylet 警告
    # local_mode=True 可以用 pdb 调试
    # CUDA_VISIBLE_DEVICES 必须要加
    exp_name = None
    if opts.exp:
        exp_name = opts.exp
    else:
        time_str = datetime.datetime.now().strftime('%d-%H%M')
        exp_name = f'{time_str}-{opts.model}'
    with open(f'/home/tangyibang/rmfs/layouts/{opts.layout}.json', 'r', encoding='utf-8') as f: # TODO
        json_data = json.load(f)
    env_config = {
        "print_env_info": not opts.disable_env_info,
        "seed_l": opts.seed_l,
        "seed_r": opts.seed_r,
        'extra_reward': opts.extra_reward,
        'layout': json_data,
        'stage': 'train',
        # 'exp_name': exp_name
    }
    eval_env_config = {
        "print_env_info": not opts.disable_env_info,
        "seed_l": opts.eval_seed_l,
        "seed_r": opts.eval_seed_r,
        'extra_reward': opts.extra_reward,
        'layout': json_data,
        'stage': 'evaluate',
        # 'exp_name': exp_name
    }
    env = RMFSEnv(env_config)
    resources = ray.cluster_resources()
    # pdb.set_trace()
    num_cpus = resources['CPU']
    num_gpus = resources['GPU']
    num_gpus_per = num_gpus / (opts.num_env_runners + opts.num_learners + 1) # TODO why +1?
    print(f'num_gpus_per: {num_gpus_per}')
    if opts.model in ['transformer', 'hierarchical']:
        if opts.num_env_runners > 1 or opts.num_learners > 1:
            assert opts.disable_local_mode
        if opts.model == 'transformer':
            module_class = TransformerModule
        else:
            module_class = HierarchicalModule
        # exit(0)
        config = (
            PPOConfig()
            .environment(
                env=RMFSEnv,
                env_config=env_config
            )
            .framework("torch")
            .env_runners(
                num_env_runners=opts.num_env_runners,
                num_envs_per_env_runner=opts.num_envs_per_env_runner,
                num_gpus_per_env_runner=num_gpus_per
                # Number of EnvRunner actors to create for parallel sampling. 
                # Setting this to 0 forces sampling to be done in the local EnvRunner (main process or the Algorithm's actor when using Tune).
            )
            .training(
                train_batch_size_per_learner=opts.train_batch_size_per_learner,
                num_epochs=opts.num_epochs,
                minibatch_size=opts.minibatch_size,
                gamma=opts.gamma,
                lambda_=opts.lambda_,
                use_gae=not opts.disable_gae,
                use_critic=not opts.disable_critic,
                lr=opts.lr,
                # lr=tune.grid_search([0.0001, 0.00001]), TODO tune 有问题
                # train_batch_size=opts.batch_size,
                # num_sgd_iter=opts.num_sgd_iter
            )
            # .resources(num_gpus=num_gpus)
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=module_class,
                    model_config={
                        'embed_dim': opts.embed_dim,
                        'nhead': opts.nhead,
                        'num_layers': opts.num_layers,
                        'layout': json_data
                    }
                ),
            )
            .learners(
                num_learners=opts.num_learners, 
                num_gpus_per_learner=num_gpus_per
                # Cannot set both `num_cpus_per_learner` > 1 and  `num_gpus_per_learner` > 0! 
                # Either set `num_cpus_per_learner` > 1 (and `num_gpus_per_learner`=0) OR 
                #   set `num_gpus_per_learner` > 0 (and leave `num_cpus_per_learner` at its default value of 1).
            )
            .callbacks(MyCallback)
            .evaluation(
                evaluation_interval=opts.evaluation_interval,
                evaluation_num_env_runners=opts.evaluation_num_env_runners,
                evaluation_duration_unit="episodes",
                evaluation_config=AlgorithmConfig.overrides(
                    env_config=eval_env_config
                ),
                evaluation_duration=opts.eval_seed_r - opts.eval_seed_l + 1,
            )
        )
        config.exp_name = exp_name
        tuner = Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=RunConfig(
                callbacks=[],
                stop={"training_iteration": opts.training_iteration},
                verbose=2
            ),
        )
        st = time.time()
        results = tuner.fit()
        ed = time.time()
        print(f'Training time: {ed - st}')
        best_result = results.get_best_result(
            metric='evaluation/env_runners/reward',
            mode='max'
        )
        metrics_df = best_result.metrics_dataframe
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        df = metrics_df[['training_iteration', 'evaluation/env_runners/episode_return_mean', 'evaluation/env_runners/reward', 'evaluation/env_runners/makespan']]
        print(df)
        df.to_csv(f'output/{exp_name}.csv', index=False)
        best_ckpt = best_result.checkpoint
        print(f'Save checkpoint to {Path(best_ckpt.path)}.')
        rl_module = RLModule.from_checkpoint(
            Path(best_ckpt.path)
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        ).to(torch.device("cuda:0"))
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
    else:
        # pdb.set_trace()
        ckpt_path = Path(opts.checkpoint_path)
        print(f'Eval. Load checkpoint from {ckpt_path}')
        rl_module = RLModule.from_checkpoint(
            ckpt_path
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        ).to(torch.device("cuda:0"))
        
    if opts.train_only:
        exit(0)
    
    def eval(env, rl_module):
        rewards = []
        distances = []
        makespans = []
        st = time.time()
        time_counter = 0
        for _ in tqdm(range(opts.eval_seed_l, opts.eval_seed_r + 1)):
            obs, _ = env.reset(seed=_)
            done = False
            total_reward = 0
            total_distance = 0
            
            with torch.no_grad():
                while not done:
                    model_outputs = rl_module.forward_inference({"obs": obs})
                    if isinstance(rl_module, HeuristicRLM):
                        action = model_outputs[Columns.ACTIONS][0]
                    else:
                        try:
                            action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
                        except:
                            action_dist_params = model_outputs["action_dist_inputs"][0].cpu().numpy()
                        # print(action_dist_params)
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
            # time_counter += info['time']
        ed = time.time()
        print(f'Eval time: {ed - st}')
        mean_reward = sum(rewards) / len(rewards)
        print(f'mean reward: {mean_reward}')
        print(rewards)
        mean_distance = sum(distances) / len(distances)
        print(f'mean distance: {mean_distance}')
        mean_makespan = sum(makespans) / len(makespans)
        print(f'mean makespan: {mean_makespan}')
        # print(f'Time: {time_counter}')
        return ed - st, mean_reward, mean_distance, mean_makespan

    if opts.eval_all:
        distances = []
        makespans = []
        for pick_method in pick_choices:
            for deliver_method in deliver_choices:
                for return_method in return_choices:
                    rl_module = HeuristicRLM(
                        observation_space=env.observation_space, 
                        action_space=env.observation_space,
                        model_config={
                            'pick': pick_method, # naive nearest
                            'deliver': deliver_method, # naive max_satify nearest
                            'return': return_method # origin nearest
                        },
                        env_config=env_config
                    )
                    print(pick_method, deliver_method, return_method)
                    print('')
                    _, _, distance, makespan = eval(env, rl_module)
                    distances.append(distance)
                    makespans.append(makespan)
        mean_distance = sum(distances) / len(distances)
        print(f'mean distance: {mean_distance}, min distance: {min(distances)}')
        mean_makespan = sum(makespans) / len(makespans)
        print(f'mean makespan: {mean_makespan}, min makespan: {min(makespans)}')
    else:
        eval(env, rl_module)