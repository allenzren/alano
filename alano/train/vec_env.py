import torch
import numpy as np
import gym
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)


def make_env(env_id, seed, rank, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, device, config_env, **kwargs):
    envs = [
        make_env(env_name, seed, i, **kwargs) for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device, config_env)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device, config_env):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.config_env = config_env
        self.n_envs = self.num_envs
        self.rng = np.random.default_rng(seed=config_env.VENV_SEED)

    def reset(self):
        if use_default:
            task_all = [{} for _ in range(self.n_envs)]
            task_ids = np.arange(self.n_envs)
        else:
            task_info = [(self.sample_task(return_id=True))
                         for _ in range(self.n_envs)]
            task_all = [info[0] for info in task_info]
            task_ids = [info[1] for info in task_info]
        args_all = [(random_init, state_init, task) for task in task_all]
        obs = self.venv.reset_arg(args_all)
        obs = torch.from_numpy(obs).to(self.device)
        if verbose:
            for index in range(self.n_envs):
                print("<-- Reset environment {}:".format(index))
                self.venv.env_method('report',
                                     print_obs_state=False,
                                     print_training=False,
                                     indices=[index])
        return obs, task_ids

    def reset_one(self,
                  index,
                  task=None,
                  use_default=False,
                  random_init=False,
                  state_init=None,
                  verbose=False):
        task_id = -1  #! dummy
        if task is None:
            if use_default:
                task = {}
            else:
                task, task_id = self.sample_task(return_id=True)
        obs = self.venv.env_method('reset',
                                   task=task,
                                   random_init=random_init,
                                   state_init=state_init,
                                   indices=[index])[0]
        obs = torch.from_numpy(obs).to(self.device)
        if verbose:
            print("<-- Reset environment {}:".format(index))
            self.venv.env_method('report',
                                 print_obs_state=False,
                                 print_training=False,
                                 indices=[index])
        return obs, task_id

    def get_obs(self, states):
        method_args_list = [(state, ) for state in states]
        observations = torch.FloatTensor(
            self.venv.env_method_arg('_get_obs',
                                     method_args_list=method_args_list,
                                     indices=range(self.n_envs)))
        return observations.to(self.device)

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
