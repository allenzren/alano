import torch
import gym
from .subproc_vec_env import SubprocVecEnv


def make_env(env_id, seed, rank, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, cpu_offset, device,
                  config_env, vec_env_type, **kwargs):
    envs = [
        make_env(env_name, seed, i, **kwargs) for i in range(num_processes)
    ]
    envs = vec_env_type(envs, cpu_offset, device, config_env)
    return envs


class VecEnvBase(SubprocVecEnv):
    """
    Mostly for torch
    """
    def __init__(self, venv, cpu_offset, device, config_env):
        super(VecEnvBase, self).__init__(venv, cpu_offset)
        self.device = device
        self.config_env = config_env


    def reset(self, tasks):
        args_all = [(task, ) for task in tasks]
        obs = self.reset_arg(args_all)
        return torch.from_numpy(obs).to(self.device)


    def reset_one(self, index, task):
        obs = self.env_method('reset', task=task, indices=[index])[0]
        return torch.from_numpy(obs).to(self.device)


    # Overrides
    def step_async(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        super().step_async(actions)


    # Overrides
    def step_wait(self):
        obs, reward, done, info = super().step_wait()
        obs = torch.from_numpy(obs).to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


    def get_obs(self, states):
        method_args_list = [(state, ) for state in states]
        obs = torch.FloatTensor(
            self.env_method_arg('_get_obs',
                                method_args_list=method_args_list,
                                indices=range(self.n_envs)))
        return obs.to(self.device)
