import os
from abc import ABC, abstractmethod
from queue import PriorityQueue
import numpy as np
import pickle
import torch


class AgentBase(ABC):
    def __init__(self, CONFIG, CONFIG_ENV):
        super().__init__(CONFIG, CONFIG_ENV)

        self.config = CONFIG
        self.rng = np.random.default_rng(seed=CONFIG.SEED)
        self.device = CONFIG.DEVICE
        self.image_device = CONFIG.IMAGE_DEVICE

        # Mode
        self.eval = CONFIG.EVAL

        # Vectorized envs
        self.n_envs = CONFIG.NUM_CPUS
        self.action_dim = CONFIG_ENV.ACTION_DIM
        self.use_append = CONFIG_ENV.USE_APPEND
        self.max_train_steps = CONFIG_ENV.MAX_TRAIN_STEPS
        self.max_eval_steps = CONFIG_ENV.MAX_EVAL_STEPS
        self.env_step_cnt = [0 for _ in range(self.n_envs)]

        # Save
        self.out_folder = CONFIG.OUT_FOLDER
        self.save_top_k = CONFIG.SAVE_TOP_K
        self.pq_top_k = PriorityQueue()
        self.save_metric = CONFIG.SAVE_METRIC
        self.use_wandb = CONFIG.USE_WANDB
        # Figure folder
        # figure_folder = os.path.join(out_folder, 'figure')
        # os.makedirs(figure_folder, exist_ok=True)

        # Save loss and eval info, key is step number
        self.loss_record = {}
        self.eval_record = {}

        # Load tasks
        dataset = CONFIG_ENV.DATASET
        print("= Loading tasks from", dataset)
        with open(dataset, 'rb') as f:
            self.task_all = pickle.load(f)
        self.num_task = len(self.task_all)
        print(self.num_task, "tasks are loaded")

        # Mode
        if self.eval:
            self.set_eval_mode()
        else:
            self.set_train_mode()

        # Set starting step
        if CONFIG.CURRENT_STEP is None:
            self.cnt_step = 0
        else:
            self.cnt_step = CONFIG.CURRENT_STEP
            print("starting from {:d} steps".format(self.cnt_step))

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    # @abstractmethod
    # def finish_eval(self):
    #     raise NotImplementedError

    def set_train_mode(self):
        self.num_eval_episode = 0

        self.eval_mode = False
        self.max_env_step = self.max_train_steps

    def set_eval_mode(self):
        self.num_eval_episode = 0
        self.num_eval_success = 0  # for calculating expected success rate
        self.num_eval_safe = 0  # for calculating expected safety rate
        self.eval_reward_cumulative = [0 for _ in range(self.n_envs)
                                       ]  # for calculating cumulative reward
        self.eval_reward_best = [0 for _ in range(self.n_envs)]
        self.eval_reward_cumulative_all = 0
        self.eval_reward_best_all = 0
        self.env_step_cnt = [0 for _ in range(self.n_envs)]

        self.eval_mode = True
        self.max_env_step = self.max_eval_steps

    # === Venv ===
    def step(self, action):
        return self.venv.step(action)

    def reset_sim(self):
        self.venv.env_method('close_pb')

    def reset_env_all(self, task_ids=None, verbose=False):
        if task_ids is None:
            task_ids = self.rng.integers(low=0,
                                         high=self.num_task,
                                         size=(self.n_envs, ))
        tasks = [self.task_all[id] for id in task_ids]
        s = self.venv.reset(tasks)
        if verbose:
            for index in range(self.n_envs):
                print("<-- Reset environment {} with task {}:".format(
                    index, task_ids[index]))
        self.env_step_cnt = [0 for _ in range(self.n_envs)]
        return s, task_ids

    def reset_env(self, env_ind, task_id=None, verbose=False):
        if task_id is None:
            task_id = self.rng.integers(low=0, high=self.num_task)
        s = self.venv.reset_one(env_ind, self.task_all[task_id])
        if verbose:
            print("<-- Reset environment {} with task {}:".format(
                env_ind, task_id))
        self.env_step_cnt[env_ind] = 0
        return s, task_id

    # === Models ===
    def save(self, metric=None, force_save=False):
        assert metric is not None or force_save, \
            "should provide metric of force save"
        save_current = False
        if force_save:
            save_current = True
        elif self.pq_top_k.qsize() < self.save_top_k:
            self.pq_top_k.put((metric, self.cnt_step))
            save_current = True
        elif metric > self.pq_top_k.queue[0][0]:  # overwrite
            # Remove old one
            _, step_remove = self.pq_top_k.get()
            for module, module_folder in zip(self.module_all,
                                             self.module_folder_all):
                module.remove(int(step_remove), module_folder)
            self.pq_top_k.put((metric, self.cnt_step))
            save_current = True

        if save_current:
            print()
            print('Saving current model...')
            for module, module_folder in zip(self.module_all,
                                             self.module_folder_all):
                module.save(self.cnt_step, module_folder)
            print(self.pq_top_k.queue)

    # TODO
    def restore(self, step, logs_path, agent_type, actor_path=None):
        """Restore the weights of the neural network.

        Args:
            step (int): #updates trained.
            logs_path (str): the path of the directory, under this folder there
                should be critic/ and agent/ folders.
        """
        model_folder = path_c = os.path.join(logs_path, agent_type)
        path_c = os.path.join(model_folder, 'critic',
                              'critic-{}.pth'.format(step))
        if actor_path is not None:
            path_a = actor_path
        else:
            path_a = os.path.join(model_folder, 'actor',
                                  'actor-{}.pth'.format(step))
        self.learner.critic.load_state_dict(
            torch.load(path_c, map_location=self.device))
        self.learner.critic_target.load_state_dict(
            torch.load(path_c, map_location=self.device))
        self.learner.actor.load_state_dict(
            torch.load(path_a, map_location=self.device))
        print('  <= Restore {} with {} updates from {}.'.format(
            agent_type, step, model_folder))
