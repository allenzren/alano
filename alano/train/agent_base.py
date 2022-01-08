# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from abc import ABC, abstractmethod
from collections import namedtuple
from queue import PriorityQueue
import torch
import os
import numpy as np

from .replay_memory import ReplayMemory

Transition= namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])


class AgentBase(ABC):
    def __init__(self, CONFIG, CONFIG_ENV, CONFIG_UPDATE):
        super(AgentBase, self).__init__()
        self.config = CONFIG

        # Device
        self.device = CONFIG.DEVICE
        self.image_device = CONFIG.IMAGE_DEVICE

        # Vectorized envs
        self.n_envs = CONFIG.NUM_CPUS
        self.action_dim = CONFIG_ENV.ACTION_DIM
        self.use_append = CONFIG_ENV.USE_APPEND
        self.max_train_steps = CONFIG_ENV.MAX_TRAIN_STEPS
        self.max_eval_steps = CONFIG_ENV.MAX_EVAL_STEPS
        self.env_step_cnt = [0 for _ in range(self.n_envs)]

        # Sampling
        self.max_sample_steps = CONFIG.MAX_SAMPLE_STEPS
        self.opt_freq = CONFIG.OPTIMIZE_FREQ
        self.num_update_per_opt = CONFIG.UPDATE_PER_OPT
        self.check_opt_freq = CONFIG.CHECK_OPT_FREQ
        self.min_step_b4_opt = CONFIG.MIN_STEPS_B4_OPT
        self.num_episode_per_eval = CONFIG.NUM_EVAL_EPISODE

        # Training
        self.batch_size = CONFIG_UPDATE.BATCH_SIZE
        self.update_period = CONFIG_UPDATE.UPDATE_PERIOD

        # memory
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY, CONFIG.SEED)
        self.rng = np.random.default_rng(seed=CONFIG.SEED)

        # saving models
        self.out_folder = CONFIG.OUT_FOLDER
        self.save_top_k = CONFIG.SAVE_TOP_K
        self.pq_top_k = PriorityQueue()
        self.save_metric = CONFIG.SAVE_METRIC
        self.use_wandb = CONFIG.USE_WANDB

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    def set_train_mode(self):
        self.eval_mode = False
        self.num_eval_episode = 0
        self.venv.env_method('set_train_mode')
        s, _ = self.venv.reset()
        return s
    
    def set_eval_mode(self):
        self.venv.env_method('set_eval_mode')
        self.num_eval_episode = 0
        self.num_eval_success = 0
        self.eval_mode = True
        s, _ = self.venv.reset()
        return s

    # === Replay ===

    def sample_batch(self, batch_size=None, recent_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        transitions, _ = self.memory.sample(batch_size, recent_size)
        batch = Transition(*zip(*transitions))
        return batch

    def store_transition(self, *args):
        self.memory.update(Transition(*args))

    def unpack_batch(self,
                     batch,
                     get_rnn_hidden=False):
        non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)),
                                      dtype=torch.bool).view(-1).to(
                                          self.device)
        non_final_state_nxt = torch.cat([
            s for done, s in zip(batch.done, batch.s_) if not done
        ]).to(self.device)
        state = torch.cat(batch.s).to(self.device)
        reward = torch.FloatTensor(batch.r).to(self.device)
        action = torch.cat(batch.a).to(self.device)

        # Debug
        # import matplotlib.pyplot as plt
        # for ind in range(10):
            # f, axarr = plt.subplots(2, 4)
            # for i_ind in range(4):
            #     axarr[0, i_ind].imshow(
            #         np.transpose(
            #             state[ind][(i_ind * 3):((i_ind+1) * 3)].cpu().numpy(),
            #             [1, 2, 0]
            #         )
            #     )
            #     axarr[1, i_ind].imshow(
            #         np.transpose(
            #             non_final_state_nxt[ind][(i_ind
            #                                       * 3):((i_ind+1)
            #                                             * 3)].cpu().numpy(),
            #             [1, 2, 0]
            #         )
            #     )
            # f, axarr = plt.subplots(2, 2)
            # axarr[0, 0].imshow(
            #     np.transpose(
            #         state[ind][1:].cpu().numpy(),
            #         [1, 2, 0]
            #     )
            # )
            # axarr[0, 1].imshow(state[ind][0].cpu().numpy())     
            # axarr[1, 0].imshow(
            #     np.transpose(
            #         non_final_state_nxt[ind][1:].cpu().numpy(),
            #         [1, 2, 0]
            #     )
            # )
            # axarr[1, 1].imshow(non_final_state_nxt[ind][0].cpu().numpy())     
            # plt.show()

        # Optional
        append = None
        non_final_append_nxt = None
        if self.use_append:
            append = torch.cat([info['append']
                                for info in batch.info]).to(self.device)
            non_final_append_nxt = torch.cat([
                info['append_nxt'] for info in batch.info
            ]).to(self.device)[non_final_mask]
        hn = None
        cn = None
        if get_rnn_hidden:
            hn = batch.info[0]['hn'].to(self.device)  # only get initial, 1x
            cn = batch.info[0]['hn'].to(self.device)
        return (non_final_mask, non_final_state_nxt, state, action, reward,
                append, non_final_append_nxt, hn, cn)

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

    def restore(self, step, logs_path, agent_type, actor_path=None):
        """Restore the weights of the neural network.

        Args:
            step (int): #updates trained.
            logs_path (str): the path of the directory, under this folder there
                should be critic/ and agent/ folders.
            agent_type (str): performance or backup.
        """
        model_folder = path_c = os.path.join(logs_path, agent_type)
        path_c = os.path.join(model_folder, 'critic',
                              'critic-{}.pth'.format(step))
        if actor_path is not None:
            path_a = actor_path
        else:
            path_a = os.path.join(model_folder, 'actor',
                                  'actor-{}.pth'.format(step))
        if agent_type == 'backup':
            self.backup.critic.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.backup.critic.to(self.device)
            self.backup.critic_target.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.backup.critic_target.to(self.device)
            self.backup.actor.load_state_dict(
                torch.load(path_a, map_location=self.device))
            self.backup.actor.to(self.device)
        elif agent_type == 'performance':
            self.performance.critic.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.performance.critic.to(self.device)
            self.performance.critic_target.load_state_dict(
                torch.load(path_c, map_location=self.device))
            self.performance.critic_target.to(self.device)
            self.performance.actor.load_state_dict(
                torch.load(path_a, map_location=self.device))
            self.performance.actor.to(self.device)
        print('  <= Restore {} with {} updates from {}.'.format(
            agent_type, step, model_folder))

    # === Visualization ===

    def get_figures(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError
