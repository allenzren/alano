# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from abc import ABC, abstractmethod
from collections import namedtuple
from queue import PriorityQueue
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from .replay_memory import ReplayMemory

TransitionLatent = namedtuple('TransitionLatent',
                              ['z', 's', 'a', 'r', 's_', 'done', 'info'])


class AgentBase(ABC):
    def __init__(self, CONFIG, CONFIG_ENV, CONFIG_UPDATE):
        super(AgentBase, self).__init__()

        self.device = CONFIG.DEVICE
        self.image_device = CONFIG.IMAGE_DEVICE
        self.n_envs = CONFIG.NUM_CPUS
        self.action_dim = CONFIG_ENV.ACTION_DIM
        self.CONFIG = CONFIG
        self.MAX_TRAIN_STEPS = CONFIG_ENV.MAX_TRAIN_STEPS
        self.MAX_EVAL_STEPS = CONFIG_ENV.MAX_EVAL_STEPS
        self.OBS_BUFFER = CONFIG_ENV.OBS_BUFFER
        self.NUM_VISUALIZE_TASK = CONFIG.NUM_VISUALIZE_TASK

        #! We assume backup and performance use the same parameters.
        self.BATCH_SIZE = CONFIG_UPDATE.BATCH_SIZE
        self.UPDATE_PERIOD = CONFIG_UPDATE.UPDATE_PERIOD

        # memory
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY, CONFIG.SEED)
        self.rng = np.random.default_rng(seed=CONFIG.SEED)

        # saving models
        self.save_top_k = self.CONFIG.SAVE_TOP_K
        self.pq_top_k = PriorityQueue()

    @property
    @abstractmethod
    def has_backup(self):
        raise NotImplementedError

    def sample_batch(self, batch_size=None, recent_size=None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        transitions, _ = self.memory.sample(batch_size, recent_size)
        batch = TransitionLatent(*zip(*transitions))
        return batch

    def store_transition(self, *args):
        self.memory.update(TransitionLatent(*args))

    def unpack_batch(self,
                     batch,
                     get_latent=True,
                     get_perf_action=False,
                     get_l_x_ra=False,
                     get_rnn_hidden=False,
                     get_latent_backup=False):
        non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)),
                                      dtype=torch.bool).view(-1).to(
                                          self.device)
        non_final_state_nxt = torch.cat([
            s for done, s in zip(batch.done, batch.s_) if not done
        ]).to(self.device)
        state = torch.cat(batch.s).to(self.device)
        reward = torch.FloatTensor(batch.r).to(self.device)

        # import matplotlib.pyplot as plt
        # for ind in range(10):
        #     f, axarr = plt.subplots(2, 4)
        #     for i_ind in range(4):
        #         axarr[0, i_ind].imshow(
        #             np.transpose(
        #                 state[ind][(i_ind * 3):((i_ind+1) * 3)].cpu().numpy(),
        #                 [1, 2, 0]
        #             )
        #         )
        #         axarr[1, i_ind].imshow(
        #             np.transpose(
        #                 non_final_state_nxt[ind][(i_ind
        #                                           * 3):((i_ind+1)
        #                                                 * 3)].cpu().numpy(),
        #                 [1, 2, 0]
        #             )
        #         )
        #     plt.show()

        g_x = torch.FloatTensor([info['g_x'] for info in batch.info
                                 ]).to(self.device).view(-1)
        # l_x = torch.FloatTensor([info['l_x'] for info in batch.info]
        #                        ).to(self.device).view(-1)

        if get_perf_action:  # recovery RL separates a_shield and a_perf.
            if batch.info[0]['a_perf'].dim() == 1:
                action = torch.FloatTensor(
                    [info['a_perf'] for info in batch.info])
            else:
                action = torch.cat([info['a_perf'] for info in batch.info])
            action = action.to(self.device)
        else:
            action = torch.cat(batch.a).to(self.device)

        latent = None
        if get_latent:
            if get_latent_backup:
                latent = torch.cat([info['z_backup'] for info in batch.info])
                latent = latent.to(self.device)
            else:
                latent = torch.cat(batch.z).to(self.device)

        append = torch.cat([info['append']
                            for info in batch.info]).to(self.device)
        non_final_append_nxt = torch.cat([
            info['append_nxt'] for info in batch.info
        ]).to(self.device)[non_final_mask]

        l_x_ra = None
        if get_l_x_ra:
            l_x_ra = torch.FloatTensor([info['l_x_ra'] for info in batch.info])
            l_x_ra = l_x_ra.to(self.device).view(-1)

        binary_cost = torch.FloatTensor(
            [info['binary_cost'] for info in batch.info])
        binary_cost = binary_cost.to(self.device).view(-1)

        hn = None
        cn = None
        if get_rnn_hidden:
            hn = batch.info[0]['hn'].to(self.device)  # only get initial, 1x
            cn = batch.info[0]['hn'].to(self.device)
        return (non_final_mask, non_final_state_nxt, state, action, reward,
                g_x, latent, append, non_final_append_nxt, l_x_ra, binary_cost,
                hn, cn)

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

    def get_figures(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError
