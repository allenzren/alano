# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#          Allen Z. Ren (allen.ren@princeton.edu)

from train import *
from train.base_sac import BaseSAC


class SAC_mini(BaseSAC):
    def __init__(self, CONFIG, CONFIG_ARCH, CONFIG_ENV):
        super().__init__(CONFIG, CONFIG_ARCH, CONFIG_ENV)

    @property
    def has_latent(self):
        return False

    @property
    def latent_dist(self):
        return None

    def build_network(self,
                      build_optimizer=True,
                      verbose=True,
                      actor_path=None,
                      critic_path=None,
                      tie_conv=True):
        super().build_network(verbose,
                              actor_path=actor_path,
                              critic_path=critic_path,
                              tie_conv=tie_conv)

        # Set up optimizer
        if build_optimizer:
            super().build_optimizer()
        else:
            for _, param in self.actor.named_parameters():
                param.requires_grad = False
            for _, param in self.critic.named_parameters():
                param.requires_grad = False
            self.actor.eval()
            self.critic.eval()

    # region: main update functions
    def update_critic(self, batch):
        (non_final_mask, non_final_state_nxt, state, action, reward, g_x, _,
         append, non_final_append_nxt, l_x_ra, binary_cost, _, _) = batch
        self.critic.train()
        self.critic_target.eval()
        self.actor.eval()

        # == get Q(s,a) ==
        q1, q2 = self.critic(
            state, action,
            append=append)  # Used to compute loss (non-target part).

        # == placeholder for target ==
        y = torch.zeros(self.BATCH_SIZE).float().to(self.device)

        # == compute actor next_actions and feed to critic_target ==
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                non_final_state_nxt, append=non_final_append_nxt)
            if self.critic_has_act_ind:
                act_ind_rep = self.act_ind.repeat(next_actions.shape[0], 1)
                next_actions = torch.cat((next_actions, act_ind_rep), dim=-1)
            next_q1, next_q2 = self.critic_target(non_final_state_nxt,
                                                  next_actions,
                                                  append=non_final_append_nxt)
            # max for reach-avoid Bellman equation, safety Bellman equation and
            # risk (recovery RL)
            if (self.mode == 'RA' or self.mode == 'safety'
                    or self.mode == 'risk'):
                q_max = torch.max(next_q1, next_q2).view(-1)
            elif self.mode == 'performance':
                q_min = torch.min(next_q1, next_q2).view(-1)
            else:
                raise ValueError("Unsupported RL mode.")

            final_mask = torch.logical_not(non_final_mask)
            if self.mode == 'RA':
                y[non_final_mask] = (
                    (1.0 - self.GAMMA) *
                    torch.max(l_x_ra[non_final_mask], g_x[non_final_mask]) +
                    self.GAMMA *
                    torch.max(g_x[non_final_mask],
                              torch.min(l_x_ra[non_final_mask], q_max)))
                if self.terminal_type == 'g':
                    y[final_mask] = g_x[final_mask]
                elif self.terminal_type == 'max':
                    y[final_mask] = torch.max(l_x_ra[final_mask],
                                              g_x[final_mask])
                else:
                    raise ValueError("invalid terminal type")
            elif self.mode == 'safety':
                # V(s) = max{ g(s), V(s') }
                # Q(s, u) = V( f(s,u) ) = max{ g(s'), min_{u'} Q(s', u') }
                # normal state
                y[non_final_mask] = (
                    (1.0 - self.GAMMA) * g_x[non_final_mask] +
                    self.GAMMA * torch.max(g_x[non_final_mask], q_max))

                # terminal state
                y[final_mask] = g_x[final_mask]
            elif self.mode == 'performance':
                target_q = q_min - self.alpha * next_log_prob.view(
                    -1)  # already masked - can be lower dim than y
                y = reward
                y[non_final_mask] += self.GAMMA * target_q
            elif self.mode == 'risk':
                y = binary_cost  # y = 1 if it's a terminal state
                y[non_final_mask] += self.GAMMA * q_max

        # == MSE update for both Q1 and Q2 ==
        loss_q1 = F.mse_loss(input=q1.view(-1), target=y)
        loss_q2 = F.mse_loss(input=q2.view(-1), target=y)
        loss_q = loss_q1 + loss_q2

        # == backpropagation ==
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        return loss_q.item()

    def update_actor(self, batch):
        """
        Use detach_encoder=True to not update conv layers
        """
        _, _, state, _, _, _, _, append, _, _, _, _, _ = batch

        self.critic.eval()
        self.actor.train()

        action_sample, log_prob = self.actor.sample(state,
                                                    append=append,
                                                    detach_encoder=True)
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(action_sample.shape[0], 1)
            action_sample = torch.cat((action_sample, act_ind_rep), dim=-1)
        q_pi_1, q_pi_2 = self.critic(state,
                                     action_sample,
                                     append=append,
                                     detach_encoder=True)

        if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
            q_pi = torch.max(q_pi_1, q_pi_2)
        elif self.mode == 'performance':
            q_pi = torch.min(q_pi_1, q_pi_2)

        # cost: min_theta E[ Q + alpha * (log pi + H)]
        # loss_pi = Q + alpha * log pi
        # reward: max_theta E[ Q - alpha * (log pi + H)]
        # loss_pi = -Q + alpha * log pi
        loss_entropy = log_prob.view(-1).mean()
        loss_q_eval = q_pi.mean()
        if self.mode == 'RA' or self.mode == 'safety' or self.mode == 'risk':
            loss_pi = loss_q_eval + self.alpha * loss_entropy
        elif self.mode == 'performance':
            loss_pi = -loss_q_eval + self.alpha * loss_entropy
        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        # Automatic temperature tuning
        loss_alpha = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        if self.LEARN_ALPHA:
            self.log_alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.log_alpha_optimizer.step()
        return loss_pi.item(), loss_entropy.item(), loss_alpha.item()

    def update(self, batch, timer, update_period=2):
        self.critic.train()
        self.actor.train()

        loss_q = self.update_critic(batch)
        loss_pi, loss_entropy, loss_alpha = 0, 0, 0
        if timer % update_period == 0:
            loss_pi, loss_entropy, loss_alpha = self.update_actor(batch)
            self.update_target_networks()

        self.critic.eval()
        self.actor.eval()

        return loss_q, loss_pi, loss_entropy, loss_alpha

    # endregion

    def value(self, obs, append):
        u = self.actor(obs, append=append)
        u = torch.from_numpy(u).to(self.device)
        if self.critic_has_act_ind:
            act_ind_rep = self.act_ind.repeat(u.shape[0], 1)
            u = torch.cat((u, act_ind_rep), dim=-1)
        v = self.critic(obs, u, append=append)[0]
        if len(obs.shape) == 3:
            v = v[0]
        return v
