import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.prod(env.observation_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e9)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class SAC:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.actor = Actor(env).to(self.device)
        self.qf1 = SoftQNetwork(env).to(self.device)
        self.qf2 = SoftQNetwork(env).to(self.device)
        self.qf1_target = SoftQNetwork(env).to(self.device)
        self.qf2_target = SoftQNetwork(env).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        self.replay_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0])

    def train(self):
        obs = self.env.reset()
        start_time = time.time()
        episode_num = 0
        episode_reward = 0
        history_reward = []
        smooth_reward = []
        SMOOTH_REWARD_WINDOW = 50
        length_ep = 0
        converged = False

        for global_step in range(self.args.total_timesteps):

            if global_step < self.args.learning_starts:
                actions = np.array(self.env.action_space.sample())
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            next_obs, reward, sum_reward, done, steps_count, all_actions = self.env.step(actions)

            self.replay_buffer.add(obs, actions, next_obs, reward, done)
            
            obs = next_obs
            episode_reward += reward
            length_ep +=1
            if done:
                obs = self.env.reset()
                episode_num += 1
                history_reward.append(episode_reward)
                smooth_reward.append(np.mean(history_reward[-SMOOTH_REWARD_WINDOW:]))
                if self.args.print_reward:
                    print('Episode:', episode_num)
                    print('----------')
                    print('Lenght:', length_ep)
                    print('Reward:', episode_reward)
                    print('\n')
                length_ep = 0
                episode_reward = 0
                # if (not converged) and (smooth_reward[-1] >= 9.8):
                #     converged=True
                #     print(f'for wave.a = {self.env.wave.a}: nbr of episode to convergence {global_step + 1}')
                #     print('\n')
                #     return global_step + 1


            if global_step > self.args.learning_starts:
                data = self.replay_buffer.sample(self.args.batch_size)
                self.update_parameters(data, global_step)

        execution_time = time.time() - start_time

        return self.actor, self.args.total_timesteps, episode_num, smooth_reward, execution_time


    def update_parameters(self, data, global_step):
        # Update Q functions
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data[2])
            qf1_next_target = self.qf1_target(data[2], next_state_actions)
            qf2_next_target = self.qf2_target(data[2], next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data[3].flatten() + data[4].flatten() * self.args.gamma * min_qf_next_target.view(-1)

        qf1_a_values = self.qf1(data[0], data[1]).view(-1)
        qf2_a_values = self.qf2(data[0], data[1]).view(-1)
        qf_loss = F.mse_loss(qf1_a_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # Policy update and target network update
        if global_step % self.args.policy_frequency == 0:
            for _ in range(self.args.policy_frequency):  # TD 3 style delayed update
                pi, log_pi, _ = self.actor.get_action(data[0])
                qf1_pi = self.qf1(data[0], pi)
                qf2_pi = self.qf2(data[0], pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data[0])
                    alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # Target networks soft update
        if global_step % self.args.target_network_frequency == 0:
            self.update_targets()

        # Optional: Logging and print statements for monitoring
        if global_step % 1000 == 0 and self.args.print_reward:
            print(f"Step: {global_step}, QF Loss: {qf_loss.item() / 2.0}")
            print(f"Step: {global_step}, Actor Loss: {actor_loss.item()}")

    def update_targets(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


