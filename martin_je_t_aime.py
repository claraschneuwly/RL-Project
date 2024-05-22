
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from env import *


env_param = dict(a=0.5, 
                T=10, 
                k=0.1, 
                Ux=1, 
                Uy=1, 
                alpha=1, 
                sigma=0.1, 
                x_goal=4, 
                y_goal=4, 
                pos0=np.array([0, 0, 0]), 
                theta0=0, 
                dist_threshold=0.2, 
                max_steps=200,
                ocean=True,
                dt=1,
                max_thrust_speed=5,
                )

env = FluidMechanicsEnv(**env_param)



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
    


#### Load the actor model 
policy = Actor(env)
policy.load_state_dict(torch.load("SAC_seed0_wave05.pth"))

## Run and plot trajectory
done = False
obs = env.reset()
env.theta0 = 0
x, y = [obs[0]], [obs[1]]
while not done: 
    actions, _, _ = policy.get_action(torch.Tensor(obs).unsqueeze(0))
    actions = actions.detach().cpu().numpy()
    next_obs, rewards, sum_reward, terminations, steps_count, all_actions, = env.step(actions[0])
    x.append(next_obs[0])
    y.append(next_obs[1])
    obs = next_obs
    done = terminations
    
fig = plt.figure(figsize = (10, 5))
plt.grid(True)
plt.scatter([env.x_goal], [env.y_goal], marker = "o", color = "r")
plt.plot(x, y, 'k-o')
plt.show()