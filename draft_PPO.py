from DiscreteEnv import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import pandas as pd

# Define Actor and Critic

class ACModel(nn.Module):
    """ Represents an Actor Crictic model
        Parameters
        ----
        num_actions : The action space of the environment. """
    def __init__(self, num_actions=5):
        super(ACModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(5, 128),  # state has 5 dimensions [x, y, z, rudder, orientation]
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        self.critic = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """ Performs a forward pass through the actor-critic network 
        returns: dist( The distribution of actions from policy), value output by critic network"""
        x = torch.tensor(x, dtype=torch.float32)
        logits = self.actor(x)
        value = self.critic(x)
        return Categorical(logits=logits), value.squeeze()


# Evaluate the policy

def collect_experiences(env, model, num_steps):
    """Collects rollouts and computes advantages."""
    states, actions, rewards, values, log_probs = [], [], [], [], []
    state = env.reset()
    for i in range(num_steps):
        dist, value = model([state])
        ###########################
        action = dist.sample() # Probleme ici. je comprends pas comment on définit la distribution
        print(action)
        
        next_state, reward, done, _, info = env.step(action.item())
        
        log_prob = dist.log_prob(action)
        
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        values.append(value.item())
        log_probs.append(log_prob.item())
        
        state = next_state
        if done:
            state = env.reset()

    return states, actions, rewards, values, log_probs

def compute_returns_and_advantages(rewards, values, discount=0.99, gae_lambda=0.95):
    """Compute Adavantage wiht GAE. See Section 4.4.2 in the lecture notes."""
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)

    returns, advantages = torch.zeros_like(rewards), torch.zeros_like(values)

    gae = 0
    R = 0
    for i in reversed(range(len(rewards)-1)):
        delta = rewards[i] + discount * values[i + 1] - values[i]
        gae = delta + discount * gae_lambda * advantages[i+1]
        advantages[i] = gae
        R = rewards[i] + discount * R
        returns[i] = R

    return returns, advantages

# Function to update the model
def update_model(model, optimizer, states, actions, log_probs, returns, advantages, clip_param=0.2):
    dist, values = model(states)
    new_log_probs = dist.log_prob(torch.tensor(actions))
    
    ratios = torch.exp(new_log_probs - torch.tensor(log_probs))
    clipped_ratios = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
    loss_clip = torch.min(ratios * torch.tensor(advantages), clipped_ratios * torch.tensor(advantages))
    
    loss_critic = (torch.tensor(returns) - values).pow(2)
    
    loss = -(loss_clip - 0.5 * loss_critic).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Main training loop
def run_training_loop(env, num_episodes, num_steps_per_episode):
    model = ACModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for episode in range(num_episodes):
        print(env.coords)
        states, actions, rewards, values, log_probs = collect_experiences(env, model, num_steps_per_episode)
        returns, advantages = compute_returns_and_advantages(rewards, values)
        loss = update_model(model, optimizer, states, actions, log_probs, returns, advantages)
        
        print(f'Episode {episode + 1}, Loss: {loss}')

env = Env(x_goal=1, y_goal=1, max_steps=10)
run_training_loop(env, 20, 20)

# from DiscreteEnv import *
# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# from tqdm import tqdm
# import pandas as pd

# class ActorCritic(nn.Module):
#     def __init__(self):
#         super(ActorCritic, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(5, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#         )
#         self.policy_head = nn.Linear(128, 5)  # 5 discrete actions for rudder positions [0, 0.25, 0.5, 0.75, 1]
#         self.value_head = nn.Linear(128, 1)

#     def forward(self, x):
#         x = self.fc(x)
#         return Categorical(logits=self.policy_head(x)), self.value_head(x)

# def compute_losses(acmodel, obs, actions, old_log_probs, returns, advantages, clip_ratio=0.2):
#     dist, values = acmodel(obs)
#     new_log_probs = dist.log_prob(actions)

#     # Policy loss using the clipped PPO objective
#     ratio = torch.exp(new_log_probs - old_log_probs)
#     clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
#     policy_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()

#     # Approximation of the KL divergence
#     approx_kl = (old_log_probs - new_log_probs).mean()

#     # Value loss as mean squared error
#     value_loss = (returns - values.squeeze()).pow(2).mean()

#     return policy_loss, approx_kl, value_loss

# def update_parameters_ppo(optimizer, acmodel, data, clip_ratio=0.2):
#     obs = torch.tensor(data['obs'], dtype=torch.float32)
#     actions = torch.tensor(data['actions'], dtype=torch.int32)
#     old_log_probs = torch.tensor(data['log_probs'], dtype=torch.float32)
#     returns = torch.tensor(data['returns'], dtype=torch.float32)
#     advantages = torch.tensor(data['advantages'], dtype=torch.float32)

#     policy_loss, approx_kl, value_loss = compute_losses(acmodel, obs, actions, old_log_probs, returns, advantages, clip_ratio)

#     optimizer.zero_grad()
#     total_loss = policy_loss + value_loss
#     total_loss.backward()
#     optimizer.step()

#     return {
#         "policy_loss": policy_loss.item(),
#         "value_loss": value_loss.item(),
#         "approx_kl": approx_kl.item()
#     }

# # Model and optimizer
# acmodel = ActorCritic()
# optimizer = optim.Adam(acmodel.parameters(), lr=0.001)

# # Example of data format expected
# data = {
#     'obs': np.random.randn(10, 5),  # 10 observations of the state
#     'actions': np.random.randint(0, 5, size=(10,)),
#     'log_probs': np.random.randn(10),
#     'returns': np.random.randn(10),
#     'advantages': np.random.randn(10)
# }

# # Run update
# logs = update_parameters_ppo(optimizer, acmodel, data)
# print(logs)
