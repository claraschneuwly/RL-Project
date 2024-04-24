from RL_env import *
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
        action = dist.sample()
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