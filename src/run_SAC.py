# SAC Base Framework
import numpy as np
import random
import math
from tqdm.notebook import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from gym import spaces
import matplotlib.pyplot as plt
from dataclasses import dataclass
from Env import *
from SAC import *
from utils import *

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

@dataclass
class Args:
    
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    # Algorithm specific arguments
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 25000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    print_reward: bool = True
    """print the smooth reward after each epoch during training"""

# Seed everything for reproducibility
args = Args()
args.print_reward = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.backends.cudnn.deterministic = args.torch_deterministic

# Create and train SAC agent
sac_agent = SAC(env, args)

policy, t, episode_num, smooth_reward, execution_time = sac_agent.train()

# plot_reward(smooth_reward)