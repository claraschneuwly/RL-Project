import numpy as np
import torch
import matplotlib.pyplot as plt
import time

import ReplayBuffer
import TD3
from FinalEnv import *

# Define parameters for Twin Delayed Deep Deterministic
class Config:
    def __init__(self,
                discount=0.995,
                tau=0.005, 
                policy_noise=0.2,
                expl_noise=0.1,
                noise_clip=0.5,
                policy_freq=2,
                max_timesteps=8e2,
                start_timesteps=25e1,
                batch_size=256,
                seed=0):
        
        self.discount = discount # discount factor
        self.tau = tau # Target network update rate
        self.policy_noise = policy_noise # Noise added to target policy during critic update
        self.policy_freq = policy_freq  # Frequency of delayed policy updates
        self.expl_noise = expl_noise # Std of Gaussian exploration noise
        self.noise_clip = noise_clip # Range to clip target policy noise
        self.max_timesteps = max_timesteps # Max time steps to run environment
        self.start_timesteps = start_timesteps # Time steps during which initial random policy is used, after that we have collected sufficient data to train the agent 
        self.batch_size = batch_size # Batch size for both actor and critic
        self.seed = seed


def run_TD3(env, kwargs, seed=0):
    start_time = time.time()
    args = Config(seed=seed)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action

    policy = TD3.TD3(**kwargs)

    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    cumulative_episode_timesteps = 0 # track the time steps for plot
    history_reward = []  # store rewards to plot cumulative rewards
    smooth_reward = []
    SMOOTH_REWARD_WINDOW = 50

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space_sample()

        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, _, done, _, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env.max_steps else 0
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            history_reward.append(episode_reward)
            smooth_reward.append(np.mean(history_reward[-SMOOTH_REWARD_WINDOW:]))
            # if t+1 >= int(args.max_timesteps)-10: # Plot roughly the last 2 trajectories
            #     x = replay_buffer.next_state[cumulative_episode_timesteps:cumulative_episode_timesteps+episode_timesteps, 0]
            #     y = replay_buffer.next_state[cumulative_episode_timesteps:cumulative_episode_timesteps+episode_timesteps, 1]
            #     theta = replay_buffer.next_state[cumulative_episode_timesteps:cumulative_episode_timesteps+episode_timesteps, 3]
            #     x = np.insert(x, 0, 0) # Add the first position (0,0,0) for the plot
            #     y = np.insert(y, 0, 0)
            #     theta = np.insert(theta, 0, 0)
            #     fig = plt.figure(figsize = (14, 12))
            #     plt.grid(True)
            #     plt.scatter([env.x_goal], [env.y_goal], marker = "o", color = "r")
            #     plt.plot(x, y, 'k-o')
            #     plt.show()

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            cumulative_episode_timesteps += episode_timesteps
            episode_timesteps = 0
            episode_num += 1		
    end_time = time.time()
    execution_time = end_time - start_time
    return policy, t, episode_num, smooth_reward, execution_time



# Parameters for the environment
# env_param = dict(a=0.1, # range 0.1, 0.5, 1, 2, 5
#                         T=1, # wave period, range 10 to 20
#                         k=0.1, #wave number m^-1: 0.05 to 0.5
#                         Ux=0, #wind x component: -2 to 2
#                         Uy=0, 
#                         alpha=1, # vertical wind decay: around 1
#                         sigma=0, # noise wind parameter: around 10% wind speed
#                         x_goal=4, 
#                         y_goal=4, 
#                         pos0=np.array([0, 0, 0]), 
#                         theta0=0,
#                         dist_threshold=0.2, 
#                         max_steps=200, 
#                         ocean=True, # if false: still water env. If true: ocean like env
#                         dt=1, # time step. For now keep 1, we could go smaller
#                         max_thrust_speed = 1 # Robot's speed at 100% thrust 
#                         )

#policy, t, episode_num, smooth_reward, execution_time = run_TD3(env_param)
#plot_reward(smooth_reward)

#Source: https://github.com/sfujim/TD3/blob/master/TD3.py