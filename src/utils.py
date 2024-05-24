# Some useful functions

import json 
import ast
import numpy as np
import matplotlib.pyplot as plt

from TD3_main import *

def sample_trajectory(env, policy, max_steps):
    state = env.reset()
    done = False
    trajectory = []
    total_reward = 0

    for t in range(max_steps):
        state = torch.FloatTensor(np.array(state).reshape(1, -1))
        action = policy(state).cpu().data.numpy().flatten()
        next_state, reward, _, done, _, _ = env.step(action)
        #trajectory.append((state, action, reward, next_state))
        trajectory.append(np.array(state[0][:2]))
        state = next_state
        total_reward += reward

        if done:
            break
    trajectory.append(np.array(np.array(next_state).reshape(1, -1)[0][:2]))
    return trajectory, total_reward

def plot_trajectory(trajectory):
    x = [element[3][0] for element in trajectory]
    y = [element[3][1] for element in trajectory]
    x = [0] + x # Add the first position (0,0,0) for the plot
    y = [0] + y
    fig = plt.figure(figsize = (12, 10))
    plt.grid(True)
    plt.scatter([env.x_goal], [env.y_goal], marker = "o", color = "r")
    plt.plot(x, y, 'k-o')
    plt.show()

def plot_reward(smooth_reward):
    # Plot cumulative rewards
    plt.figure(figsize=(10, 5))
    plt.plot(smooth_reward)
    plt.title("Cumulative Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.show()

def plot_reward_from_file(file_name, algo_name):
    # Read the dictionary from the file
    with open(file_name, 'r') as file:
        data = file.read()
        rewards_dict = ast.literal_eval(data)

    # Plotting the cumulative rewards
    for key, values in rewards_dict.items():
        plt.plot(values, label=f'Seed {key}')
    
    # Calculate and plot the average of the curves
    lengths = [len(values) for values in rewards_dict.values()]
    min_length = min(lengths)
    # Truncate arrays to the minimum lengths
    truncated_arrays = [np.array(values[:min_length]) for values in rewards_dict.values()]
    average_rewards = np.mean(truncated_arrays, axis=0)
    plt.plot(average_rewards, label='Average', color='black', linestyle='--', linewidth=2)

    plt.title(f'Cumulative Rewards and their average {algo_name}')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    
    plt.legend()
    plt.show()


# plot_curve("DDPG.txt", "DDPG")
# plot_curve("TD3.txt", "TD3")