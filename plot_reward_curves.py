import matplotlib.pyplot as plt
import ast
import numpy as np

def plot_curve(file_name, algo_name):
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
    # Truncate arrays to the minimum lengthws
    truncated_arrays = [np.array(values[:min_length]) for values in rewards_dict.values()]
    average_rewards = np.mean(truncated_arrays, axis=0)
    plt.plot(average_rewards, label='Average', color='black', linestyle='--', linewidth=2)

    plt.title(f'Cumulative Rewards and their average {algo_name}')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    
    plt.legend()
    plt.show()

plot_curve("DDPG.txt", "DDPG")
plot_curve("TD3.txt", "TD3")