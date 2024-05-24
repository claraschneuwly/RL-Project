import ast
import numpy as np
import matplotlib.pyplot as plt
import json
 
# Read the dictionary from the file
with open("TD3_05_rewards.txt", 'r') as file:
    data_TD3 = file.read()
    rewards_dict_TD3 = ast.literal_eval(data_TD3)
with open("DDPG_05_rewards.txt", 'r') as file:
    data_DDPG = file.read()
    rewards_dict_DDPG = ast.literal_eval(data_DDPG)
with open("SAC_05.txt", 'r') as file:
    data_SAC = file.read()
    rewards_dict_SAC = ast.literal_eval(data_SAC)

data_PPO = open('PPO_smooth_reward_real_param.json')
rewards_dict_PPO = json.load(data_PPO)
print(len(rewards_dict_PPO))

# Calculate and plot the average of the curves
lengths_TD3 = [len(values) for values in rewards_dict_TD3.values()]
lengths_DDPG = [len(values) for values in rewards_dict_DDPG.values()]
lengths_SAC = [len(values) for values in rewards_dict_SAC.values()]
lengths_PPO = [len(rewards_dict_PPO)]
min_length_TD3 = min(lengths_TD3)
min_length_DDPG = min(lengths_DDPG)
min_length_SAC = min(lengths_SAC)
min_length_PPO = min(lengths_PPO)
print(f"#Episodes PPO: {min_length_PPO}, TD3 {min_length_TD3}, DDPG {min_length_DDPG}, SAC {min_length_SAC}")

# Truncate arrays to the minimum lengthws
truncated_arrays_DDPG = [np.array(values[:min_length_DDPG]) for values in rewards_dict_DDPG.values()]
average_rewards_DDPG = np.mean(truncated_arrays_DDPG, axis=0)

truncated_arrays_TD3 = [np.array(values[:min_length_TD3]) for values in rewards_dict_TD3.values()]
average_rewards_TD3 = np.mean(truncated_arrays_TD3, axis=0)

truncated_arrays_SAC = [np.array(values[:min_length_SAC]) for values in rewards_dict_SAC.values()]
average_rewards_SAC = np.mean(truncated_arrays_SAC, axis=0)

truncated_arrays_PPO = rewards_dict_PPO#[np.array(values[:min_length_PPO]) for values in rewards_dict_PPO.values()]
average_rewards_PPO = truncated_arrays_PPO# np.mean(truncated_arrays_PPO, axis=0)

plt.plot(average_rewards_TD3, label='TD3', color='blue', linewidth=2)
plt.plot(average_rewards_DDPG, label='DDPG', color='orange', linewidth=2)
plt.plot(average_rewards_SAC, label='SAC', color='green', linewidth=2)
plt.plot(average_rewards_PPO, label='PPO', color='gray', linewidth=2)
plt.xscale('log')
#plt.title(f'Cumulative Rewards averaged over 5 seeds')
#plt.xlabel('Episodes')
#plt.ylabel('Cumulative Reward')

plt.legend()
plt.show()

