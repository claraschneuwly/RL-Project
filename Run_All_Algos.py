import json 

# from TD3_main import *
from DDPG_main import *

# smooth_reward_TD3 = dict()
# for seed in range(3):
#     smooth_reward = run_TD3(seed=seed)
#     smooth_reward_TD3[seed] = smooth_reward

# with open('TD3.txt', 'w') as convert_file: 
#      convert_file.write(json.dumps(smooth_reward_TD3))

smooth_reward_DDPG = dict()
for seed in range(3):
    smooth_reward = run_DDPG(seed=seed)
    smooth_reward_DDPG[seed] = smooth_reward

with open('DDPG.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(smooth_reward_DDPG))