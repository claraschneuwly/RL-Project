import torch
import json
import TD3
import TD3_main
import FinalEnv
from utils import *

env = FinalEnv.FluidMechanicsEnv(a=0.5, # range 0.1, 0.5, 1, 2, 5
                        T=1, # wave period, range 10 to 20
                        k=0.1, #wave number m^-1: 0.05 to 0.5
                        Ux=0, #wind x component: -2 to 2
                        Uy=0, 
                        alpha=1, # vertical wind decay: around 1
                        sigma=0, # noise wind parameter: around 10% wind speed
                        x_goal=4, 
                        y_goal=4, 
                        pos0=np.array([0, 0, 0]), 
                        theta0=0,
                        dist_threshold=0.2, 
                        max_steps=200, 
                        ocean=True, # if false: still water env. If true: ocean like env
                        dt=1, # time step. For now keep 1, we could go smaller
                        max_thrust_speed = 1 # Robot's speed at 100% thrust 
                        )

args = TD3_main.Config()

kwargs={"state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "max_action": env.max_action,
        "discount": args.discount,
        "tau": args.tau}
kwargs["policy_noise"] = args.policy_noise * env.max_action
kwargs["noise_clip"] = args.noise_clip * env.max_action
kwargs["policy_freq"] = args.policy_freq

reward_dict = dict()
#### Train and save the actor model
for seed in range(0,5):
        model, t, episode_num, smooth_reward, execution_time = TD3_main.run_TD3(env, kwargs, seed=seed)
        print(f"seed: {seed}, Total steps: {t}, Num episodes: {episode_num}, execution_time : {execution_time}")
        # torch.save(policy.actor.state_dict(), "TD3_policy")
        reward_dict[seed] = smooth_reward

        # #### Plot smooth reward
        plot_reward(smooth_reward)

        # #### Load the actor model 
        # model = TD3.TD3(**kwargs)
        # policy = model.actor

        if seed == 0:
                torch.save(model.actor.state_dict(), "TD3_seed0_wave05")
        # policy.load_state_dict(torch.load("TD3_policy"))
        # policy.eval()

        #### Sample a trajectory 
        # trajectory, total_reward = sample_trajectory(env, policy, env.max_steps)

        # print("Total reward from sampled trajectory:", total_reward)
        # plot_trajectory(trajectory)
with open('TD3_05.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(reward_dict))