import numpy as np
import torch

import utils_TD3
import TD3
from FinalEnv import *

env_param = dict(a=0, 
				  T=1, 
				  k=0.1, 
				  Ux=0, 
				  Uy=0, 
				  alpha=1, 
				  sigma=0, 
				  x_goal=2, 
				  y_goal=2, 
				  pos0=np.array([0, 0, 0]), 
				  theta0=0, 
				  dist_threshold=0.3, 
				  max_steps=200)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# def eval_policy(policy, env_name, seed, eval_episodes=10):
def eval_policy(policy, env_param, eval_episodes=10):
	eval_env = FluidMechanicsEnv(**env_param)
	# eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, _, done, _, _= eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

class Config:
    def __init__(self,
                discount=0.995,
                tau=0.005,
                policy_noise=0.2,
                expl_noise=0.1,
                noise_clip=0.5,
                eval_freq=2,
                policy_freq=2,
                max_timesteps=10000,
                start_timesteps= 1000,
                batch_size=50,
                seed=0):
		
        self.discount = discount # discount factor
        self.tau = tau
        self.policy_noise = policy_noise
        self.eval_freq = eval_freq
        self.policy_freq = policy_freq
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.max_timesteps = max_timesteps
        self.start_timesteps = start_timesteps
        self.batch_size = batch_size
        self.seed = seed

if __name__ == "__main__":
	
	args = Config()
	
	env = FluidMechanicsEnv(**env_param)

	# Set seeds
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	state_dim = env.state_dim
	action_dim = env.action_dim
	max_action = env.max_action

	kwargs={"state_dim": state_dim,
		 "action_dim": action_dim,
		 "max_action": max_action,
		 "discount": args.discount,
		 "tau": args.tau}
	
	# Initialize policy
	
    # Target policy smoothing is scaled wrt the action scale
	kwargs["policy_noise"] = args.policy_noise * max_action
	kwargs["noise_clip"] = args.noise_clip * max_action
	kwargs["policy_freq"] = args.policy_freq
	policy = TD3.TD3(**kwargs)

	replay_buffer = utils_TD3.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	# evaluations = [eval_policy(policy, env_param)] #, args.seed

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

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
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		# if (t + 1) % args.eval_freq == 0:
		# 	evaluations.append(eval_policy(policy, env_param))
		# 	# np.save(f"./results/{file_name}", evaluations)
		# 	# if args.save_model: policy.save(f"./models/{file_name}")