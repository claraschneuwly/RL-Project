"""## Train Agent"""
#from FinalEnv import *
from DiscreteEnv import *
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
random.seed(123)

class ACModel(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(ACModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # self.actor = nn.Sequential(
        #         nn.Linear(state_dim, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 1),
        #         nn.Tanh()
        #     )

        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, hidden_size),
        #     nn.ReLU(), 
        #     nn.Linear(hidden_size, 1)
        # )

        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        self.a_log_std = nn.Parameter(torch.zeros(1, 1))
        
    def forward(self, state):
        mean = self.actor(state)
        log_std = self.a_log_std.expand_as(mean)
        std = torch.exp(log_std)
        #print(self.actor[4].weight)
        #mean = 90 * torch.tanh(self.actor(state)) # Scale the output to -90 to 90 rang
        rudder_dist = dist.Normal(mean, std)
        value = self.critic(state)

        return rudder_dist, value

class Config:
    def __init__(self,
                score_threshold=0.93,
                discount=0.995,
                lr=1e-3,
                max_grad_norm=0.5,
                log_interval=10,
                max_episodes=1000,
                gae_lambda=0.95,
                use_critic=False,
                clip_ratio=0.2,
                target_kl=0.02,
                train_ac_iters=5,
                use_discounted_reward=False,
                entropy_coef=0.01,
                use_gae=False):

        self.score_threshold = score_threshold # criterion for early stopping. If the rolling average reward (over the last 100 episodes) is greater than it, it ends.
        self.discount = discount # discount factor
        self.lr = lr # learning rate
        self.max_grad_norm = max_grad_norm # the maximum gradient norm (https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
        self.log_interval = log_interval # logging interval
        self.max_episodes = max_episodes # the maximum number of episodes.
        self.use_critic = use_critic # whether to use critic or not.
        self.clip_ratio = clip_ratio # clip_ratio of PPO.
        self.target_kl = target_kl # target KL divergence for early stoping train_ac_iters for PPO
        self.train_ac_iters = train_ac_iters # how many time to train ac_model using current computed old_logps
        self.gae_lambda=gae_lambda # lambda in Generalized Advantage Estimation (GAE)
        self.use_discounted_reward=use_discounted_reward # whether use discounted reward or not.
        self.entropy_coef = entropy_coef # entropy coefficient for PPO
        self.use_gae = use_gae # whether to use GAE or not.

def compute_discounted_return(rewards, discount, device=None):
    """
		rewards: reward obtained at timestep.  Shape: (T,)
		discount: discount factor. float
    ----
    returns: sum of discounted rewards. Shape: (T,)
		"""
    returns = torch.zeros(*rewards.shape, device=device)

    R = 0
    for t in reversed(range((rewards.shape[0]))):
        R = rewards[t] + discount * R
        returns[t] = R
    return returns

def compute_advantage_gae(values, rewards, T, gae_lambda, discount):
    """
    Compute Adavantage wiht GAE. See Section 4.4.2 in the lecture notes.

    values: value at each timestep (T,)
    rewards: reward obtained at each timestep.  Shape: (T,)
    T: the number of frames, float
    gae_lambda: hyperparameter, float
    discount: discount factor, float
    -----
    returns:

    advantages : tensor.float. Shape [T,]

                 gae advantage term for timesteps 0 to T

    """
    advantages = torch.zeros_like(values)
    for i in reversed(range(T)):
        next_value = values[i+1]
        next_advantage = advantages[i+1]

        delta = rewards[i] + discount * next_value  - values[i]
        advantages[i] = delta + discount * gae_lambda * next_advantage
    return advantages[:T]

def collect_experiences(env, acmodel, args, device=None):
    """Collects rollouts and computes advantages.
    Returns
    -------
    exps : dict
        Contains actions, rewards, advantages etc as attributes.
        Each attribute, e.g. `exps['reward']` has a shape
        (self.num_frames, ...).
    logs : dict
        Useful stats about the training process, including the average
        reward, policy loss, value loss, etc.
    """


    MAX_FRAMES_PER_EP = 300
    shape = (MAX_FRAMES_PER_EP, )

    actions = torch.zeros((MAX_FRAMES_PER_EP, 1), device=device, dtype=torch.int)
    values = torch.zeros(*shape, device=device)
    rewards = torch.zeros(*shape, device=device)
    log_probs = torch.zeros(*shape, device=device)
    obss = torch.zeros((MAX_FRAMES_PER_EP, env.state_dim), device=device)

    obs = np.array(env.reset())

    total_return = 0

    T = 0

    while True:
        # Do one agent-environment interaction
        #print(env.coords)

        with torch.no_grad():
            obs = torch.from_numpy(obs).float()
            obs = obs.unsqueeze(0)
            #thrust_dist, rudder_dist, value = acmodel(obs)
            rudder_dist, value = acmodel(obs)
        # print('\n')
        # print('obs', obs)
        action = rudder_dist.sample().squeeze()
        # print('action', action)
        obss[T] = obs
        obs, reward, done, _, _ = env.step(action)
        # print('next obs', obs)

        # Update experiences values
        actions[T] = action
        values[T] = value
        rewards[T] = reward

        log_probs[T] = rudder_dist.log_prob(action).squeeze().squeeze()
        # print('log_probs', rudder_dist.log_prob(action).squeeze().squeeze())
        total_return += reward
        T += 1

        if done or T>=MAX_FRAMES_PER_EP-1:
            break

    discounted_reward = compute_discounted_return(rewards[:T], args.discount, device)
    exps = dict(
        obs = obss[:T],
        action = actions[:T],
        value  = values[:T],
        reward = rewards[:T],
        advantage = discounted_reward-values[:T],
        log_prob = log_probs[:T],
        discounted_reward = discounted_reward,
        advantage_gae=compute_advantage_gae(values, rewards, T, args.gae_lambda, args.discount)
    )

    logs = {
        "return_per_episode": total_return,
        "num_frames": T
    }

    return exps, logs

def run_experiment(args, parameter_update, env_param, seed=0):
    """
    Upper level function for running experiments to analyze reinforce and
    policy gradient methods. Instantiates a model, collects epxeriences, and
    then updates the neccessary parameters.

    args: Config arguments. dict
    paramter_update: function used to update model parameters
    seed: random seed. int

    return: DataFrame indexed by episode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env(**env_param)

    acmodel = ACModel(env.state_dim)
    acmodel.to(device)

    is_solved = False

    SMOOTH_REWARD_WINDOW = 50

    pd_logs, rewards = [], [0]*SMOOTH_REWARD_WINDOW

    optimizer = torch.optim.Adam(acmodel.parameters(), lr=args.lr)
    num_frames = 0
    
    pbar = tqdm(range(args.max_episodes))
    for update in pbar:
        exps, logs1 = collect_experiences(env, acmodel, args, device)
        logs2 = parameter_update(optimizer, acmodel, exps, args)

        logs = {**logs1, **logs2}

        num_frames += logs["num_frames"]

        rewards.append(logs["return_per_episode"])

        smooth_reward = np.mean(rewards[-SMOOTH_REWARD_WINDOW:])

        data = {'episode':update, 'num_frames':num_frames, 'smooth_reward':smooth_reward,
                'reward':logs["return_per_episode"], 'policy_loss':logs["policy_loss"]}

        if args.use_critic:
            data['value_loss'] = logs["value_loss"]

        pd_logs.append(data)

        pbar.set_postfix(data)

        # Early terminate
        if smooth_reward >= args.score_threshold:
            is_solved = True
            break

    if is_solved:
        print('Solved!')

    return pd.DataFrame(pd_logs).set_index('episode'), exps

def update_parameters_ppo(optimizer, acmodel, sb, args):
    def _compute_policy_loss_ppo(obs, old_logp, actions, advantages):
        '''
        Computes the policy loss for PPO.

        obs: observeration to pass into acmodel. shape: (T,)
        old_logp: log probabilities from previous timestep. shape: (T,)
        actions: action at this timestep. shape: (T,ImWidth,ImHeight,Channels)
        advantages: the computed advantages. shape: (T,)

        ---
        returns

        policy_loss : ppo policy loss as shown in line 6 of PPO alg. tensor.float. Shape (,1)
        approx_kl: an appoximation of the kl_divergence. tensor.float. Shape (,1)
        '''
        policy_loss, approx_kl = 0, 0

        # Policy loss
        T = len(obs)
        eps = args.clip_ratio
        rudder_dist, _ = acmodel(obs)

        #print(actions[:,0])
        # print("actions", actions)
        logp = rudder_dist.log_prob(actions)
        #print(torch.exp(logp).sum())
        #print(old_logp)

        for t in range(T):
            if advantages[t] >= 0:
              g = (1+eps)*advantages[t]
            else:
              g = (1-eps)*advantages[t]

            policy_loss -= torch.min(g, logp[t].exp()/old_logp[t].exp()*advantages[t])

        # Add entropy
        entropy = rudder_dist.entropy()
        policy_loss -= args.entropy_coef*entropy.sum()

        # Normlaizes
        policy_loss = policy_loss/T
        # KL oldprobs / new probs
        for t in range(T):
          r = (logp[t] -old_logp[t]).exp() # Division by zero 
          #print("logp: ", logp[t])
          #print("old_logp: ", old_logp[t])
          approx_kl += (r-1) - r.log()


        return policy_loss, approx_kl

    def _compute_value_loss(obs, returns):

        _, values = acmodel(obs)
        value_loss = F.mse_loss(values.squeeze(),returns)

        return value_loss

    rudder_dist, _ = acmodel(sb['obs'])
    old_logp = rudder_dist.log_prob(sb['action']).detach()

    advantage = sb['advantage_gae'] if args.use_gae else sb['advantage']

    policy_loss, _ = _compute_policy_loss_ppo(sb['obs'], old_logp, sb['action'], advantage)
    value_loss = _compute_value_loss(sb['obs'], sb['discounted_reward'])

    for i in range(args.train_ac_iters):
        optimizer.zero_grad()
        loss_pi, approx_kl = _compute_policy_loss_ppo(sb['obs'], old_logp, sb['action'], advantage)
        loss_v = _compute_value_loss(sb['obs'], sb['discounted_reward'])

        loss = loss_v + loss_pi
        #print(approx_kl)
        #print(args.target_kl)
        if approx_kl > args.target_kl:
            break

        loss.backward(retain_graph=True)
        optimizer.step()

    update_policy_loss = policy_loss.item()
    update_value_loss = value_loss.item()

    logs = {
        "policy_loss": update_policy_loss,
        "value_loss": update_value_loss,
    }

    return logs

env_param = dict(x_goal=2, y_goal=2, max_steps=1000)

args = Config(use_critic=True, use_gae=True)
df_ppo, exps = run_experiment(args, update_parameters_ppo, env_param)


x = exps['obs'][:, 0]
y = exps['obs'][:, 1]
x_goal = 2
y_goal = 2

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o')  # Plot points with markers
for i, (x_i, y_i) in enumerate(zip(x, y)):
    plt.text(x_i, y_i, str(i), fontsize=12, ha='right')  # Annotate each point with its index

plt.scatter(x_goal, y_goal, color='red', s=100, edgecolor='black', label='Goal Point')  # s is the size of the marker

plt.title('Evolution of Position in 2D Space')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.show()


