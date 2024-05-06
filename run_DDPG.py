import torch
import DDPG
import DDPG_main
import FinalEnv
from utils import *

env = FinalEnv.FluidMechanicsEnv(a=0.1, # range 0.1, 0.5, 1, 2, 5
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

args = DDPG_main.Config()

kwargs={"state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "max_action": env.max_action,
        "discount": args.discount,
        "tau": args.tau}

#### Train and save the actor model
policy, _, _, _, _ = DDPG_main.run_DDPG(env, kwargs, seed=0)
torch.save(policy.actor.state_dict(), "DDPG_policy")


#### Load the actor model 
model = DDPG.DDPG(**kwargs)
actor = model.actor

actor.load_state_dict(torch.load("DDPG_policy"))
actor.eval()


#### Sample a trajectory 
trajectory, total_reward = sample_trajectory(env, actor, env.max_steps)

print("Total reward from sampled trajectory:", total_reward)
plot_trajectory(trajectory)