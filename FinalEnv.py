import numpy as np
import random
from tqdm.notebook import tqdm
import pandas as pd
import torch.nn.functional as F

"""## Fluid Environment"""

class FluidMechanicsEnv:

    class Wave:
        def __init__(self, a, T, k) :
            self.a = a                  # Wave amplitude
            self.T = T                  # Wave period
            self.omega = 2 * np.pi / T  # Wave frequency
            self.k = .1

    class Wind:
        def __init__(self, Ux, Uy, alpha, sigma) :
            self.Ux = Ux                  # Wave amplitude
            self.Uy = Uy                  # Wave period
            self.alpha = alpha            # Wave frequency
            self.sigma = sigma

    def __init__(self, a, T, k,  Ux, Uy, alpha, sigma, x_goal, y_goal, pos0, theta0, dist_threshold=0.1, max_steps=1000):
        self.t = 0
        self.wave = self.Wave(a, T, k)
        self.wind = self.Wind(Ux, Uy, alpha, sigma)
        self.max_steps = max_steps
        self.dist_threshold = dist_threshold
        self.max_x, self.min_x = 100 , -100  # agent has drifted too far, admit defeat
        self.max_y, self.min_y = 100 , -100 # agent has drifted too far, admit defeat
        self.x_goal, self.y_goal, self.z_goal = x_goal, y_goal, 0 # coordinates of goal
        self.done = False
        self.goal_reached = False
        self.steps_count = 0
        self.sum_reward = 0
        self.all_actions = []
        self.pos = pos0
        self.theta = theta0
        self.vel = np.array([0, 0, 0]).astype(np.float32)
        self.thrust = 0 # [0; 1]
        self.rudder = 0.0 # [-pi/4; pi/4]
        self.action = np.array([0, 0])
        self.u_history = []
        self.v_history = []

        self.state_dim = 3  # x, y, z. Should add later u_swell, u_wind, v_wind, w_swell
        self.action_dim = 2  # thrust, rudder angle

    def water_surface_level(self, pos) :
        x, _, _ = pos
        eta = self.wave.a * np.sin(self.wave.omega * self.t - self.wave.k * x)
        return eta

    def water_speed(self, pos) :
        x, y, z = pos
        eta = self.water_surface_level(pos)

        u_swell = self.wave.a * self.wave.omega * np.exp(self.wave.k * z) * np.sin(self.wave.omega * self.t - self.wave.k * x)
        w_swell = self.wave.a * self.wave.omega * np.exp(self.wave.k * z) * np.cos(self.wave.omega * self.t - self.wave.k * x)

        u_wind = np.random.normal(self.wind.Ux, self.wind.sigma) * np.exp(-self.wind.alpha * (eta - z))
        v_wind = np.random.normal(self.wind.Uy, self.wind.sigma) * np.exp(-self.wind.alpha * (eta - z))

        # u = u + np.random.normal(0, noise, u.shape)
        # v = v + np.random.normal(0, noise, v.shape)
        # w = w + np.random.normal(0, noise, w.shape)

        return u_swell + u_wind, v_wind, w_swell

    def inertia(self, lag = 5) :

        if len(self.u_history) > 0 :

            k = np.minimum(lag, len(self.u_history))
            coefs = np.array([1 / (4 ** (i + 1)) for i in reversed(range(k))])
            u = (self.u_history[-k:] * coefs).sum() / coefs.sum()
            v = (self.v_history[-k:] * coefs).sum() / coefs.sum()

        else :
            u, v = 0, 0

        return np.array([u, v, 0])

    def update_pos(self, action):
        # Sets agent action
        self.thrust = action[0]
        self.rudder = action[1]

        # Find the water velocity at agent position
        x, y, z = self.pos
        u, v, w = self.water_speed(self.pos)
        self.vel = np.array([u, v, w])

        # Add inertia to the agent's velocity
        self.vel += self.inertia()

        # Perform agent action
        self.theta -= self.rudder # Update agent's orientation from rudder angle
        u_action = self.thrust * np.sin(self.theta)
        v_action = self.thrust * np.cos(self.theta)
        self.vel += np.array([u_action, v_action, 0])

        # Update velocity history
        self.u_history.append(u)
        self.v_history.append(v)

        # Update agent position
        x += self.vel[0]
        y += self.vel[1]
        z = self.water_surface_level((x, y, z))

        return np.array([x, y, z])

    def get_reward(self):

        # Calculate euclidian dist to goal Without z coord
        goal_pos = np.array([self.x_goal, self.y_goal])
        dist_to_goal = np.linalg.norm(np.array(self.pos[:2]) - goal_pos)
        reward = - dist_to_goal
        if dist_to_goal <= self.dist_threshold:
            reward += 10

        return reward

    def success(self):
        """Returns True if x,y is near enough goal"""
        goal_pos = np.array([self.x_goal, self.y_goal])
        dist_to_goal = np.linalg.norm(np.array(self.pos[:2]) - goal_pos)
        if  dist_to_goal <= self.dist_threshold:
            return True
        else:
            return False

    def admit_defeat(self):
        """Returns True if the agent has drifted too far away from goal"""
        if self.pos[0] > self.max_x or self.pos[0] < self.min_x or self.pos[1] > self.max_y or self.pos[1] < self.min_y:
            return True
        else:
            return False

    def step(self, action) :

        self.pos = self.update_pos(action)
        self.reward = self.get_reward()
        self.sum_reward += self.reward
        self.steps_count += 1
        self.all_actions += [action]

        if self.success():
            self.done = True
            self.goal_reached = True

        elif self.admit_defeat() or self.steps_count > self.max_steps:
            self.done = True

        return self.pos, self.reward, self.sum_reward, self.done, self.steps_count, self.all_actions

    def reset(self):

        self.rudder = 0
        self.thrust = 0
        self.pos = np.array([0, 0, 0])
        self.done = False
        self.goal_reached = False
        self.steps_count = 0
        self.sum_reward = 0

        return self.pos

env = FluidMechanicsEnv(a=0,
                        T=1,
                        k=0.1,
                        Ux=0,
                        Uy=0,
                        alpha=1,
                        sigma=0,
                        x_goal=1,
                        y_goal=1,
                        pos0=np.array([0, 0, 0]),
                        theta0=0,
                        dist_threshold=0.1,
                        max_steps=1000)

### Test the environment

# action = np.array([1, -np.pi/4])
# env.step(action)
# print(env.pos, env.theta)

# action = np.array([.4, 0])
# env.step(action)
# print(env.pos, env.theta)

# env.reset()
# print(env.pos, env.theta)