seed = 0
import numpy as np
import random
import math
from tqdm.notebook import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

"""## Fluid Environment"""

def angle_between_vectors(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if magnitude_v1 != 0 and magnitude_v2 != 0:
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        if cos_theta > 1:
                cos_theta = 1
        elif cos_theta < -1:
            cos_theta = -1
        angle_radians = math.acos(cos_theta)
        return angle_radians
    else:
        return 0

def normalize_vector_from_point(x, y, x0, y0):
    # Calcul des composantes du vecteur
    vector_x = x - x0
    vector_y = y - y0

    # Calcul de la magnitude
    magnitude = math.sqrt(vector_x**2 + vector_y**2)

    # Normalisation
    if magnitude != 0:
        normalized_x = vector_x / magnitude
        normalized_y = vector_y / magnitude
        return (normalized_x, normalized_y)
    else:
        return (0, 0)

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
        self.dir_goal = normalize_vector_from_point(self.x_goal, self.y_goal, 0, 0)
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

        self.state_dim = 6  # x, y, z. Should add later u_swell, u_wind, v_wind, w_swell
        self.action_dim = 2  # thrust, rudder angle
        self.max_action = torch.tensor([1, np.pi/4])

        self.action_space_high = np.array([1, np.pi/4])  # Upper bounds for each action dimension
        self.action_space_low = np.array([0, -np.pi/4])
    
    def action_space_sample(self):
        return np.random.uniform(self.action_space_low, self.action_space_high)

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
        #print(self.thrust)

        ## new action categorical
        '''
        if action == 0:
            self.rudder = 0
        elif action == 1:
            self.rudder = -np.pi/4
        elif action == 2:
            self.rudder = np.pi/4

        self.thrust = 0.5
        '''

        # Find the water velocity at agent position
        x, y, z = self.pos
        u, v, w = self.water_speed(self.pos)
        #print([u, v, w])
        #self.vel = np.array([u, v, w])

        # Add inertia to the agent's velocity
        self.vel += self.inertia()

        # Perform agent action
        self.theta += self.rudder
        self.theta %= (2*np.pi) # Update agent's orientation from rudder angle
        u_action = self.thrust * np.sin(self.theta)
        v_action = self.thrust * np.cos(self.theta)
        self.vel = np.array([u_action, v_action, 0])
        #print("theta", self.theta)
        #print("vitesse", [u_action, v_action])

        # Update velocity history
        self.u_history.append(u)
        self.v_history.append(v)

        # Update agent position
        x += self.vel[0]
        y += self.vel[1]
        #print("new pos", [x, y])
        z = self.water_surface_level((x, y, z))
        self.dir_goal = normalize_vector_from_point(self.x_goal, self.y_goal, x, y)

        return np.array([x, y, z])

    def get_reward(self):

        # Calculate euclidian dist to goal Without z coord
        goal_pos = np.array([self.x_goal, self.y_goal])
        dist_to_goal = np.linalg.norm(np.array(self.pos[:2]) - goal_pos)
        dist_to_dir = angle_between_vectors(self.dir_goal, (np.sin(self.theta), np.cos(self.theta)))/np.pi
        ##reward = - (dist_to_goal/100 + np.float64(dist_to_dir))/50
        reward = - (dist_to_goal/5000 + (np.exp((1 + np.float64(dist_to_dir))) - 1)/200)
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

        return np.concatenate((self.pos, np.array([self.theta]), np.array([self.x_goal, self.y_goal]))), self.reward, self.sum_reward, self.done, self.steps_count, self.all_actions

    def reset(self):

        self.rudder = 0
        self.thrust = 0
        self.pos = np.array([0, 0, 0])
        self.done = False
        self.goal_reached = False
        self.steps_count = 0
        self.sum_reward = 0
        self.theta = 0
        self.dir_goal = normalize_vector_from_point(self.x_goal, self.y_goal, 0, 0)

        return np.concatenate((self.pos, np.array([self.theta]), np.array([self.x_goal, self.y_goal])))

env = FluidMechanicsEnv(a=0,
                        T=1,
                        k=0.1,
                        Ux=0,
                        Uy=0,
                        alpha=1,
                        sigma=0,
                        x_goal=4,
                        y_goal=4,
                        pos0=np.array([0, 0, 0]),
                        theta0=0,
                        dist_threshold=0.2,
                        max_steps=200)

# action = np.array([1, -np.pi/4])
# env.step(action)

# action = np.array(0)
# env.step(action)

# env.reset()
