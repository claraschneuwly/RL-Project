## Env augmented to fit the SAC req
# Added some action and obs boxes


import numpy as np 
import math
from gym import spaces

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
            self.k = k   

    class Wind:
        def __init__(self, Ux, Uy, alpha, sigma) :
            self.Ux = Ux                  # Wave amplitude
            self.Uy = Uy                  # Wave period
            self.alpha = alpha            # Wave frequency
            self.sigma = sigma

    def __init__(self, a, T, k,  Ux, Uy, alpha, sigma, x_goal, y_goal, pos0, theta0, dist_threshold=0.1, max_steps=1000, ocean=False, dt=1, max_thrust_speed=1):
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
        self.straight = False
        self.alpha = 0.1
        self.ocean = ocean
        if self.ocean:
            self.state_dim = 9 # u_water, v_water
        else:
            self.state_dim = 7  # x, y, z. 
        self.action_dim = 2  # thrust, rudder angle
        self.dt = dt
        self.max_thrust_speed = max_thrust_speed
        self.init_dist = np.linalg.norm(np.array(self.pos[:2]) - np.array([self.x_goal, self.y_goal]))

        if self.ocean:
            self.observation_space = spaces.Box(low=-100, high=100, shape=(9,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-100, high=100, shape=(7,), dtype=np.float32)
        
        self.action_space = spaces.Box(low=np.array([0, -np.pi/4]), high=np.array([max_thrust_speed, np.pi/4]), dtype=np.float32)

    def water_surface_level(self, pos) :
        x, _, _ = pos
        eta = self.wave.a * np.sin(self.wave.omega * self.t - self.wave.k * x)
        return eta

    def water_speed(self, pos) :
        x, y, z = pos
        eta = self.water_surface_level(pos)

        u_swell = self.wave.a * self.wave.omega * np.exp(self.wave.k * z) * np.sin(self.wave.omega * self.t - self.wave.k * x)
        w_swell = self.wave.a * self.wave.omega * np.exp(self.wave.k * z) * np.cos(self.wave.omega * self.t - self.wave.k * x)
        
        u_wind = np.random.normal(self.wind.Ux, self.wind.sigma) * np.exp(self.wind.alpha * (z-self.wave.a))
        v_wind = np.random.normal(self.wind.Uy, self.wind.sigma) * np.exp(self.wind.alpha * (z-self.wave.a))

        # u = u + np.random.normal(0, noise, u.shape)
        # v = v + np.random.normal(0, noise, v.shape)
        # w = w + np.random.normal(0, noise, w.shape)

        return u_swell + u_wind, v_wind, w_swell

    def inertia(self, lag = 3):

        if len(self.u_history) > 0 :
            k = np.minimum(lag, len(self.u_history))
            coefs = np.array([1 / (4 ** (i + 1)) for i in reversed(range(k))])
            u = (self.u_history[-k:] * coefs).sum()
            v = (self.v_history[-k:] * coefs).sum()

        else :
            u, v = 0, 0

        return np.array([u, v, 0])
    
    def update_pos(self, action):
        # Sets agent action
        self.thrust = action[0]*self.max_thrust_speed
        self.rudder = action[1]
    
        # Find the water velocity at agent position
        x, y, z = self.pos
        u, v, w = self.water_speed(self.pos)
        self.vel = np.array([u, v, w])

        # Add inertia to the agent's velocity
        self.vel += self.inertia()

        # Perform agent action
        self.theta += self.rudder 
        self.theta %= (2*np.pi) # Update agent's orientation from rudder angle
        u_action = self.thrust * np.cos(self.theta)
        v_action = self.thrust * np.sin(self.theta)
        self.vel += np.array([u_action, v_action, 0])

        # Update velocity history
        self.u_history.append(u)
        self.v_history.append(v)

        # Update agent position
        x += self.vel[0]*self.dt
        y += self.vel[1]*self.dt
        z = self.water_surface_level((x, y, z))

        # Lucas' alignement checks (Lucas double check stp)
        self.dir_goal = normalize_vector_from_point(self.x_goal, self.y_goal, x, y)
        if not self.straight and angle_between_vectors(self.dir_goal, (np.sin(self.theta), np.cos(self.theta))) < 2*self.alpha:
            self.straight = True

        return np.array([x, y, z])
    
    def get_reward(self):
        
        # Calculate euclidian dist to goal Without z coord
        goal_pos = np.array([self.x_goal, self.y_goal])
        dist_to_goal = np.linalg.norm(np.array(self.pos[:2]) - goal_pos)
        dist_to_dir = angle_between_vectors(self.dir_goal, (np.sin(self.theta), np.cos(self.theta)))/np.pi
        
        reward = 0
        reward -= (dist_to_goal)/(self.max_steps*self.init_dist)
        reward -= (np.exp((np.float64(dist_to_dir))) - 1)/self.max_steps

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
        self.t += self.dt
        if self.success():
            self.done = True
            self.goal_reached = True

        elif self.admit_defeat() or self.steps_count > self.max_steps:
            self.done = True
        
        if self.ocean: 
            # TODO: Add noise to observation if algo works fine with perfect estimate of wave and winds speed

            # TODO: separate wind and water component

            u_water, v_water, _ = self.water_speed(self.pos)
            return np.concatenate((self.pos, np.array([np.cos(self.theta), np.sin(self.theta)]),  np.array([self.x_goal, self.y_goal]), np.array([u_water, v_water]))), self.reward, self.sum_reward, self.done, self.steps_count, self.all_actions
        
        else:
            return np.concatenate((self.pos, np.array([np.cos(self.theta), np.sin(self.theta)]),  np.array([self.x_goal, self.y_goal]))), self.reward, self.sum_reward, self.done, self.steps_count, self.all_actions
        
    def reset(self):

        self.rudder = 0
        self.thrust = 0  
        self.pos = np.array([0, 0, 0])
        self.done = False
        self.goal_reached = False
        self.steps_count = 0
        self.sum_reward = 0
        self.t = 0
        # Statement: by randomly initilizing theta, we learn beter how to adjust/turn round to reach goal
        self.theta = np.random.uniform(0, 2*np.pi) # if statement false: self.theta = 0
        self.dir_goal = normalize_vector_from_point(self.x_goal, self.y_goal, 0, 0)
        self.straight = False

        if self.ocean: 
            # TODO: Add noise to observation if algo works fine with perfect estimate of wave and winds speed

            # TODO: separate wind and water component

            u_water, v_water, _ = self.water_speed(self.pos)
            return np.concatenate((self.pos, np.array([np.cos(self.theta), np.sin(self.theta)]),  np.array([self.x_goal, self.y_goal]), np.array([u_water, v_water])))
        else:
            return np.concatenate((self.pos, np.array([np.cos(self.theta), np.sin(self.theta)]),  np.array([self.x_goal, self.y_goal])))


# env = FluidMechanicsEnv(a=0.1, # range 0.1, 0.5, 1, 2, 5
#                         T=1, # wave period, range 10 to 20
#                         k=0.1, #wave number m^-1: 0.05 to 0.5
#                         Ux=0, #wind x component: -2 to 2
#                         Uy=0, 
#                         alpha=1, # vertical wind decay: around 1
#                         sigma=0, # noise wind parameter: around 10% wind speed
#                         x_goal=4, 
#                         y_goal=4, 
#                         pos0=np.array([0, 0, 0]), 
#                         theta0=0,
#                         dist_threshold=0.2, 
#                         max_steps=200, 
#                         ocean=False, # if false: still water env. If true: ocean like env
#                         dt=1, # time step. For now keep 1, we could go smaller
#                         max_thrust_speed = 1# Robot's speed at 100% thrust 
#                         )