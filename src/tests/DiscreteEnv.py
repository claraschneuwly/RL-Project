import numpy as np

class Env:
    def __init__(self, x_goal, y_goal, dist_threshold=0.9, max_steps=200):
        # assume constant speed
        self.max_steps = max_steps
        self.dist_threshold = dist_threshold
        self.max_x, self.min_x = abs(x_goal) * 5 , - abs(x_goal) * 5  # agent has drifted too far, admit defeat
        self.max_y, self.min_y = abs(y_goal) * 5 , - abs(y_goal) * 5 # agent has drifted too far, admit defeat
        self.max_z = 100 # is this even useful ??
        self.x_goal, self.y_goal, self.z_goal = x_goal, y_goal, 0 # coordinates of goal
        self.rudder = 0 # Neutral position. can take values in -90 to 90 degres
        self.orientation = 0  # Facing east (0 degrees)
        self.x, self.y, self.z = 0, 0, 0
        self.done = False
        self.goal_reached = False
        self.steps_count = 0
        self.all_actions = []
        self.coords = np.array([self.x, self.y, self.z, self.orientation, self.x_goal, self.y_goal]) # self.rudder, 

        self.state_dim = 6  # x, y, z, orientation, goal x, goal y. Should add later u_swell, u_wind, v_wind, w_swell
        self.action_dim = 1  # rudder angle

    def reset(self):
        self.rudder = 0
        self.orientation = 0  # Facing North (0 degrees)
        self.speed = 1
        self.x, self.y, self.z = 0, 0, 0
        self.done = False
        self.goal_reached = False
        self.steps_count = 0
        self.coords = np.array([self.x, self.y, self.z, self.orientation, self.x_goal, self.y_goal]) # self.rudder,
        return self.coords
    
    def admit_defeat(self, x, y):
        """Returns True if the agent has drifted too far away from goal"""
        if x > self.max_x or x < self.min_x or y > self.max_y or y < self.min_y:
            return True
    
    def success(self, x, y, z):
        """Returns True if x,y,z is near enough goal"""
        #goal_pos = np.array([self.x_goal, self.y_goal, self.z_goal])
        #dist_to_goal = np.linalg.norm(np.array([x,y,z]) - goal_pos)
        dist_to_goal = np.sqrt((self.x - self.x_goal)**2 + (self.y - self.y_goal)**2)
        if  dist_to_goal <= self.dist_threshold:
            return True
        
    def take_action(self):
        """Update coordinates of agent based on angle of rudder"""
        # angle_changes = {0: 90, 0.25: 45, 0.5: 0, 0.75: -45, 1: -90}
        # angle_change = angle_changes[self.action]
        angle_change = self.action.item()

        # Update orientation
        self.orientation = (self.orientation + angle_change) % 360
        # Calculate movement based on new orientation
        radians = np.radians(self.orientation)
        #change = (round(np.cos(radians)), round(np.sin(radians)))
        change = (np.cos(radians), np.sin(radians))
        # if change[0] == 1 and change[0] == 0:
        #     self.orientation = 0  # or 360 degrees
        # elif change[0] == 0 and change[0] == 1:
        #     self.orientation = 90
        # elif change[0] == -1 and change[0] == 0:
        #     self.orientation = 180
        # elif change[0] == 0 and change[0] == -1:
        #     self.orientation = 270
        # elif change[0] == 1 and change[0] == 1:
        #     self.orientation = 45
        # elif change[0] == 1 and change[0] == -1:
        #     self.orientation = -45
        # elif change[0] == -1 and change[0] == 1:
        #     self.orientation = 135
        # elif change[0] == -1 and change[0] == -1:
        #     self.orientation = 225

        self.x += change[0] #* self.speed
        self.y += change[1] #* self.speed

        self.rudder = self.action

        return np.array([self.x, self.y, self.z, self.orientation, self.x_goal, self.y_goal]) #self.rudder,
    
    def get_reward(self):
        # Calculate euclidian dist to goal
        goal_pos = np.array([self.x_goal, self.y_goal, self.z_goal])
        dist_to_goal = np.linalg.norm(np.array(self.coords[:3]) - goal_pos)
        # Calculate direction to goal in degrees
   
        reward = -dist_to_goal

        if dist_to_goal <= self.dist_threshold:
            reward += 10
        return reward #- orientation_diff / 90 

    def step(self, action):
        """Takes a step. 
        Input: Action = angle of the rudder (may add speed later)
        """
        self.action = action
        self.reward = self.get_reward()
        self.coords = self.take_action()
        self.steps_count += 1
        self.all_actions += [action]

        if self.success(self.x, self.y, self.z):
            self.done = True
            self.goal_reached = True
            return self.coords, self.reward, self.done, self.steps_count, self.all_actions
        elif self.admit_defeat(self.x, self.y) or self.steps_count > self.max_steps:
            self.done = True
            return self.coords, self.reward, self.done, self.steps_count, self.all_actions

        return self.coords, self.reward, self.done, self.steps_count, self.all_actions
    
    def visualise(self):
        """Visualize the current state, will do later"""
        pass
        
#test_agent = Env(x_goal=3, y_goal=3, max_steps=10)


# Do simple tests to check if code works 

# print(test_agent.coords)
# a = test_agent.step(0)
# print(test_agent.coords)
# a = test_agent.step(1)
# print(test_agent.coords)
# a = test_agent.step(1)
# print(test_agent.coords)
