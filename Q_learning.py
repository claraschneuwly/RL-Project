from RL_env import *
import numpy as np
import random

class QLearning:
    def __init__(self, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.Q_values_dict = {}  # Q-value table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = np.arange(0,1,0.25)  # All possible actions

    def choose_action(self, state):
        state = tuple(state)
        if random.random() < self.epsilon:
            print("explore")
            # Explore: choose a random action
            action = random.choice(self.actions)
        else:
            # Exploit: choose the best action based on current Q-values
            q_values = self.Q_values_dict.get(state, {})
            if q_values:
                action = max(q_values, key=q_values.get)
                print(f"action chosen: {action}, q_value= {max(q_values)}")
            else:
                action = random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)
        old_q_value = self.Q_values_dict.get(state, {}).get(action, 0)
        future_q_values = self.Q_values_dict.get(next_state, {})
        max_future_q_value = max(future_q_values.values(), default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q_value - old_q_value)
        
        if state not in self.Q_values_dict:
            self.Q_values_dict[state] = {}
        self.Q_values_dict[state][action] = new_q_value

def run_q_learning(env, agent, episodes):
    history_reward = []
    history_success = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, _, done, steps_count, all_actions = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            print(f"Episode {episode+1}: Total reward = {total_reward}, coords: {state}")
        total_success = 1 if env.goal_reached else 0
        history_reward.append(total_reward)
        history_success.append(total_success)
        #print(f"Episode {episode+1}: Total reward = {total_reward}, coords: {state}")
    return history_reward, history_success

# Setup the environment
env = Env(x_goal=2, y_goal=2, max_steps=100)
actions = [0, 1/4, 1/2, 3/4, 1]
agent = QLearning()

# Run the Q-learning algorithm
history_reward, history_success = run_q_learning(env, agent, episodes=200)

#print(f"Total reward = {history_reward}")
print(f"Success : {history_success}")






