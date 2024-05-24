from DiscreteEnv import *
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


class QLearning:
    def __init__(self, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.Q_values_dict = {}  # Q-value table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = np.arange(-90,90,45)  # All possible actions

    def choose_action(self, state):
        state = tuple(state)
        if random.random() < self.epsilon:
            #print("explore")
            # Explore: choose a random action
            action = random.choice(self.actions)
        else:
            # Exploit: choose the best action based on current Q-values
            q_values = self.Q_values_dict.get(state, {})
            if q_values:
                #print("print ")
                action = max(q_values, key=q_values.get)
                #print(f"action chosen: {action}, q_value= {max(q_values)}")
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
            next_state, reward, done, steps_count, all_actions = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            #print(f"Episode {episode+1}: Total reward = {total_reward}, coords: {state}")
        total_success = 1 if env.goal_reached else 0
        history_reward.append(total_reward)
        history_success.append(total_success)
        #print(f"Episode {episode+1}: Total reward = {total_reward}, coords: {state}")
    return history_reward, history_success

# Setup the environment
x_goal, y_goal= 5, 4
env = Env(x_goal, y_goal, max_steps=200)
agent = QLearning()

# Run the Q-learning algorithm
history_reward, history_success = run_q_learning(env, agent, episodes=500)

#print(f"Total reward = {history_reward}")
print(f"Success : {history_success}")

import matplotlib.pyplot as plt


def plot_cumulative_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Cumulative Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Episodes')
    plt.legend()
    plt.show()

# # Example usage:
rewards = [sum(history_reward[i:i+10]) for i in range(0, len(history_reward))]  # Assuming reward_list is a list of rewards per episode
plot_cumulative_rewards(rewards)

def plot_success_rate(successes, window_size=50):
    """
    Plots the success rate over time using a rolling window average.
    
    Parameters:
        successes (list of int): List where each element is 0 (failure) or 1 (success) for each episode.
        window_size (int): The number of episodes to include in the rolling average.
    """
    cumulative_success = [sum(successes[max(0, i-window_size):i+1]) / (i - max(0, i-window_size) + 1) 
                          for i in range(len(successes))]
    
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_success, label='Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Over Episodes')
    plt.legend()
    plt.show()

plot_success_rate(history_success, window_size=5)
def run_trajectory(agent):
    x = []
    y = []
    state = env.reset()
    x.append(state[0])
    y.append(state[1])
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, steps_count, all_actions = env.step(action)
        state = next_state
        x.append(state[0])
        y.append(state[1])
        
    return x, y

x, y = run_trajectory(agent)

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o')  # Plot points with markers
#for i, (x_i, y_i) in enumerate(zip(x, y)):
#    plt.text(x_i, y_i, str(i), fontsize=12, ha='right')  # Annotate each point with its index

plt.scatter(x_goal, y_goal, color='red', s=100, edgecolor='black', label='Goal Point')  # s is the size of the marker

plt.title('Evolution of Position in 2D Space')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.show()

x, y = run_trajectory(agent)

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o')  # Plot points with markers
#for i, (x_i, y_i) in enumerate(zip(x, y)):
#    plt.text(x_i, y_i, str(i), fontsize=12, ha='right')  # Annotate each point with its index

plt.scatter(x_goal, y_goal, color='red', s=100, edgecolor='black', label='Goal Point')  # s is the size of the marker

plt.title('Evolution of Position in 2D Space')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.show()




