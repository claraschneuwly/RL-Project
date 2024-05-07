from RL_env import *

env = Env(x_goal=2, y_goal=2, max_steps=10)
print(env.coords)
# take user input for the action
action = float(input("Enter action:"))

coords, _, sum_reward, done, steps_count, actions = env.step(action)
print(env.coords)
# repeat until agent is done
while not done:
    action = float(input("Enter action:"))
    coords, _, sum_reward, done, steps_count, actions = env.step(action)
    if env.goal_reached:
        print("Success! goal reached by agent")
    print(env.coords)
