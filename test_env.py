from RL_env import *
from FluidMechanics import *

test = int(input("Enter 0 to test FluidMechanics, 1 to test DiscreteEnv:"))

if test == 0:
    # Test fluid dynamics env
    env = FluidMechanicsEnv(a=2.5, T=10, k=.1, Ux=1, Uy=1, alpha=1, sigma=.1)
    agent = Agent(pos0=np.array([0, 0, 0]))
    print(agent.pos)
    # take user input for the action
    thrust = float(input("Enter action thrust:"))
    rudder = float(input("Enter action rudder:"))

    agent.take_action([thrust, rudder])
    print(agent.pos)

    agent.step(env)
    print(agent.pos)

else:
    # Test first discrete env
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
