from Env import *

# Test fluid dynamics environment

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
