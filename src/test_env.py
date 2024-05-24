from Env import *

# Test fluid dynamics environment

agent = FluidMechanicsEnv(a=0.5, # range 0.1, 0.5, 1, 2, 5
                        T=10, # wave period, range 10 to 20
                        k=0.1, #wave number m^-1: 0.05 to 0.5
                        Ux=1, #wind x component: -2 to 2
                        Uy=1, 
                        alpha=1, # vertical wind decay: around 1
                        sigma=0.1, # noise wind parameter: around 10% wind speed
                        x_goal=4, 
                        y_goal=4, 
                        pos0=np.array([0, 0, 0]), 
                        theta0=0,
                        dist_threshold=0.2, 
                        max_steps=200, 
                        ocean=True, # if false: still water env. If true: ocean like env
                        dt=1, # time step. For now keep 1, we could go smaller
                        max_thrust_speed = 2 # Robot's speed at 100% thrust 
                        )
print(agent.pos)
# take user input for the action
thrust = float(input("Enter action thrust:"))
rudder = float(input("Enter action rudder:"))

agent.step([thrust, rudder])
print(agent.pos)
