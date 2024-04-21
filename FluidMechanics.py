import numpy as np


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

    def __init__(self, a, T, k,  Ux, Uy, alpha, sigma):

        self.t = 0
        self.wave = self.Wave(a, T, k)
        self.wind = self.Wind(Ux, Uy, alpha, sigma)

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

    def step(self) :

        self.t += 1


class Agent :

    def __init__(self, pos0, theta0 = 0) :
        
        self.pos = pos0
        self.theta = theta0
        self.vel = np.array([0, 0, 0]).astype(np.float32)
        self.thrust = 0 # [0; 1]
        self.rudder = 0 # [-pi/4; pi/4]
        self.action = np.array([0, 0])
        self.u_history = []
        self.v_history = []

    def inertia(self, lag = 5) :

        if len(self.u_history) > 0 :

            k = np.minimum(lag, len(self.u_history))
            coefs = np.array([1 / (4 ** (i + 1)) for i in reversed(range(k))])
            u = (self.u_history[-k:] * coefs).sum() / coefs.sum()
            v = (self.v_history[-k:] * coefs).sum() / coefs.sum()

        else :
            u, v = 0, 0

        return np.array([u, v, 0])

    def take_action(self, action) :

        # Sets agent action
        self.thrust = action[0]
        self.rudder = action[1]


    def step(self, env) :

        # Find the water velocity at agent position
        x, y, z = self.pos
        u, v, w = env.water_speed(self.pos)
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
        z = env.water_surface_level((x, y, z))
        self.pos = np.array([x, y, z])