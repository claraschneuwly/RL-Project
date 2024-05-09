import numpy as np
import matplotlib.pyplot as plt

waves = np.arange(0, 5.5, 0.5)

convergence_steps_TD3 = {0.0: 30702, 0.5: 50653, 1.0: 60048, 1.5: 45232, 2.0: 65221, 2.5: 54927, 3.0: 55901, 3.5: 53453, 4.0: 84274, 4.5: 57965, 5.0: 48086}
convergence_steps_SAC = {}

plt.plot(waves, list(convergence_steps_SAC.values()), label="SAC", color="green")
plt.plot(waves, list(convergence_steps_TD3.values()), label="TD3", color="orange")

#plt.title('Convergence Time of Algorithms vs. Wave Amplitude in Policy Training')
plt.xlabel('Wave Amplitude')
plt.ylabel('Steps')

plt.legend()
plt.show()

