import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logic import *

'''
img = plt.imshow(lattice, cmap="plasma")
plt.xlabel("L [i]")
plt.ylabel("L [j]")
plt.show()


lattices = np.zeros((10, L, L))
lattices[i] = lattice

fig, axes = plt.subplots(2, 5, figsize=(10, 10))

for lat, ax in zip(lattices, axes.flatten()):
  ax.matshow(lat, cmap="plasma")
plt.show()


'''

if __name__ == "__main__":
  
  '''
  Paper values 
  L = 50
  s = 0.5
  tolerance = 0.3
  prob_migrate = np.arange(0.02, 0.98, 0.04) # 24 steps
  p_density = np.arange(0.02, 0.98, 0.04) # 24 steps
  '''

  L = 50  # lattice dimension
  density = 0.9 # formula = N / L ** 2 -> N (number of agents) can be estimated
  tolerance = 0.3 # tolerance
  prestige = 0.3 # prestige of language
  prob_migrate = 0.2  # possibility to migrate 

  num_iters = 30
  probs = np.arange(0, 1, 0.1)
  simulation_results = np.zeros(probs.shape[0])
  
  for i, prob_mig in enumerate(probs):
    sim_res = 0
    for j in range(num_iters):
      print(f"Iteration {j + 1} out of {num_iters}")
      lattice = initialize(L, density)
      lattice = simulation(lattice, tolerance, prob_mig, prestige)
      res = fraction_minority_language(lattice)
      print(f"<S>={res}")
      sim_res += res

    simulation_results[i] = sim_res / num_iters


  plt.plot(probs, simulation_results)
  plt.ylim([-0.1, 1.1])
  plt.show()