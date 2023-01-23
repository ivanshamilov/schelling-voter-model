import numpy as np
import matplotlib.pyplot as plt
from logic import *

'''
img = plt.imshow(lattice, cmap="plasma")
plt.xlabel("L [i]")
plt.ylabel("L [j]")
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

  # N = 50
  # p = N / L ** 2

  L = 10  # lattice dimension
  density = 0.5 # formula = N / L ** 2 -> N (number of agents) can be estimated
  tolerance = 0.3 # tolerance
  prestige = 0.3 # prestige of language
  prob_migrate = 0.5 # possibility to migrate 


  lattice = initialize(L, density)
  print(lattice)
  simulation(lattice, tolerance, prob_migrate, prestige)
  print(lattice)