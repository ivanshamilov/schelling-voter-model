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
  
  N = 10 # agents
  L = 5  # lattice dimension
  tolerance = 0.5 # tolerance
  s = 0.3 # prestige of language
  prob_migrate = 0.5 # possibility to migrate 


  p = N / L ** 2
  lattice = initialize(N, p)
  print(lattice)
  simulation(lattice, tolerance, prob_migrate)