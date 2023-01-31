import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logic import *


def fig1_simulation():
  migrations = np.linspace(0, 1, 51)
  simulation_results = np.zeros(migrations.shape[0])
  
  for i, prob_migrate in enumerate(migrations):
    sim_res = 0
    for j in range(num_iters):
      print(f"Iteration {j + 1} out of {num_iters} for {prob_migrate}")
      lattice = initialize(L, density)
      lattice = simulation(lattice, tolerance, prob_migrate, prestige)
      print(count_speakers(lattice))
      res = fraction_minority_language(lattice)
      print(f"<S>={res}")
      sim_res += res

    simulation_results[i] = sim_res / num_iters

  save_pickle(f"fig1_density{density}.pickle", simulation_results)


def fig2_simulation():
  densities = np.linspace(1, 0, 51)
  simulation_results = np.zeros(densities.shape[0])
  
  for i, density in enumerate(densities):
    sim_res = 0
    for j in range(num_iters):
      print(f"Iteration {j + 1} out of {num_iters} for {prob_migrate}")
      lattice = initialize(L, density)
      lattice = simulation(lattice, tolerance, prob_migrate, prestige)
      print(count_speakers(lattice))
      res = fraction_minority_language(lattice)
      print(f"<S>={res}")
      sim_res += res

    simulation_results[i] = sim_res / num_iters

  save_pickle(f"fig2_migration{prob_migrate}.pickle", simulation_results)

def fig3_simulation():
  num_iters = 200

  lattice = initialize(L, density) 
  ones = np.zeros(num_iters)
  minus_ones = np.zeros(num_iters)
  ones[0] = empty_places_for(lattice, language=1, tolerance=tolerance)
  minus_ones[0] = empty_places_for(lattice, language=-1, tolerance=tolerance)
  
  for i in range(1, num_iters):
    lattice = simulation(lattice, tolerance, prob_migrate, prestige)
    ones[i] = empty_places_for(lattice, language=1, tolerance=tolerance)
    minus_ones[i] = empty_places_for(lattice, language=-1, tolerance=tolerance)
  
  save_pickle(f"fig3_migrate{prob_migrate}_1.pickle", ones)
  save_pickle(f"fig3_migrate{prob_migrate}_-1.pickle", minus_ones)

def fig4_simulation():
  # create phase matrix, columns -> densities (1 - ro), rows -> prob. migrate (p)
  prob_migrates = np.linspace(1, 0, 51)
  densities = np.linspace(1, 0, 51)
  results = np.zeros((prob_migrates.shape[0], densities.shape[0]))
  num_iters = 30

  for i, prob_migrate in enumerate(prob_migrates):
    for j, density in enumerate(densities):
      sim_res = 0
      for idx in range(num_iters):
        print(f"migrate: {prob_migrate}, density: {density}, iter: {idx + 1} / {num_iters}")
        lattice = initialize(L, density)
        lattice = simulation(lattice, tolerance, prob_migrate, prestige)
        res = fraction_minority_language(lattice)
        print(f"<S>={res}")
        sim_res += res
      sim_res /= num_iters

      results[i][j] = sim_res
  
  save_pickle(f"fig4_all_data.pickle", results)


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
  density = 0.94 # formula = N / L ** 2 -> N (number of agents) can be estimated
  tolerance = 0.3 # tolerance
  prestige = 0.3 # prestige of language
  prob_migrate = 0.1 # possibility to migrate 

  # fig1_simulation()
  # fig2_simulation()
  # fig3_simulation()
  fig4_simulation()

  # Fig1

  # res1 = load_pickle(f"fig1_density0.9.pickle")
  # res2 = load_pickle(f"fig1_density0.7.pickle")
  # res3 = load_pickle(f"fig1_density0.5.pickle")
  # res4 = load_pickle(f"fig1_density0.1.pickle")

  # create_graphs(results=[res1, res2, res3, res4],
  #               xaxis=migrations,
  #               legend_titles=["1 - p = 0.1", "1 - p = 0.3", "1 - p = 0.5", "1 - p = 0.1"],
  #               xlabel="p",
  #               ylabel=r"$<S>$",
  #               title="test")

  # Fig2

  # res1 = load_pickle(f"fig2_migration0.1.pickle")
  # res2 = load_pickle(f"fig2_migration0.3.pickle")
  # res3 = load_pickle(f"fig2_migration0.5.pickle")
  # res4 = load_pickle(f"fig2_migration0.9.pickle")
  # create_graphs(results=[res1[::-1], res2[::-1], res3[::-1], res4[::-1]],
  #               xaxis=densities, 
  #               legend_titles=["p = 0.1", "p = 0.3", "p = 0.5", "p = 0.9"],
  #               xlabel=r"$1 - \rho$", 
  #               ylabel=r"$<S>$", 
  #               title="Size of the population speaking the minority language (S) as a function of 1 - ro")

  # Figs 3

  # pm = 0.8
  # ones = load_pickle(f"fig3/fig3_migrate{pm}_1.pickle")
  # minus_ones = load_pickle(f"fig3/fig3_migrate{pm}_-1.pickle")
  # plot_scatters(results=[ones, minus_ones], 
  #               xaxis=ones.shape[0], 
  #               ylim=max(max(ones), max(minus_ones)) + 10, 
  #               markers=["*", "+"], 
  #               xlabel="t", 
  #               ylabel="available empty spaces", 
  #               title=f"p = {pm}")
