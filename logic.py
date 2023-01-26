import numpy as np
from numba import njit
import random
import matplotlib.pyplot as plt


def plot_lattice(lattice: np.array, show_labels: bool = False):
    """
    Plot lattice with specified colors
    """
    data_3d = np.ndarray(shape=(lattice.shape[0], lattice.shape[1], 3), dtype=int)
    color_map = {0: np.array([255, 255, 255]), # WHITE
                1: np.array([0, 0, 0]), # BLACK
                -1: np.array([255, 0, 0])} # blue 
    for i in range(0, lattice.shape[0]):
        for j in range(0, lattice.shape[1]):
            data_3d[i][j] = color_map[lattice[i][j]]
    fig, ax = plt.subplots(1,1, figsize=(9, 9))
    ax.imshow(data_3d)  
    if show_labels:
        for i in range(0, lattice.shape[0]):
            for j in range(0, lattice.shape[1]):
                c = lattice[j,i]
                ax.text(i, j, str(c), va='center', ha='center')

    plt.show()

def initialize(L: int, p: float = 1.0) -> np.array:
    '''
    Inizalization of the lattice - inserting agents on the lattice
    '''
    lattice = np.array([[np.random.choice([0, 1], p=[1 - p, p]) for _ in range(L)]
                        for _ in range(L)])
    for x, y in zip(*np.where(lattice == 1)):
        lattice[x][y] = np.random.choice([-1, 1]) 
    return lattice

@njit
def find_agents(lattice: np.array):
    """
    Find indexes of all available agents on the lattice (non-zero elements)
    """
    return np.column_stack((np.where(lattice != 0)))

@njit 
def find_empty_spots(lattice: np.array):
    """
    Returns a list of empty spots available for migration
    """
    return np.column_stack((np.where(lattice == 0)))

@njit
def count_speakers(lattice: np.array, agent_language: int = 1):
    """
    Counts the number of speakers of language 1 and -1 
    Return (agents language count, other language count).
    """
    return np.count_nonzero(lattice == agent_language),  \
           np.count_nonzero(lattice == -agent_language)

@njit
def fraction_minority_language(lattice: np.array):
    """
    Calculate the fraction of the minority language
    """
    speakers_1, speakers_2 = count_speakers(lattice)
    return min(speakers_1, speakers_2) / (speakers_1 + speakers_2)

@njit
def find_tolerance(lattice: np.array, i: int, j: int, color: int=0, return_counts: bool=False):
    """
    Count all neighbours and neighbours of the other type for agent (i, j)
    """
    neighbours = 0
    other_type = 0
    if color == 0:
        color = lattice[i][j]
    for x in [i, i - 1, i + 1]:
        for y in [j, j - 1, j + 1]:
            if (x, y) == (i, j):
                continue
            curr = lattice[x % lattice.shape[0]][y % lattice.shape[1]]
            neighbours += np.abs(curr)
            if curr not in [color, 0]:
                other_type += 1
    if return_counts:
        return (neighbours, other_type)
    return (other_type / neighbours, 0) if neighbours != 0 else (0, 0)

@njit 
def general_tolerance(lattice: np.array, tolerance: float, threshold: float = 0.99):
    """
    Checks if the system is stable (the number of unhappy agents is smaller than the threshold value)
    """
    # acceptable number of unhappy agents
    allowed_number_of_unhappy_agents = int((1 - threshold) * lattice.shape[0] * lattice.shape[1]) 
    c = 0 # current number of unhappy agents
    for (i, j), x in np.ndenumerate(lattice):
        curr_tolerance = find_tolerance(lattice, i, j)[0]
        if curr_tolerance >= tolerance:
            c += 1
        if c >= allowed_number_of_unhappy_agents: 
            return 0  # the number of unhappy agents crossed the threshold -> system is unstable yet
    return 1  # system is stable

@njit
def schelling_migrate(lattice: np.array, i: int, j: int, tolerance: float): 
    """
    Finds a place to migrate the agent (i, j) until a happy place was found 
    or max iterations number reached
    """
    num_iters = int(lattice.shape[0] * lattice.shape[1] / 20)
    x, y = 0, 0 # looking for a new home :) 
    flag = 0

    for _ in range(num_iters):
        empty_spots = find_empty_spots(lattice)
        if empty_spots.shape[0] == 0:
            break
        x, y = empty_spots[np.random.randint(empty_spots.shape[0])] # choose empty spot randomly
        potential_tol, _ = find_tolerance(lattice, x, y, color=lattice[i][j]) # tolerance on the new place
        if potential_tol <= tolerance:
            flag = 1
            break

    if flag == 1:    # happy place was found -> migrate
        lattice[x][y] = lattice[i][j]
        lattice[i][j] = 0 # move out from the initial place

    return lattice

@njit
def voter_new_language(lattice: np.array, i: int, j: int, prestige: float): 
    """
    Learning new language depending on the transition probability and the prestige of the language
    """
    agent_language = lattice[i][j]
    agent_language_count, other_language_count = count_speakers(lattice, agent_language)

    prob = other_language_count / (2 * (agent_language_count + other_language_count))

    # TODO check prestige
    if np.random.uniform(0, 1) < prob:
        lattice[i][j] = -agent_language

    return lattice

@njit
def update(lattice: np.array, i: int, j: int, tolerance: float, prob_migrate: float, prestige: float):
    """
    Single update of the lattice for agent (i, j)
    Schelling - agent tries to change his location, choosing a new one at random among 
                the empty sites in the lattice, where the agent is bound to be happy
    Voter - agent tries to change his color / language
    """
    agent_tol, _ = find_tolerance(lattice, i, j)
    if agent_tol > tolerance:
        if np.random.uniform(0, 1) < prob_migrate:
            lattice = schelling_migrate(lattice, i, j, tolerance)
        else:
            lattice = voter_new_language(lattice, i, j, prestige)

    return lattice

@njit
def simulation(lattice: np.array, tolerance: float, prob_migrate: float, prestige: float):
    """
    Single simulation for the whole lattice
    """
    N = lattice.shape[0]
    run_for = 50000
    
    initial_values = count_speakers(lattice, 1)
    # TODO: check time steps
    for i in range(run_for):
        if i % 10000 == 0:
            print(f"Step {i} / {run_for}")
        agents = find_agents(lattice)
        x, y = agents[np.random.randint(agents.shape[0])] # make sure we choose agent (non-zero element)
        update(lattice, x, y, tolerance, prob_migrate, prestige)
        if i % 100 == 0:
            if general_tolerance(lattice, tolerance):
                break
    
    print("Starting values: ", initial_values)
    print("Ending values: ", count_speakers(lattice, 1))

    return lattice
