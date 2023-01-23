import numpy as np
from numba import njit
import random

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
def find_tolerance(lattice: np.array, i: int, j: int, return_counts: bool=False):
    """
    Count all neighbours and neighbours of the other type for agent (i, j)
    """
    neighbours = 0
    other_type = 0
    
    for x in [i, i - 1, i + 1]:
        for y in [j, j - 1, j + 1]:
            if (x, y) == (i, j):
                continue
            curr = lattice[x % lattice.shape[0]][y % lattice.shape[1]]
            neighbours += np.abs(curr)
            if curr not in [lattice[i][j], 0]:
                other_type += 1
    if return_counts:
        return (neighbours, other_type)
    return (other_type / neighbours, 0) if neighbours != 0 else (0, 0)

@njit
def schelling_migrate(lattice: np.array, i: int, j: int, tolerance: float): 
    is_happy = False
    x, y = 0, 0 # looking for a new home :) 
    while not is_happy:
        # repeat until a "happy" spot found
        empty_spots = find_empty_spots(lattice)
        x, y = empty_spots[np.random.randint(empty_spots.shape[0])] # choose empty spot randomly
        lattice[x][y] = lattice[i][j] # test migrate
        potential_tolerance, _ = find_tolerance(lattice, x, y) # tolerance on the new place
        if potential_tolerance <= tolerance:
            is_happy = True
        else:
            lattice[x][y] = 0

    lattice[i][j] = 0 # move out from the initial place

    return lattice

@njit
def voter_new_language(lattice: np.array, i: int, j: int, prestige: float): 
    agent_language = lattice[i][j]
    agent_language_count, other_language_count = count_speakers(lattice, agent_language)
    # TODO

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
            # Schelling behaviour (try to migrate). 
            # Change is not mandatory, migration can fail if the agent is unhappy in 
            # any emtty space
            schelling_migrate(lattice, i, j, tolerance)
        else:
            # Voter behaviour (try to learn a new language)
            # The language change would depend on transition probabilities q
            # which can include both the number of agent of each language 
            # (locally or globally measured) and the perceived prestige of the other 
            # language
            voter_new_language(lattice, i, j, prestige)

@njit
def simulation(lattice: np.array, tolerance: float, prob_migrate: float, prestige: float):
    """
    Single simulation for the whole lattice
    """
    N = lattice.shape[0]
    run_for = 100
    curr_step = 0
    for _ in range(N * N):
        if curr_step == run_for:
            break
        agents = find_agents(lattice)
        x, y = agents[np.random.randint(agents.shape[0])] # make sure we choose agent (non-zero element)
        update(lattice, x, y, tolerance, prob_migrate, prestige)
        curr_step += 1
