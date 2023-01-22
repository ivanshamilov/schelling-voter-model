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
def find_agents(lattice):
    """
    Find indexes of all available agents on the lattice (non-zero elements)
    """
    return np.column_stack((np.where(lattice != 0)))

@njit
def simulation(lattice: np.array, tolerance: float, prob_migrate: float):
    """
    Single simulation for the whole lattice
    """
    N = lattice.shape[0]
    run_for = 50000
    curr_step = 0
    for _ in range(N * N):
        if curr_step == run_for:
            break
        agents = find_agents(lattice)
        x, y = agents[np.random.randint(agents.shape[0])] # make sure we choose agent (non-zero element)
        update(lattice, x, y, tolerance, prob_migrate)
        curr_step += 1

@njit
def count_neighbours(lattice: np.array, i: int, j: int):
    """
    Count all neighbours and neighbours of the other type for agent (i, j)
    """
    neighbours = 0
    other_type = 0
    
    for x in [i, i - 1, i + 1]:
        for y in [j, j - 1, j + 1]:
            if (x, y) == (i, j):
                continue
            curr = lattice[x % lattice.shape[0]][y % lattice.shape[0]]
            neighbours += np.abs(curr)
            if curr not in [lattice[i][j], 0]:
                other_type += 1
    
    return neighbours, other_type

@njit
def update(lattice: np.array, i: int, j: int, tolerance: float, prob_migrate: float):
    """
    Single update of the lattice for agent (i, j)
    """
    neighbours, other_type = count_neighbours(lattice, i, j)
    if neighbours != 0:
        if other_type / neighbours > tolerance:
            if np.random.uniform(0, 1) > prob_migrate:
                # Schelling
                pass
                # Voter
