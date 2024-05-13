import numpy as np
import pandas as pd

p = 2
epsilon = .01

data = np.genfromtxt("betti.csv", delimiter=",")
data = data[:10]
print(data)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

ax.scatter(x, y, z)
plt.show()

def distance(point1, point2, p):
    return np.power(np.sum(np.power(np.abs(point1 - point2), p)), 1/p)

def calculate_distance_matrix(dataset, p):
    distance_matrix = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            distance_matrix[i, j] = distance(dataset[i], dataset[j], p)
    return distance_matrix


distance_matrix = calculate_distance_matrix(data, p) 
print(distance_matrix)


def vietoris_rips_filtration(distance_matrix, epsilon):
    n = distance_matrix.shape[0]
    complex = []

    def add_simplex(simplex, max_distance):
        complex.append(simplex)
        if len(simplex) < n:
            for i in range(max(simplex) + 1, n):
                new_distance = max(max_distance, distance_matrix[simplex[0], i])
                if new_distance <= epsilon:
                    add_simplex(simplex + [i], new_distance)

    for i in range(n):
        add_simplex([i], 0)

    return complex

complex = vietoris_rips_filtration(distance_matrix, epsilon)
print(f"Epsilon {epsilon}: {complex}")

from scipy.sparse import dok_matrix

def calculate_betti_numbers(complex, n):
    # Create boundary operators
    boundary_operators = [dok_matrix((len(complex[k+1]), len(complex[k]))) for k in range(n)]
    
    for k in range(n):
        for i, simplex in enumerate(complex[k]):
            for j, next_simplex in enumerate(complex[k+1]):
                if set(simplex).issubset(set(next_simplex)):
                    boundary_operators[k][j, i] = 1

    # Calculate ranks
    ranks = [np.linalg.matrix_rank(boundary_operators[k].toarray()) for k in range(n)]
    
    # Calculate Betti numbers
    betti_numbers = [ranks[k] - ranks[k+1] for k in range(n-1)]
    betti_numbers.append(ranks[n-1])
    
    return betti_numbers

betti_numbers = calculate_betti_numbers(complex, 3)
print(f"Betti numbers: {betti_numbers}")