import numpy as np

def distance(point1, point2, p):
    return np.power(np.sum(np.power(np.abs(point1 - point2), p)), 1/p)

def calculate_distance_matrix(dataset, p):
    distance_matrix = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            distance_matrix[i, j] = distance(dataset[i], dataset[j], p)
    return distance_matrix

dataset = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
p = 2 
distance_matrix = calculate_distance_matrix(dataset, p)  
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

epsilon = 100
complex = vietoris_rips_filtration(distance_matrix, epsilon)
print(f"Epsilon {epsilon}: {complex}")