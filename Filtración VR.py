import numpy as np

# This function calculates the distance between two points
def distance(point1, point2, p):
    # It subtracts the two points, takes the absolute value, raises it to the power of p,
    # sums the result, and then takes the p-th root.
    return np.power(np.sum(np.power(np.abs(point1 - point2), p)), 1/p)

# This function calculates a matrix of distances between all pairs of points in a dataset
def calculate_distance_matrix(dataset, p):
    # It creates an empty matrix of zeros
    distance_matrix = np.zeros((dataset.shape[0], dataset.shape[0]))
    # It fills the matrix with the distances between all pairs of points
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            distance_matrix[i, j] = distance(dataset[i], dataset[j], p)
    return distance_matrix

# This is an example of how to use the functions
dataset = np.array([[1, 2], [3, 4], [5, 6]])  # This is your dataset
p = 2  # This is the power you want to use in the distance calculation
distance_matrix = calculate_distance_matrix(dataset, p)  # This calculates the distance matrix
print(distance_matrix)  # This prints the distance matrix

def vietoris_rips_filtration(distance_matrix, epsilon):
    # Initialize an empty simplicial complex
    complex = []

    # For each pair of points whose distance is less than or equal to epsilon
    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[0]):
            if distance_matrix[i, j] <= epsilon:
                # Add the pair of points to the simplicial complex
                complex.append({i, j})

                # For each other point in the simplicial complex
                for k in range(j+1, distance_matrix.shape[0]):
                    # If the maximum distance to the pair of points is less than or equal to epsilon
                    if max(distance_matrix[i, k], distance_matrix[j, k]) <= epsilon:
                        # Add a simplex consisting of the point and the pair of points to the simplicial complex
                        complex.append({i, j, k})

    # Return the simplicial complex
    return complex

# Example usage
epsilon = 0.5  # Or whatever value you want to use
complex = vietoris_rips_filtration(distance_matrix, epsilon)
print(f"Epsilon {epsilon}: {complex}")