import numpy as np

def compute_sum_of_square_distances_to_mean(data):
    """Compute the sum of squared distances to the mean
    for `data`. `data` should be a rows-as-observations
    matrix."""
    displacements = data - np.mean(data, axis=0)
    distances = np.linalg.norm(displacements, axis=1)
    return np.sum(distances**2)

def compute_cluster_index(data, labels):
    """Compute the cluster index for the two-class clustering
    given by `labels`."""
    class_1 = data[labels==1]
    class_2 = data[labels==2]

    class_1_sum_squares = compute_sum_of_square_distances_to_mean(class_1)
    class_2_sum_squares = compute_sum_of_square_distances_to_mean(class_2)
    total_sum_squares = compute_sum_of_square_distances_to_mean(data)

    cluster_index = (class_1_sum_squares + class_2_sum_squares) / total_sum_squares
    return cluster_index
