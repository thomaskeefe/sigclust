import numpy as np
import pandas as pd

def split_data(data, labels):
    labels = np.array(labels)

    if len(labels) != data.shape[0]:
        raise ValueError("Number of labels must match number of observations")

    if np.any(pd.isna(labels)):
        raise ValueError("Labels must not contain nan or None")

    if len(np.unique(labels)) != 2:
        raise ValueError("Labels must have exactly 2 unique members")

    class_names = np.unique(labels)

    X1 = data[labels==class_names[0]]
    X2 = data[labels==class_names[1]]

    return (X1, X2)

def sort_by_n_desc(dataframes):
    return sorted(dataframes, reverse=True, key=lambda df: df.shape[0])

def compute_sum_of_square_distances_to_point(X, y):
    """Compute the sum of squared distances of the rows of X
    to the given point y.
    X: 2-dimensional np.ndarray or pd.DataFrame
    y: 1-dimensional np.ndarray or pd.Series"""
    # NOTE: if X is np.ndarray, then subtraction is broadcast along axis 0 (rows)
    # if X is pd.DataFrame, then substraction is handled intelligently using index
    displacements = X - y
    distances = np.linalg.norm(displacements, axis=1)
    return np.sum(distances**2)

def compute_sum_of_square_distances_to_mean(X):
    """Compute the sum of squared distances of the rows of X to the row-mean
    for X.
    X: 2-dimensional np.ndarray or pd.DataFrame
    """
    mean = np.mean(X, axis=0)
    return compute_sum_of_square_distances_to_point(X, mean)

def compute_cluster_index(data, labels):
    """Compute the cluster index for the two-class clustering
    given by `labels`."""
    class_1, class_2 = split_data(data, labels)
    ci = compute_cluster_index_given_classes(class_1, class_2)
    return ci

def compute_cluster_index_given_classes(class_1, class_2):
    """Compute the cluster index for two given classes."""
    class_1_sum_squares = compute_sum_of_square_distances_to_mean(class_1)
    class_2_sum_squares = compute_sum_of_square_distances_to_mean(class_2)
    total_sum_squares = compute_sum_of_square_distances_to_mean(np.concatenate([class_1, class_2]))

    cluster_index = (class_1_sum_squares + class_2_sum_squares) / total_sum_squares
    return cluster_index

