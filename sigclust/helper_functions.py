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

def compute_weighted_mean(X1, X2):
    # The weighted mean is the mean of the means
    means = np.array([X1.mean(axis=0), X2.mean(axis=0)])
    return means.mean(axis=0)

def compute_weighted_covariance(X1, X2):
    maj_class, min_class = sort_by_n_desc([X1, X2])
    nmaj = maj_class.shape[0]
    nmin = min_class.shape[0]

    X = np.concatenate([maj_class, min_class], axis=0)
    X_bar = compute_weighted_mean(maj_class, min_class)
    C = X - X_bar  # C for 'centered'
    weights = np.concatenate([np.ones(nmaj), np.repeat(nmaj/nmin, nmin)])
    W = np.diag(weights)
    return 1/(2*nmaj-1) * C.T.dot(W).dot(C)

def compute_weighted_cluster_index(X1, X2):
    """Compute the weighted cluster index for X1 and X2"""
    maj_class, min_class = sort_by_n_desc([X1, X2])
    nmaj = maj_class.shape[0]
    nmin = min_class.shape[0]

    maj_class_sum_squares = compute_sum_of_square_distances_to_mean(maj_class)
    min_class_sum_squares = compute_sum_of_square_distances_to_mean(min_class)

    # Sum of squared distances to weighted mean
    weighted_mean = compute_weighted_mean(X1, X2)
    SSWM = lambda x: compute_sum_of_square_distances_to_point(x, weighted_mean)

    total_sum_squares = SSWM(maj_class) + (nmaj/nmin)*SSWM(min_class)

    cluster_index = (maj_class_sum_squares + (nmaj/nmin)*min_class_sum_squares) / total_sum_squares
    return cluster_index

# def compute_best_average_cluster_index_p_exp(data, p):
#     scores = PCA().fit_transform(data)
#     scores_df = pd.DataFrame(scores)
#
#     pc1 = scores_df.iloc[:, 0]
#
#     def split_data_by_score(score):
#         class_1 = data[pc1 <= score]
#         class_2 = data[pc1 > score]
#         return (class_1, class_2)
#
#     def compute_avg_ci_p_by_score(score):
#         class_1, class_2 = split_data_by_score(score)
#         return compute_average_cluster_index_p_exp(class_1, class_2, p)
#
#     cis = pc1.map(compute_avg_ci_p_by_score)
#     return cis.min()

# def compute_average_cluster_index_p_exp(class_1, class_2, p):
#     """Compute the average cluster index for the two-class clustering
#     given by `labels`, and using the exponent p"""
#     n1 = class_1.shape[0]
#     n2 = class_2.shape[0]
#     if (n1 == 0) or (n2 == 0):
#         return np.nan
#
#     class_1_SSE = compute_sum_of_square_distances_to_mean(class_1)
#     class_2_SSE = compute_sum_of_square_distances_to_mean(class_2)
#
#     overall_mean = np.concatenate([class_1, class_2]).mean(axis=0)
#
#     numerator = (1/n1)**p * class_1_SSE + (1/n2)**p * class_2_SSE
#     denominator = (compute_sum_of_square_distances_to_point(class_1, overall_mean) / (n1**p) +
#                    compute_sum_of_square_distances_to_point(class_2, overall_mean) / (n2**p) )
#
#     return numerator/denominator
