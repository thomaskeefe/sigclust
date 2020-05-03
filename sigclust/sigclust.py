from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class SigClust(object):
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        self.simulated_cluster_indices = None
        self.p_value = None
        self.z_score = None

    def fit(self, data, labels):
        """Fit the SigClust object.
        Currently only implementing the version of
        SigClust that uses the sample covariance matrix.

        data: a matrix where rows are observations and cols are features.
        labels: a list or array of cluster labels. Must have two unique members.
        """
        eigenvalues = np.linalg.eigvals(np.cov(data.T))

        # Number of eigenvalues is min(n, d). We pad to length d
        n, d = data.shape
        padded_eigenvalues = np.zeros(d)
        padded_eigenvalues[:len(eigenvalues)] = eigenvalues

        sample_cluster_index = compute_cluster_index(data, labels)

        def simulate_cluster_index():
            # Recall that using invariance, it suffices to simulate
            # mean 0 MVNs with diagonal covariance matrix of sample
            # eigenvalues
            simulated_matrix = np.random.multivariate_normal(np.zeros(d), np.diag(padded_eigenvalues), size=n)
            # NOTE: Using k-means++ starts. Marron's version has options
            # for random starts or PCA starts. Implement?
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(simulated_matrix)
            total_sum_squares = compute_sum_of_square_distances_to_mean(simulated_matrix)

            cluster_index = kmeans.inertia_ / total_sum_squares
            return cluster_index

        self.simulated_cluster_indices = [simulate_cluster_index() for i in range(self.num_simulations)]

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)

class SamplingSigClust(SigClust):
    "A SigClust that takes samples from the majority class"
    def __init__(self, num_samplings=100, num_simulations_per_sample=100):
        self.num_samplings = num_samplings
        self.num_simulations_per_sample = num_simulations_per_sample
        super().__init__()

    def fit(self, data, labels):
        """Fit the SamplingSigClust object.
        Under the hood, this just fits SigClust instances to subsamples,
        and collects all the resulting simulated cluster indicies.

        data: a matrix where rows are observations and cols are features.
        labels: a list or array of cluster labels. Must have two unique members.
        """
        data = pd.DataFrame(data)
        sample_cluster_index = compute_cluster_index(data, labels)

        class_1, class_2 = split_data(data, labels)
        majority_class, minority_class = sort_by_n_desc([class_1, class_2])

        nmin = minority_class.shape[0]

        self.simulated_cluster_indices = []
        for i in range(self.num_samplings):
            sample_from_majority_class = majority_class.sample(nmin, replace=False)
            new_data = pd.concat([sample_from_majority_class, minority_class], ignore_index=True)
            new_labels = np.concatenate([np.repeat(1, nmin), np.repeat(2, nmin)])
            sc = SigClust(self.num_simulations_per_sample)
            sc.fit(new_data, new_labels)
            self.simulated_cluster_indices.extend(sc.simulated_cluster_indices)

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)

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

def compute_sum_of_square_distances_to_mean(X):
    """Compute the sum of squared distances of the rows of X to the row-mean
    for X.
    X: np.ndarray or pd.DataFrame
    """
    mean = np.mean(X, axis=0)
    # NOTE: if X is np.ndarray, then subtraction is broadcast along axis 0 (rows)
    # if X is pd.DataFrame, then substraction is handled intelligently using column index
    displacements = X - mean
    distances = np.linalg.norm(displacements, axis=1)
    return np.sum(distances**2)

def compute_cluster_index(data, labels):
    """Compute the cluster index for the two-class clustering
    given by `labels`."""
    class_1, class_2 = split_data(data, labels)

    class_1_sum_squares = compute_sum_of_square_distances_to_mean(class_1)
    class_2_sum_squares = compute_sum_of_square_distances_to_mean(class_2)
    total_sum_squares = compute_sum_of_square_distances_to_mean(data)

    cluster_index = (class_1_sum_squares + class_2_sum_squares) / total_sum_squares
    return cluster_index
