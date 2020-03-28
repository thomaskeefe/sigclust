from sklearn.cluster import KMeans
from .compute_cluster_index import compute_cluster_index, compute_sum_of_square_distances_to_mean
import numpy as np

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
        labels: a list or vector of 1's and 2's
        """

        eigenvalues = np.linalg.eigvalsh(np.dot(data.T, data))
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
