from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


import sigclust.helper_functions as helper
import sigclust.avg_2means as avg_2means
import sigclust.soft_thresholding as soft_thresholding
import sigclust.sdp_clustering as sdp_clustering

class SigClust(object):
    def __init__(self, num_simulations=1000, covariance_method='soft_thresholding'):
        self.num_simulations = num_simulations
        self.simulated_cluster_indices = None
        self.sample_cluster_index = None
        self.p_value = None
        self.z_score = None
        self.covariance_method = covariance_method

    def fit(self, data, labels):
        """Fit the SigClust object.
        Currently only implementing the version of
        SigClust that uses the sample covariance matrix.

        data: a matrix where rows are observations and cols are features.
        labels: a list or array of cluster labels. Must have two unique members.
        """
        n, d = data.shape
        eigenvalues = get_eigenvalues(data, self.covariance_method)

        self.sample_cluster_index = helper.compute_cluster_index(data, labels)

        def simulate_cluster_index():
            # Recall that using invariance, it suffices to simulate
            # mean 0 MVNs with diagonal covariance matrix of sample
            # eigenvalues
            simulated_matrix = np.random.standard_normal(size=(n, d)) * np.sqrt(eigenvalues)  # much faster than np.random.multivariate_normal
            # NOTE: Using k-means++ starts. Marron's version has options
            # for random starts or PCA starts. Implement?
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(simulated_matrix)
            total_sum_squares = helper.compute_sum_of_square_distances_to_mean(simulated_matrix)

            cluster_index = kmeans.inertia_ / total_sum_squares
            return cluster_index

        self.simulated_cluster_indices = np.array([simulate_cluster_index() for i in range(self.num_simulations)])

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(self.sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (self.sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)

    def plot_test_statistic(self, ax=None):
        "Plot the test statistic and its null distribution"
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(self.simulated_cluster_indices, color='lightgray')
        ax.axvline(self.sample_cluster_index, color='red')
        ax.set_xlabel("Cluster Index")
        ax.text(0.05, .95, f"p-value: {self.p_value:.3f}\nz-score: {self.z_score:.2f}")

    def plot_qq_fit(self, ax=None):
        pass


class AvgCISigClust(object):
    def __init__(self, num_simulations=1000, covariance_method='soft_thresholding', max_components=1):
        self.num_simulations = num_simulations
        self.covariance_method = covariance_method
        self.max_components = max_components
        self.simulated_cluster_indices = None
        self.p_value = None
        self.z_score = None
        self.sample_cluster_index = None
        self.clusterer = avg_2means.Avg2Means(self.max_components, progressbar=False)

    def fit(self, data, labels, g):
        """Fit the SigClust object.
        Currently only implementing the version of
        SigClust that uses the sample covariance matrix.

        data: a matrix where rows are observations and cols are features.
        labels: a list or array of cluster labels. Must have two unique members.
        """
        n, d = data.shape
        eigenvalues = get_eigenvalues(data, self.covariance_method)

        class_1, class_2 = helper.split_data(data, labels)
        n1 = class_1.shape[0]
        n2 = class_2.shape[0]

        self.sample_cluster_index = sdp_clustering.compute_average_cluster_index_g_exp(class_1, class_2, g)

        def simulate_cluster_index():
            simulated_matrix = np.random.standard_normal(size=(n, d)) * np.sqrt(eigenvalues)  # much faster than np.random.multivariate_normal
            self.clusterer.fit(simulated_matrix, g)
            return self.clusterer.ci

        self.simulated_cluster_indices = np.array(
            [simulate_cluster_index() for i in tqdm(range(self.num_simulations))]
            )

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(self.sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (self.sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)

class SDPSigClust(object):
    def __init__(self, num_simulations=1000, covariance_method='soft_thresholding'):
        self.num_simulations = num_simulations
        self.covariance_method = covariance_method
        self.max_components = max_components
        self.simulated_cluster_indices = None
        self.p_value = None
        self.z_score = None
        self.sample_cluster_index = None

    def fit(self, data, labels, g):
        """Fit the SigClust object.
        Currently only implementing the version of
        SigClust that uses the sample covariance matrix.

        data: a matrix where rows are observations and cols are features.
        labels: a list or array of cluster labels. Must have two unique members.
        """
        n, d = data.shape
        eigenvalues = get_eigenvalues(data, self.covariance_method)

        class_1, class_2 = helper.split_data(data, labels)
        n1 = class_1.shape[0]
        n2 = class_2.shape[0]

        self.sample_cluster_index = avg_2means.compute_average_cluster_index_p_exp(class_1, class_2, g)

        def simulate_cluster_index():
            simulated_matrix = np.random.standard_normal(size=(n, d)) * np.sqrt(eigenvalues)  # much faster than np.random.multivariate_normal
            clusterer = avg_2means.Avg2Means(self.max_components)
            clusterer.fit(simulated_matrix, g)
            return clusterer.ci

        self.simulated_cluster_indices = np.array([simulate_cluster_index() for i in range(self.num_simulations)])

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(self.sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (self.sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)

def get_eigenvalues(data, covariance_method):
    """Get the eigenvalues of the sample covariance matrix of the data,
    and return them directly (covariance_method='sample_covariance') or
    soft threshold them (covariance_method='soft_thresholding')
    """
    n, d = data.shape
    # NOTE: the scipy routines used here return eigenvalues in ascending order.
    # SigClust doesn't care what order the eigenvalues are.
    if d <= n:
        eigenvalues, eigenvectors = scipy.linalg.eigh(np.cov(data.T))
    else:
        # When d > n, the sample covariance matrix will have (d-n) zero eigenvalues,
        # so it is much faster to just ask for the n largest and fill the rest
        # with zeros.
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(np.cov(data.T), k=n, which='LM')
        eigenvalues = np.concatenate([np.zeros(d-n), eigenvalues])

    assert len(eigenvalues) == d

    if covariance_method == 'sample_covariance':
        return eigenvalues

    elif covariance_method == 'soft_thresholding':
        sig2b = soft_thresholding.estimate_background_noise(data)
        soft_thresholded_eigenvalues = soft_thresholding.soft_threshold_hanwen_huang(eigenvalues, sig2b)
        return soft_thresholded_eigenvalues

    else:
        raise ValueError("Covariance method must be 'sample_covariance' or 'soft_thresholding'")
