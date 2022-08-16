from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from sigclust.constrained_kmeans import ConstrainedKMeans
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




class SamplingSigClust(object):
    """A SigClust that provides better power when clusters are unbalanced, by
    running successive SigClusts on the minority class and samples of the
    majority class.
    """
    def __init__(self, num_samplings=100, num_simulations_per_sample=100, covariance_method='soft_thresholding'):
        self.num_samplings = num_samplings
        self.num_simulations_per_sample = num_simulations_per_sample
        self.covariance_method = covariance_method
        self.sigclusts = []
        self.differences = None

    def fit(self, data, labels):
        """Fit the SamplingSigClust object.
        Under the hood, this just fits SigClust instances to subsamples,
        and collects all the resulting simulated cluster indicies.

        data: a matrix where rows are observations and cols are features.
        labels: a list or array of cluster labels. Must have two unique members.
        """
        data = pd.DataFrame(data)

        class_1, class_2 = helper.split_data(data, labels)
        majority_class, minority_class = helper.sort_by_n_desc([class_1, class_2])

        nmin = minority_class.shape[0]

        for i in range(self.num_samplings):
            sample_from_majority_class = majority_class.sample(nmin, replace=False)
            new_data = pd.concat([sample_from_majority_class, minority_class], ignore_index=True)
            new_labels = np.concatenate([np.repeat(1, nmin), np.repeat(2, nmin)])
            sc = SigClust(self.num_simulations_per_sample, covariance_method=self.covariance_method)
            sc.fit(new_data, new_labels)
            self.sigclusts.append(sc)

        difference_arrays = [sc.simulated_cluster_indices - sc.sample_cluster_index for sc in self.sigclusts]
        self.differences = np.concatenate(difference_arrays)

        # since we compute the differences, we compare them to 0 instead of any of the sample CIs
        self.p_value = np.mean(self.differences < 0)
        self.z_score = (0 - np.mean(self.differences))/np.std(self.differences, ddof=1)


class WeightedSigClust(object):
    """Weighed SigClust that computes a weighted eigenstructure for the MVN
    simulations, weighted CI test statistic, uses the usual KMeans algorithm and
    usual KMeans CI for the null distribution of CIs.

    conservative: if False, take samples of size twice the majority class.
                  this is less conservative, but matches the behavior of
                  repeating the points in the minority class to match the majority.
    """
    def __init__(self, num_simulations=1000, conservative=False):
        self.num_simulations = num_simulations
        self.simulated_cluster_indices = None
        self.p_value = None
        self.z_score = None
        self.conservative = conservative
        self.sample_cluster_index = None

    def fit(self, data, labels):
        """Fit the SigClust object.
        Currently only implementing the version of
        SigClust that uses the sample covariance matrix.

        data: a matrix where rows are observations and cols are features.
        labels: a list or array of cluster labels. Must have two unique members.
        """

        data = pd.DataFrame(data)
        class_1, class_2 = helper.split_data(data, labels)
        majority_class, minority_class = helper.sort_by_n_desc([class_1, class_2])

        nmaj, d = majority_class.shape
        nmin = minority_class.shape[0]

        covariance = helper.compute_weighted_covariance(majority_class, minority_class)
        eigenvalues = np.linalg.eigvals(covariance)

        # Number of eigenvalues is min(n, d). We pad to length d
        padded_eigenvalues = np.zeros(d)
        padded_eigenvalues[:len(eigenvalues)] = eigenvalues

        self.sample_cluster_index = helper.compute_weighted_cluster_index(majority_class, minority_class)
        # We compute a weighted cluster index in order to match the behavior
        # of dropping more points on the minority class.

        if self.conservative:
            sim_size = nmaj + nmin
        else:
            sim_size = 2*nmaj

        def simulate_cluster_index():
            # Recall that using invariance, it suffices to simulate
            # mean 0 MVNs with diagonal covariance matrix of sample
            # eigenvalues
            simulated_matrix = np.random.standard_normal(size=(sim_size, d)) * np.sqrt(padded_eigenvalues)  # much faster than np.random.multivariate_normal
            # NOTE: Using k-means++ starts. Marron's version has options
            # for random starts or PCA starts. Implement?
            kmeans = KMeans(n_clusters=2)
            # NOTE: doesn't make sense to use weights here, we wouldn't know which rows to weight.
            kmeans.fit(simulated_matrix)
            total_sum_squares = helper.compute_sum_of_square_distances_to_mean(simulated_matrix)

            cluster_index = kmeans.inertia_ / total_sum_squares
            return cluster_index

        self.simulated_cluster_indices = np.array([simulate_cluster_index() for i in range(self.num_simulations)])

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(self.sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (self.sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)

class ConstrainedKMeansSigClust(object):
    """A SigClust where the simulated clusterings must have the same
    cardinalities as the input clustering"""
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

        sample_cluster_index = helper.compute_cluster_index(data, labels)

        class_1, class_2 = helper.split_data(data, labels)
        n1 = class_1.shape[0]
        n2 = class_2.shape[0]

        def simulate_cluster_index():
            # Recall that using invariance, it suffices to simulate
            # mean 0 MVNs with diagonal covariance matrix of sample
            # eigenvalues
            simulated_matrix = np.random.standard_normal(size=(n, d)) * np.sqrt(padded_eigenvalues)  # much faster than np.random.multivariate_normal

            kmeans = ConstrainedKMeans()
            kmeans.fit(simulated_matrix, demand=(n1, n2))

            ci = helper.compute_cluster_index(simulated_matrix, kmeans.labels)
            return ci

        self.simulated_cluster_indices = np.array([simulate_cluster_index() for i in range(self.num_simulations)])

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)

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
