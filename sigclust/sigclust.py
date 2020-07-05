from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from sigclust.constrained_kmeans import ConstrainedKMeans
import sigclust.helper_functions as helper
import sigclust.avg_2means as avg_2means

class SigClust(object):
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        self.simulated_cluster_indices = None
        self.sample_cluster_index = None
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

        self.sample_cluster_index = helper.compute_cluster_index(data, labels)

        def simulate_cluster_index():
            # Recall that using invariance, it suffices to simulate
            # mean 0 MVNs with diagonal covariance matrix of sample
            # eigenvalues
            simulated_matrix = np.random.multivariate_normal(np.zeros(d), np.diag(padded_eigenvalues), size=n)
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


class SamplingSigClust(object):
    """A SigClust that provides better power when clusters are unbalanced, by
    running successive SigClusts on the minority class and samples of the
    majority class.
    """
    def __init__(self, num_samplings=100, num_simulations_per_sample=100):
        self.num_samplings = num_samplings
        self.num_simulations_per_sample = num_simulations_per_sample
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
            sc = SigClust(self.num_simulations_per_sample)
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
            simulated_matrix = np.random.multivariate_normal(np.zeros(d), np.diag(padded_eigenvalues), size=sim_size)
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
            simulated_matrix = np.random.multivariate_normal(np.zeros(d), np.diag(padded_eigenvalues), size=n)

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
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        self.simulated_cluster_indices = None
        self.p_value = None
        self.z_score = None
        self.sample_cluster_index = None

    def fit(self, data, labels, p):
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


        class_1, class_2 = helper.split_data(data, labels)
        n1 = class_1.shape[0]
        n2 = class_2.shape[0]

        self.sample_cluster_index = avg_2means.compute_average_cluster_index_p_exp(class_1, class_2, p)

        def simulate_cluster_index():
            simulated_matrix = np.random.multivariate_normal(np.zeros(d), np.diag(padded_eigenvalues), size=n)
            clusterer = avg_2means.Avg2Means()
            clusterer.fit(simulated_matrix, p)
            return clusterer.ci

        self.simulated_cluster_indices = np.array([simulate_cluster_index() for i in range(self.num_simulations)])

        # TODO: Implement Marron's continuous empirical probability procedure
        # for the p-value.
        self.p_value = np.mean(self.sample_cluster_index >= self.simulated_cluster_indices)
        self.z_score = (self.sample_cluster_index - np.mean(self.simulated_cluster_indices))/np.std(self.simulated_cluster_indices, ddof=1)
