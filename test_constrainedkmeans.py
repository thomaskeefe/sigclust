from sigclust import constrained_kmeans
from unittest import TestCase
import numpy as np
from collections import Counter

class TestConstrainedKMeans(TestCase):
    def test_constrained_k_means(self):
        "Test constrained_kmeans produces same result as author's version"
        np.random.seed(824)
        data = np.random.random((75, 3))
        (C, M, f) = constrained_kmeans(data, [25, 25, 25])

        # These reference centroids are computed using the original author's
        # function so if we make readability improvements to the function it
        # should produce these same results
        reference_centroids = np.array([[0.32760021, 0.65153249, 0.37212288],
                                        [0.73819571, 0.29718086, 0.33535881],
                                        [0.40933757, 0.47199784, 0.84708941]])
        try:
            np.testing.assert_allclose(C, reference_centroids)
        except AssertionError:
            self.fail()

    def test_cluster_sizes(self):
        "Check that cluster sizes are correctly constrained"
        # We want 15 in class 0 and 5 in class 1
        data = np.random.random((20, 2))
        (C, M, f) = constrained_kmeans(data, [15, 5])
        cluster_sizes = Counter(M)
        self.assertEqual(cluster_sizes, {0: 15, 1:5})
