from sigclust import constrained_kmeans
from unittest import TestCase
import numpy as np

class TestConstrainedKMeans(TestCase):
    def test_constrained_k_means(self):
        np.random.seed(824)
        data = np.random.random((75, 3))
        (C, M, f) = constrained_kmeans(data, [25, 25, 25])
        reference_centroids = np.array([[0.32760021, 0.65153249, 0.37212288],
                                        [0.73819571, 0.29718086, 0.33535881],
                                        [0.40933757, 0.47199784, 0.84708941]])
        try:
            np.testing.assert_allclose(C, reference_centroids)
        except AssertionError:
            self.fail()
