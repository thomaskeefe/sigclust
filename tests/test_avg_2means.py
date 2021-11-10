import sigclust.avg_2means
import numpy as np
from sklearn.cluster import KMeans
from pandas import Series
from unittest import TestCase
import scipy.stats
from scipy.spatial.distance import pdist, squareform

class TestAvg2Means(TestCase):
    "Test the Avg2Means class"
    def setUp(self):
        self.data = np.random.standard_normal(size=(20, 2))
        self.data *= np.array([3, 1])  # stretch
        self.clusterer = sigclust.avg_2means.Avg2Means(progressbar=False)

    def assert_arrays_close(self, test_array, ref_array):
        "Custom assertion for arrays being almost equal"
        try:
            np.testing.assert_allclose(test_array, ref_array)
        except AssertionError:
            self.fail()

    def test_labels_are_1s_and_2s(self):
        self.clusterer.fit(self.data, g=.59)
        labels = self.clusterer.labels
        self.assertEqual(set(labels), {1, 2})

    def test_that_2_dimensional_data_is_ok(self):
        "Test Avg2Means can handle a 2D-array input"
        pass

    def test_that_1_dimensional_data_is_ok(self):
        "Test Avg2Means can handle a 1D-array input"
        pass

    def test_avg_ci_is_in_unit_interval(self):
        self.clusterer.fit(self.data, g=.59)
        ci = self.clusterer.ci
        self.assertTrue(0 <= ci <= 1)

    def test_p_equals_0_is_usual_ci(self):
        "Test that p=0 matches the conventional cluster index"
        pass

    def test_that_p_of_half_splits_gaussian_in_middle(self):
        norm_inv = scipy.stats.norm.ppf
        ideal_gaussian = norm_inv(np.linspace(0.001, .999, 200)).reshape(-1,1)
        self.clusterer.fit(ideal_gaussian)

    def test_EuclideanDistanceMatrix(self):
        A = np.arange(32).reshape(8,4)
        D2 = squareform(pdist(A, 'sqeuclidean'))
        candidate = sigclust.avg_2means.EuclideanDistanceMatrix(A)
        self.assert_arrays_close(candidate, D2)
