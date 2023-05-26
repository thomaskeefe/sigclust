import sigclust
import numpy as np
from pandas import Series
from unittest import TestCase

class TestUtilityFunctions(TestCase):
    "Test the deterministic functions that SigClust uses"
    def setUp(self):
        self.test_data = np.array([[-1, 1], [-1, 0], [-1, -1], [1, 1], [1, 0], [1, -1]])

    def test_compute_sum_of_square_distances_to_mean(self):
        # On paper we can work out that the mean of test_data is the origin
        # and the sum of square distances is 4*2 + 2 = 10.
        sum_squares = sigclust.compute_sum_of_square_distances_to_mean(self.test_data)
        self.assertAlmostEqual(sum_squares, 10, places=8)

    def test_compute_cluster_index_with_numeric_labels(self):
        # Set labels so that test points with x coordinate of -1 are class 1
        # and x coordinate of 1 are class 2.
        # The within class sum of squares for each is 2.
        # The total sum of squares is 4*2 + 2 = 10.
        # So the cluster index is 4/10
        labels = np.array([1, 1, 1, 2, 2, 2])
        ci = sigclust.compute_cluster_index(self.test_data, labels)
        self.assertAlmostEqual(ci, 4.0/10, places=8)

    def test_compute_cluster_index_with_string_labels(self):
        # Make sure string labels are ok
        ci = sigclust.compute_cluster_index(self.test_data, ['a', 'a', 'a', 'b', 'b', 'b'])
        self.assertAlmostEqual(ci, 4.0/10, places=8)

    def test_compute_cluster_index_with_boolean_labels(self):
        # It would be very ugly to use boolean labels but it should still work
        ci = sigclust.compute_cluster_index(self.test_data, [True, True, True, False, False, False])
        self.assertAlmostEqual(ci, 4.0/10, places=8)

    def test_compute_cluster_index_with_bad_labels(self):
        "Make sure errors raise for bad input cluster labelings"
        # More than 2 unique labels
        labels = [1, 1, 2, 2, 3]
        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, labels)

        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, np.array(labels))

        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, Series(labels))

        # Contains None
        labels = [1, None, 2]
        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, labels)

        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, np.array(labels))

        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, Series(labels))

        # Contains nan
        labels = [1, np.nan, 2]
        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, labels)

        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, np.array(labels))

        with self.assertRaises(ValueError):
            sigclust.compute_cluster_index(self.test_data, Series(labels))

    def test_get_eigenvalues_for_d_less_than_n(self):
        np.random.seed(824)
        d = 5
        n = 20
        data = np.random.standard_normal(size=(n,d))
        eigenvalues = sigclust.get_eigenvalues(data, covariance_method='sample_covariance')
        self.assertEqual(len(eigenvalues), d)

    def test_get_eigenvalues_for_d_equals_n(self):
        np.random.seed(824)
        d = 5
        n = 5
        data = np.random.standard_normal(size=(n,d))
        eigenvalues = sigclust.get_eigenvalues(data, covariance_method='sample_covariance')
        self.assertEqual(len(eigenvalues), d)

    def test_get_eigenvalues_for_d_greater_than_n(self):
        np.random.seed(824)
        d = 10
        n = 5
        data = np.random.standard_normal(size=(n,d))
        assert data.shape[0]==n
        assert data.shape[1]==d
        eigenvalues = sigclust.get_eigenvalues(data, covariance_method='sample_covariance')
        self.assertEqual(len(eigenvalues), d)



class TestSigClust(TestCase):
    "Test SigClust and its variants"
    def setUp(self):
        np.random.seed(824)
        class_1 = np.random.normal(size=(20, 2)) + np.array([10, 10])
        class_2 = np.random.normal(size=(20, 2)) + np.array([-10, -10])
        self.test_data = np.concatenate([class_1, class_2], axis=0)
        self.test_labels = np.concatenate([np.repeat(1, 20), np.repeat(2, 20)])

    def test_SigClust_fit(self):
        "Test that two well separated clusters produce a p-value of 0"
        sc = sigclust.SigClust(num_simulations=100)
        # The test data is two balanced classes of bivariate gaussian data.
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(sc.p_value, 0)


    def test_WCISigClust_fit(self):
        "Test that WeightedSigClust gets a p-value of 0 for two well separated classes"
        sc = sigclust.WeightedSigClust(num_simulations=100)
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(sc.p_value, 0)

