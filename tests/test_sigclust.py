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


class TestWeightedFunctions(TestCase):
    """Test weighted functions (mean, cov, etc) by comparing them
    to their normal counterparts using suitably multiplied data."""
    def setUp(self):
        self.maj_class = np.array([[1,2], [3,4], [5,6], [7,8]])
        self.min_class = np.array([[9, 10], [11,12]])

    def assert_arrays_close(self, test_array, ref_array):
        "Custom assertion for arrays being almost equal"
        try:
            np.testing.assert_allclose(test_array, ref_array)
        except AssertionError:
            self.fail()

    def test_compute_weighted_mean(self):
        # Computing the refernce value we double up the minority class.
        reference_value = np.mean(np.concatenate([self.maj_class, self.min_class, self.min_class]), axis=0)
        computed_value = sigclust.compute_weighted_mean(self.maj_class, self.min_class)
        self.assert_arrays_close(computed_value, reference_value)

    def test_compute_weighted_covariance(self):
        # Computing the refernce value we double up the minority class.
        reference_value = np.cov(np.concatenate([self.maj_class, self.min_class, self.min_class]).T)
        computed_value = sigclust.compute_weighted_covariance(self.maj_class, self.min_class)
        self.assert_arrays_close(computed_value, reference_value)

    def test_computed_weighted_cluster_index(self):
        np.random.seed(824)
        maj_class = np.random.normal(size=(20, 2)) + np.array([10, 10])
        min_class = np.random.normal(size=(10, 2)) + np.array([-10, -10])

        # Computing the refernce value we double up the minority class.
        reference_data = np.concatenate([maj_class, min_class, min_class], axis=0)
        reference_labels = np.concatenate([np.repeat(1, 20), np.repeat(2, 20)])
        reference_ci = sigclust.compute_cluster_index(reference_data, reference_labels)

        test_ci = sigclust.compute_weighted_cluster_index(maj_class, min_class)
        self.assertAlmostEqual(reference_ci, test_ci, places=8)


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

    def test_SamplingSigClust_number_of_simulations(self):
        "Test that SamplingSigClust simulates the correct number of cluster indices"
        sc = sigclust.SamplingSigClust(num_samplings=3, num_simulations_per_sample=5)
        # Total number of simulations should be 3*5 = 15
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(len(sc.differences), 15)

    def test_SamplingSigClust_fit(self):
        "Test that SamplingSigClust gets a p-value of 0 for two well separated classes"
        sc = sigclust.SamplingSigClust(num_samplings=4, num_simulations_per_sample=25)
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(sc.p_value, 0)

    def test_WeightedSigClust_fit(self):
        "Test that WeightedSigClust gets a p-value of 0 for two well separated classes"
        sc = sigclust.WeightedSigClust(num_simulations=100)
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(sc.p_value, 0)

    def test_ConstrainedKMeansSigClust_fit(self):
        "Test that ConstrainedKMeansSigClust gets a p-value of 0 for two well separated classes"
        sc = sigclust.ConstrainedKMeansSigClust(num_simulations=100)
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(sc.p_value, 0)

    def test_AvgCISigClust_fit(self):
        "Test that AvgCISigClust gets a p-value of 0 for two well separated classes"
        sc = sigclust.AvgCISigClust(num_simulations=100)
        sc.fit(self.test_data, self.test_labels, g=0.59)
        self.assertEqual(sc.p_value, 0)
