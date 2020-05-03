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


class TestSigClust(TestCase):
    "Test the SigClust class"
    def setUp(self):
        np.random.seed(824)
        class_1 = np.random.normal(size=(20, 2)) + np.array([10, 10])
        class_2 = np.random.normal(size=(20, 2)) + np.array([-10, -10])
        self.test_data = np.concatenate([class_1, class_2], axis=0)
        self.test_labels = np.concatenate([np.repeat(1, 20), np.repeat(2, 20)])

    def test_SigClust(self):
        "Test SigClust end-to-end"
        sc = sigclust.SigClust(num_simulations=100)
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(sc.p_value, 0)

    def test_random_seed(self):
        "Test that runs of SigClust with same seed give same results"
        np.random.seed(824)
        sc = sigclust.SigClust(num_simulations=100)
        sc.fit(self.test_data, self.test_labels)

        np.random.seed(824)
        sc2 = sigclust.SigClust(num_simulations=100)
        sc2.fit(self.test_data, self.test_labels)

        self.assertEqual(sc.simulated_cluster_indices, sc2.simulated_cluster_indices)

    def test_random_seed_2(self):
        "Test that runs of SigClust with different seed give (slightly) different results"
        np.random.seed(824)
        sc = sigclust.SigClust(num_simulations=100)
        sc.fit(self.test_data, self.test_labels)

        np.random.seed(555)  # DIFFERENT SEED
        sc2 = sigclust.SigClust(num_simulations=100)
        sc2.fit(self.test_data, self.test_labels)

        self.assertNotEqual(sc.simulated_cluster_indices, sc2.simulated_cluster_indices)

class TestSamplingSigClust(TestCase):
    def setUp(self):
        np.random.seed(824)
        class_1 = np.random.normal(size=(20, 2)) + np.array([10, 10])
        class_2 = np.random.normal(size=(20, 2)) + np.array([-10, -10])
        self.test_data = np.concatenate([class_1, class_2], axis=0)
        self.test_labels = np.concatenate([np.repeat(1, 20), np.repeat(2, 20)])

    def test_correct_number_of_simulations(self):
        sc = sigclust.SamplingSigClust(num_samplings=3, num_simulations_per_sample=5)
        # Total number of simulations should be 3*5 = 15
        sc.fit(self.test_data, self.test_labels)
        self.assertEqual(len(sc.simulated_cluster_indices), 15)
