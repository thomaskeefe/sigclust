import sigclust
import numpy as np
from unittest import TestCase

class Tests(TestCase):
    def setUp(self):
        # Test data is the 4 points on the unit square
        self.test_data = np.array([[-1, 1], [-1, -1], [1, 1], [1, -1]])
        # The left hand points of the unit square are class 1,
        # right hand are class 2
        self.test_labels = np.array([1, 1, 2, 2])

    def test_compute_sum_of_square_distances_to_mean(self):
        # Initialize points on the unit square in R^2.
        # The mean is (0,0), so each point has distance
        # sqrt(2), so the sum of square distances is
        # 4*2 = 8
        sum_squares = sigclust.compute_sum_of_square_distances_to_mean(self.test_data)
        self.assertAlmostEqual(sum_squares, 8, places=8)

    def test_compute_cluster_index(self):
        # Initialize points on the unit square in R^2.
        # Points on the left side are class 1, points
        # on the right side are class 2.
        # The within class sum of squares for each is 2.
        # The total sum of squares is 4*2=8.
        # So the cluster index is 1/2
        ci = sigclust.compute_cluster_index(self.test_data, self.test_labels)
        self.assertAlmostEqual(ci, 1.0/2, places=8)

    def test_SigClust(self):
        "Test SigClust end-to-end"
        class_1 = np.random.normal(size=(20, 2)) + np.array([10, 10])
        class_2 = np.random.normal(size=(20, 2)) + np.array([-10, -10])
        data = np.concatenate([class_1, class_2], axis=0)
        labels = np.concatenate([np.repeat(1, 20), np.repeat(2, 20)])
        sc = sigclust.SigClust(num_simulations=100)
        sc.fit(data, labels)
        self.assertEqual(sc.p_value, 0)
