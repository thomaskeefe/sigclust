import sigclust
import numpy as np
from unittest import TestCase

class Tests(TestCase):
    def test_compute_cluster_index(self):
        # Initialize points on the unit square in R^2.
        # Points on the left side are class 1, points
        # on the right side are class 2.
        # The within class sum of squares for each is 2.
        # The total sum of squares is 4*2=8.
        # So the cluster index is 1/2
        data = np.array([[-1, 1], [-1, -1], [1, 1], [1, -1]])
        classes = np.array([1, 1, 2, 2])
        ci = sigclust.compute_cluster_index(data, classes)
        self.assertAlmostEqual(ci, 1.0/2, places=8)
