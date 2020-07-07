from unittest import TestCase
import numpy as np
from sigclust import soft_thresholding

class TestSoftThresholdingMingYuan(TestCase):
    def setUp(self):
        self.eigenvalues = np.array([12.2, 9, 4.3, 3, 3, 2.1, 2, 0, 0])
        self.sig2b = 2
        self.thresholded_eigenvalues = soft_thresholding.soft_threshold_ming_yuan(self.eigenvalues, self.sig2b)

    def test_total_power_is_maintained(self):
        self.assertAlmostEqual(self.thresholded_eigenvalues.sum(), self.eigenvalues.sum())

    def test_all_eigenvalues_are_at_least_background_noise_level(self):
        self.assertTrue((self.thresholded_eigenvalues >= self.sig2b).all())

    def test_eigenvalue_diffs_are_maintained_down_to_background_noise_level(self):
        thresholded_eigenvalues_above_sig2b = self.thresholded_eigenvalues[self.thresholded_eigenvalues > self.sig2b]
        corresponding_original_eigenvalues = self.eigenvalues[self.thresholded_eigenvalues > self.sig2b]

        thresholded_diffs = np.diff(thresholded_eigenvalues_above_sig2b)
        original_diffs = np.diff(corresponding_original_eigenvalues)
        np.testing.assert_allclose(thresholded_diffs, original_diffs)

class TestSoftThresholdingHanwenHuang(TestCase):
    def setUp(self):
        self.eigenvalues = np.array([12.2, 9, 4.3, 3, 3, 2.1, 2, 0, 0])
        self.sig2b = 2
        self.thresholded_eigenvalues = soft_thresholding.soft_threshold_hanwen_huang(self.eigenvalues, self.sig2b)

    def test_all_eigenvalues_are_at_least_background_noise_level(self):
        self.assertTrue((self.thresholded_eigenvalues >= self.sig2b).all())

    def test_eigenvalue_diffs_are_maintained_down_to_background_noise_level(self):
        thresholded_eigenvalues_above_sig2b = self.thresholded_eigenvalues[self.thresholded_eigenvalues > self.sig2b]
        corresponding_original_eigenvalues = self.eigenvalues[self.thresholded_eigenvalues > self.sig2b]

        thresholded_diffs = np.diff(thresholded_eigenvalues_above_sig2b)
        original_diffs = np.diff(corresponding_original_eigenvalues)
        np.testing.assert_allclose(thresholded_diffs, original_diffs)
