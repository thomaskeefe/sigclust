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

    def test_situation_when_not_enough_total_power(self):
        # If there's not enough total power to maintain total power
        # while thresholding to sig2b, then the need to threshold to sig2b
        # overrides the need to maintain total power.
        new_sig2b = self.eigenvalues.max()
        d = len(self.eigenvalues)
        thresholded_eigenvalues = soft_thresholding.soft_threshold_ming_yuan(self.eigenvalues, new_sig2b)
        np.testing.assert_allclose(thresholded_eigenvalues, np.repeat(new_sig2b, d))

    def test_situation_when_all_eigenvalues_above_sig2b(self):
        # When they're all above sig2b, they should be returned as is.
        eigenvalues = np.array([8,7,6,5])
        sig2b = 3
        thresholded_eigenvalues = soft_thresholding.soft_threshold_ming_yuan(eigenvalues, sig2b)
        np.testing.assert_allclose(thresholded_eigenvalues, eigenvalues)

    def test_same_results_as_matlab_function(self):
        matlab_results = [11.4200, 8.2200, 3.5200, 2.2200, 2.2200, 2.0, 2.0, 2.0, 2.0]
        np.testing.assert_allclose(self.thresholded_eigenvalues, matlab_results)


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

    def test_same_results_as_matlab_function(self):
        matlab_results =  [11.4278, 8.2278, 3.5278, 2.2278, 2.2278, 2.0, 2.0, 2.0, 2.0]
        np.testing.assert_allclose(self.thresholded_eigenvalues, matlab_results)

class TestBackgroundNoiseEstimation(TestCase):
    def test_same_results_as_matlab_function(self):
        # the matlab function is madSM.m
        # the data is magic(4)
        data = np.array([[16, 5,  9,  4],
                         [2, 11,  7, 14],
                         [3, 10,  6, 15],
                         [13,  8, 12, 1]]).T
        matlab_result = 5.930408874022407  # this is on the scale of SD, not variance
        matlab_sig2b = matlab_result**2
        self.assertAlmostEqual(soft_thresholding.estimate_background_noise(data), matlab_sig2b)
