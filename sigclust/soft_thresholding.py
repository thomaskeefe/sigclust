"Ming Yuan and Hanwen Huang's soft thresholding procedures for SigClust"
import numpy as np

def soft_threshold_hanwen_huang(eigenvalues, sig2b):
    "Soft threshold eigenvalues to background noise level sig2b according to Hanwen Huang's scheme"
    optimal_tau = _compute_tau(eigenvalues, sig2b)
    soft_thresholded_eigenvalues = _shift_and_threshold_eigenvalues(eigenvalues, optimal_tau, sig2b)
    return soft_thresholded_eigenvalues

def soft_threshold_ming_yuan(eigenvalues, sig2b):
    """Soft thresholds eigenvalues to background noise level sig2b using Ming Yuan's
    scheme, which maintains total power. Results in an anti-conservative SigClust
    when the relative size of the first eigenvalue is small."""

    # Starting with the smallest eigenvalue, sequentially bring eigenvalues up to
    # sig2b and distribute the difference equally over the larger eigenvalues
    # (which maintains the total power).
    d = len(eigenvalues)
    eigenvalues_asc = np.sort(eigenvalues)  # produces a copy
    for i in range(d-1):
        lambda_ = eigenvalues_asc[i]
        if lambda_ < sig2b:
            eigenvalues_asc[i] += (sig2b-lambda_)
            eigenvalues_asc[i+1:] -= (sig2b-lambda_) / (d-i-1)
        else:
            break

    # If this process has brought the largest eigenvalue below sig2b, then it is
    # impossible to threshold to sig2b while maintaining total power. In this
    # case the need to threshold to sig2b overrides the need to maintain total
    # power.
    eigenvalues_asc[d-1] = np.maximum(eigenvalues_asc[d-1], sig2b)

    thresholded_eigenvalues_desc = eigenvalues_asc[::-1]  # reverses order
    return thresholded_eigenvalues_desc

def _compute_tau(eigenvalues, sig2b):
    """Compute the tau that gives Hanwen Huang's soft thresholded eigenvalues, which
    maximizes the relative size of the first eigenvalue"""

    # NOTE: tau is found by searching between 0 and Ming Yuan's tilde_tau.
    tilde_tau = _compute_tilde_tau(eigenvalues, sig2b)
    tau_candidates = np.linspace(0, tilde_tau, 100)

    criteria = [_relative_size_of_1st_eigenvalue(
                   _shift_and_threshold_eigenvalues(eigenvalues, tau, sig2b)
                ) for tau in tau_candidates]

    optimal_tau = tau_candidates[np.argmax(criteria)]
    return optimal_tau

def _compute_tilde_tau(eigenvalues, sig2b):
    """Computes tilde_tau, the value of tau that gives Ming Yuan's soft
    thresholded eigenvalues, which maintain total power"""
    # NOTE: we compute Ming Yuan's soft thresholding estimates iteratively
    # and then back out what tilde_tau was.
    thresholded_eigenvalues = soft_threshold_ming_yuan(eigenvalues, sig2b)
    tilde_tau = max(0, eigenvalues.max() - thresholded_eigenvalues.max())
    return tilde_tau

def _shift_and_threshold_eigenvalues(eigenvalues, tau, sig2b):
    """Decrease the eigenvalues by the given tau, and threshold them at sig2b"""
    shifted_eigenvalues = eigenvalues - tau
    return np.maximum(shifted_eigenvalues, sig2b)

def _relative_size_of_1st_eigenvalue(eigenvalues):
    return eigenvalues.max() / eigenvalues.sum()
