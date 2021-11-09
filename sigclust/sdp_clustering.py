import cvxpy as cp
from sigclust.helper_functions import split_data, compute_sum_of_square_distances_to_mean, compute_sum_of_square_distances_to_point
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tqdm.autonotebook import tqdm
import logging
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: big todo: Alex says the binary search over xi may be able to be
# relaxed with an SDP constraint, but this would be an even more relaxed
# problem. We should find out if it's better.

# TODO: could the search be made faster taking xi=.5, finding
# any n_min that achieve xi<.5, then only search among those
# n_min's for the xi <.25?

# TODO: can this be replaced by np.outer?
def cp_outer(x, y):
    "cvxpy-friendly vector outer product"
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError

    return cp.reshape(x, (x.size, 1)) @ cp.reshape(y, (1, y.size))

def _solve_sdp(D, c, n_minor, n_major, g, xi):
    """D: matrix of squared pairwise distances
       c: vector of square distances to the data mean
       n_minor, n_major: number of points in each class
       g = exponent g
    """
    n = n_minor + n_major
    Z = cp.Variable((n, n), symmetric=True)
    z = cp.Variable(n)
    alpha1 = 1 / (2*n_minor**(g+1))
    alpha2 = 1 / (2*n_major**(g+1))
    gamma1 = 2 / (n_minor**g)
    gamma2 = 2 / (n_major**g)
    z1t = cp_outer(z, np.ones(n))
    _1zt = z1t.T
    _11t = cp_outer(np.ones(n), np.ones(n))

    Uz = alpha1 * cp.trace(D @ (Z + z1t + _1zt + _11t)) + alpha2 * cp.trace(D @ (Z - z1t - _1zt + _11t))
    Lz = gamma1 * c @ (1 + z) + gamma2 * c @ (1 - z)

    constraints = [
        Uz <= xi * Lz,
        cp.bmat([[np.ones((1,1)), cp.reshape(z, (1, n)), ],
                 [cp.reshape(z, (n, 1)), Z]]) >> 0,
        cp.diag(Z) == np.ones(n),
        cp.sum(z) == 2*n_minor - n,
        (2*n_minor - n)*z - Z@np.ones(n) == np.zeros(n),
        Z + _11t + z1t + _1zt >= 0,
        Z + _11t - z1t - _1zt >= 0,
        Z - _11t + z1t - _1zt <= 0,
        Z - _11t - z1t + _1zt <= 0
    ]

    objective = 1  # i.e. a feasibility problem rather than a minimization
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    return (prob, Z, z)


def optimize_over_xi(D, c, n_minor, n_major, g, tol=.1):
    L = 0
    U = 1
    maxiter = int(np.ceil(np.log2(1/tol)))
    last_optimal = None

    for it in range(maxiter):
        xi = (L+U)/2
        try:
            prob, Z, z = _solve_sdp(D, c, n_minor, n_major, g, xi)

        except Exception as e:
            # If we're here, the solver had an error, which is not
            # the same as "infeasible" but we do the same thing.
            L = (L+U)/2
            LOG.info(e)
            continue

        if prob.status == 'optimal':

            # hold on to the last optimal solution in case we hit maxiter
            last_optimal = prob, z, Z, xi

            if U - L < tol:
                return last_optimal
            U = (L+U)/2

        elif prob.status == 'infeasible':
            L = (L+U)/2

    if last_optimal is not None:
        return last_optimal
    else:
        raise ValueError("Problem could not be made feasible within the iterations")

def randomized_round(Z):
    n = Z.shape[0]
    rowsums = np.sum(Z, axis=0)
    labels = np.zeros(n)
    labels[rowsums > 0] = 1
    labels[rowsums <= 0] = 2
    return labels

def dot_cols_with_first_sing_vect(Z):
    u, s, vh = np.linalg.svd(Z)
    singular_vector = u[:, 0]
    dotproducts = Z @ singular_vector
    return dotproducts

def singularvector_round(Z):
    n = Z.shape[0]
    dotproducts = dot_cols_with_first_sing_vect(Z)
    labels = np.zeros(n)
    labels[dotproducts > 0] = 1
    labels[dotproducts <= 0] = 2
    return labels


def compute_average_cluster_index_g_exp(class_1, class_2, g):
    """Compute the average cluster index for the two-class clustering
    given by `labels`, and using the exponent g"""
    n1 = class_1.shape[0]
    n2 = class_2.shape[0]
    if (n1 == 0) or (n2 == 0):
        return np.nan

    class_1_SSE = helper.compute_sum_of_square_distances_to_mean(class_1)
    class_2_SSE = helper.compute_sum_of_square_distances_to_mean(class_2)

    overall_mean = np.concatenate([class_1, class_2]).mean(axis=0)

    numerator = (1/n1)**g * class_1_SSE + (1/n2)**g * class_2_SSE
    denominator = (helper.compute_sum_of_square_distances_to_point(class_1, overall_mean) / (n1**g) +
                   helper.compute_sum_of_square_distances_to_point(class_2, overall_mean) / (n2**g) )

    return numerator/denominator

class GClustering:
    def __init__(self, g):
        self.g = g

    def __repr__(self):
        return f"GClustering, g = {self.g}"

    def fit(self, X, tol=.1):
        n, d = X.shape
        D = squareform(pdist(X, 'euclidean'))**2
        c = norm(X - np.mean(X, axis=0), axis=1)**2

        search_bound = int(np.floor(n/2))

        self.results_ci = np.repeat(np.nan, search_bound)
        self.results_labels = np.tile(np.nan, (search_bound, n))
        self.results_z = np.tile(np.nan, (search_bound, n))
        self.results_Z = np.tile(np.nan, (search_bound, n, n))
        self.results_xi = np.tile(np.nan, search_bound)

        with logging_redirect_tqdm():  # makes sure logging plays nice with tqdm
            for i in tqdm(range(search_bound), leave=False):
                n_minor = i + 1
                n_major = n - n_minor

                best_problem, best_z, best_Z, best_xi = optimize_over_xi(D, c, n_minor, n_major, self.g, tol=tol)

                labels = singularvector_round(best_Z.value)

                if len(np.unique(labels)) != 2:
                    ci = 1
                    labels = np.ones(n)
                else:
                    ci = compute_average_cluster_index_g_exp(*split_data(X, labels), self.g)

                self.results_ci[i] = ci
                self.results_labels[i, :] = labels
                self.results_z[i, :] = best_z.value
                self.results_Z[i, :, :] = best_Z.value
                self.results_xi[i] = best_xi
