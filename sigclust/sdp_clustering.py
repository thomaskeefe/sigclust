import cvxpy as cp
from sigclust.helper_functions import split_data
from sigclust.wci_clustering import EuclideanDistanceMatrix, compute_wci
import numpy as np
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


class SDPWCIClustering:
    def __init__(self, g, solver=None):
        self.g = g
        self.solver = solver

        if solver=='MOSEK' or solver is cp.MOSEK:
            import mosek
            self.mosek_params = {mosek.iparam.intpnt_solve_form: mosek.solveform.dual}

    def __repr__(self):
        return f"GClustering, g = {self.g}"

    def _setup_sdp(self, X):
        X = np.array(X)
        g = self.g
        D = EuclideanDistanceMatrix(X)
        c = np.sum((X - np.mean(X, axis=0))**2, axis=1)

        n = X.shape[0]
        self.Z = cp.Variable((n, n), symmetric=True)
        self.z = cp.Variable(n)
        self.n_minor = cp.Parameter(name='n_minor')
        self.alpha1 = cp.Parameter(name='alpha1')
        self.alpha2 = cp.Parameter(name='alpha2')
        self.gamma1 = cp.Parameter(name='gamma1')
        self.gamma2 = cp.Parameter(name='gamma2')

        z1t = cp.reshape(self.z, (n, 1)) @ np.ones((1, n))
        _1zt = z1t.T
        _11t = np.ones((n,n))

        Uz = self.alpha1 * cp.trace(D @ (self.Z + z1t + _1zt + _11t)) + self.alpha2 * cp.trace(D @ (self.Z - z1t - _1zt + _11t))
        Lz = self.gamma1 * c @ (1 + self.z) + self.gamma2 * c @ (1 - self.z)

        constraints = [
            Uz <= Lz,
            cp.bmat([[np.ones((1,1)), cp.reshape(self.z, (1, n)), ],
                     [cp.reshape(self.z, (n, 1)), self.Z]]) >> 0,
            cp.diag(self.Z) == np.ones(n),
            cp.sum(self.z) == 2*self.n_minor - n,
            (2*self.n_minor - n)*self.z - self.Z@np.ones(n) == np.zeros(n),
            self.Z + _11t + z1t + _1zt >= 0,
            self.Z + _11t - z1t - _1zt >= 0,
            self.Z - _11t + z1t - _1zt <= 0,
            self.Z - _11t - z1t + _1zt <= 0
        ]

        objective = 1  # i.e. a feasibility problem rather than a minimization
        self.prob = cp.Problem(cp.Minimize(objective), constraints)

    def _optimize_over_xi(self, n_minor, n_major, tol):
        "Repeatedly solve the SDP using binary search over xi in [0,1]"
        self.n_minor.value = n_minor
        self.alpha1.value = 1 / (2*n_minor**(self.g+1))
        self.alpha2.value = 1 / (2*n_major**(self.g+1))

        L, U = 0, 1  # bounds for binary search on xi

        maxiter = int(np.ceil(np.log2(1/tol)))
        last_optimal = None

        for it in range(maxiter):
            xi = (L+U)/2
            self.gamma1.value = xi * 2 / (n_minor**self.g)
            self.gamma2.value = xi * 2 / (n_major**self.g)

            try:
                if self.solver is cp.MOSEK:
                    self.prob.solve(solver=self.solver, mosek_params=self.mosek_params)
                else:
                    self.prob.solve(solver=self.solver)

            except Exception as e:
                # If we're here, the solver had an error, which is not
                # the same as "infeasible" but we do the same thing.
                L = (L+U)/2
                LOG.info(e)
                continue

            if self.prob.status == 'optimal':

                # hold on to the last optimal solution in case we hit maxiter
                last_optimal = self.z.value, self.Z.value, xi

                if U - L < tol:
                    return last_optimal
                U = (L+U)/2

            elif self.prob.status == 'infeasible':
                L = (L+U)/2

        if last_optimal is not None:
            return last_optimal
        else:
            raise ValueError("Problem could not be made feasible within the iterations")


    def fit(self, X, tol=.1):
        self._setup_sdp(X)

        n = X.shape[0]
        search_bound = n // 2

        self.results_ci = np.repeat(np.nan, search_bound)
        self.results_labels = np.tile(np.nan, (search_bound, n))
        self.results_z = np.tile(np.nan, (search_bound, n))
        self.results_Z = np.tile(np.nan, (search_bound, n, n))
        self.results_xi = np.tile(np.nan, search_bound)

        with logging_redirect_tqdm():  # makes sure logging plays nice with tqdm
            for i in tqdm(range(search_bound), leave=False):
                n_minor = i + 1
                n_major = n - n_minor

                best_z, best_Z, best_xi = self._optimize_over_xi(n_minor, n_major, tol)

                labels = singularvector_round(best_Z)

                if len(np.unique(labels)) != 2:
                    ci = 1
                    labels = np.ones(n)
                else:
                    ci = compute_wci(*split_data(X, labels), self.g)

                self.results_ci[i] = ci
                self.results_labels[i, :] = labels
                self.results_z[i, :] = best_z
                self.results_Z[i, :, :] = best_Z
                self.results_xi[i] = best_xi
