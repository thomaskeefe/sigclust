from sklearn.decomposition import PCA
import numpy as np
from sigclust.helper_functions import compute_sum_of_square_distances_to_mean, compute_sum_of_square_distances_to_point
from tqdm.autonotebook import tqdm
import numba

def EuclideanDistanceMatrix(X):
    "Matrix of pairwise square euclidean distances"
    # At least 10x faster than `pdist` in scipy.
    # This function doesn't benefit from numba acceleration
    # (actually makes it slower), at least on MKL-linked numpy.
    G = X @ X.T  # Gram matrix
    Gdiag = np.diag(G)
    return Gdiag - 2*G + Gdiag.reshape(-1,1)

# TODO move this to helper functions
def compute_wci(class_1, class_2, g=0.5):
    """Compute the weighted cluster index for the two-class clustering
    given by `labels`. To use an exponent other than 0.5 (i.e. WCI), set the parameter g."""
    n1 = class_1.shape[0]
    n2 = class_2.shape[0]
    if (n1 == 0) or (n2 == 0):
        return np.nan

    class_1_SSE = compute_sum_of_square_distances_to_mean(class_1)
    class_2_SSE = compute_sum_of_square_distances_to_mean(class_2)

    overall_mean = np.concatenate([class_1, class_2]).mean(axis=0)

    numerator = (1/n1)**g * class_1_SSE + (1/n2)**g * class_2_SSE
    denominator = (compute_sum_of_square_distances_to_point(class_1, overall_mean) / (n1**g) +
                   compute_sum_of_square_distances_to_point(class_2, overall_mean) / (n2**g) )

    return numerator/denominator


@numba.njit(cache=True)
def MinimizeWCIAlongPC(D_pi, sorted_r, n, g=0.5):
    """Given sorted pairwise distances and distances to mean, minimize the WCI.
    To use an exponent other than 0.5 (standard WCI), use the param g."""
    cis = np.zeros(n-1)

    # We will start with no points in class1, and add the info about the next point
    # one by one. We only need to keep track of sums.
    n1 = 0
    n2 = n
    Sd1 = 0
    Sd2 = np.sum(D_pi)
    Sr1 = 0
    Sr2 = np.sum(sorted_r)

    for k in range(n-1):
        d_pi_k = D_pi[k, :]
        n1 += 1
        n2 -= 1
        Sd1 += 2*np.sum(d_pi_k[:k])
        Sd2 -= 2*np.sum(d_pi_k[k:])
        Sr1 += sorted_r[k]
        Sr2 -= sorted_r[k]
        num = 0.5 * n1**(-g-1) * Sd1 + 0.5 * n2**(-g-1) * Sd2
        den = n1**(-g) * Sr1 + n2**(-g) * Sr2
        cis[k] = num/den

    best_idx = np.argmin(cis)
    best_ci = cis[best_idx]
    return (best_idx, best_ci)

class WCIClustering(object):
    def __init__(self, max_components=1, progressbar=True):
        self.max_components = max_components
        self.progressbar = progressbar

    def fit(self, X, g=0.5):
        X = np.array(X)
        D = EuclideanDistanceMatrix(X)
        scores_matrix = PCA(n_components=self.max_components).fit_transform(X)
        n = X.shape[0]
        r = np.sum((X - X.mean(0))**2, axis=1)

        results_per_pc = []
        for j in tqdm(range(self.max_components), desc='Loop over PCs', disable=(not self.progressbar)):
            pc_scores = scores_matrix[:, j]
            sort_order = np.argsort(pc_scores)
            sorted_pc_scores = pc_scores[sort_order]

            D_pi = D[sort_order][:, sort_order]
            sorted_r = r[sort_order]

            best_idx, best_ci = MinimizeWCIAlongPC(D_pi, sorted_r, n, g)
            optimizing_score = sorted_pc_scores[best_idx]
            boolean_labels = (pc_scores > optimizing_score)
            labels = boolean_labels.astype(int) + 1

            results_per_pc.append((best_ci, labels))

        self.ci, self.labels = min(results_per_pc, key=lambda x: x[0])
        return self
