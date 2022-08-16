from sklearn.decomposition import PCA
import numpy as np
import sigclust.helper_functions as helper
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

@numba.njit(cache=True)
def MinimizeCI_g_AlongPC(D_pi, sorted_r, n, g):
    "Given sorted pairwise distances and distances to mean, minimize the CI_g"
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
        num = (2*n1)**(-g-1) * Sd1 + (2*n2)**(-g-1) * Sd2
        den = n1**(-g) * Sr1 + n2**(-g) * Sr2
        cis[k] = num/den

    best_idx = np.argmin(cis)
    best_ci = cis[best_idx]
    return (best_idx, best_ci)

class Avg2Means(object):
    def __init__(self, max_components=1, progressbar=True):
        self.max_components = max_components
        self.progressbar = progressbar

    def fit(self, X, g=1.0):
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

            best_idx, best_ci = MinimizeCI_g_AlongPC(D_pi, sorted_r, n, g)
            optimizing_score = sorted_pc_scores[best_idx]
            boolean_labels = (pc_scores > optimizing_score)
            labels = boolean_labels.astype(int) + 1

            results_per_pc.append((best_ci, labels))

        self.ci, self.labels = min(results_per_pc, key=lambda x: x[0])
        return self
