from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import numpy as np
import sigclust.helper_functions as helper
from tqdm.autonotebook import tqdm
import numba

@numba.njit(cache=True)
def MinimizeCriterionAlongPC(sorted_d_sums, sorted_c, n, g):
    "Given sorted pairwise distances and distances to mean, minimize the criterion"
    cis = np.zeros(n-1)

    # We will start with no points in class1, and add the info about the next point
    # one by one. We only need to keep track of sums.
    n1 = 0
    class1sum_numerator = 0
    class1sum_denominator = 0

    n2 = n
    class2sum_numerator = np.sum(sorted_d_sums)
    class2sum_denominator = np.sum(sorted_c)

    for i in range(n-1):
        n1 += 1
        n2 -= 1
        class1sum_numerator += sorted_d_sums[i]
        class2sum_numerator -= sorted_d_sums[i]
        class1sum_denominator += sorted_c[i]
        class2sum_denominator -= sorted_c[i]
        num = 1/(2*n1)**(g+1) * class1sum_numerator + 1/(2*n2)**(g+1)  * class2sum_numerator
        den = 1/n1**g * class1sum_denominator + 1/n2**g * class2sum_denominator
        cis[i] = num/den

    best_idx = np.argmin(cis)
    best_ci = cis[best_idx]
    return (best_idx, best_ci)

class Avg2Means(object):
    def __init__(self, max_components=1, progressbar=True):
        self.max_components = max_components
        self.disable_progressbar = not progressbar

    def fit(self, X, g=1.0):
        D2 = squareform(pdist(X, 'sqeuclidean'))  # matrix of pairwise square distances
        scores_matrix = PCA(n_components=self.max_components).fit_transform(X)
        n = X.shape[0]
        c = np.linalg.norm(X - X.mean(0), axis=1)**2

        results_per_pc = []
        for j in tqdm(range(self.max_components), desc='Loop over PCs', disable=self.disable_progressbar):
            pc_scores = scores_matrix[:, j]
            sort_order = np.argsort(pc_scores)
            sorted_pc_scores = pc_scores[sort_order]

            D2_sorted = D2[sort_order][sort_order]
            sums = np.sum(D2_sorted, axis=0) / 2

            best_idx, best_ci = MinimizeCriterionAlongPC(sums, c[sort_order], n, g)
            optimizing_score = sorted_pc_scores[best_idx]
            boolean_labels = (pc_scores > optimizing_score)
            labels = boolean_labels.astype(int) + 1

            results_per_pc.append((best_ci, labels))

        self.ci, self.labels = min(results_per_pc, key=lambda x: x[0])
        return self
