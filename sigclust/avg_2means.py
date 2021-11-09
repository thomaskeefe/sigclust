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
<<<<<<< HEAD
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
=======
            minimum, labels = self.minimize_along_pc(X, scores[:, j], p)
            results_per_pc.append((minimum, labels))

        self.ci, self.labels = min(results_per_pc, key=lambda x: x[0])
        return self

    def minimize_along_pc(self, X, pc_scores, p):
        n = X.shape[0]
        sort_order = np.argsort(pc_scores)
        ordered_pc_scores = pc_scores[sort_order]
        Y = np.array(X)[sort_order, :]

        cis = []
        for i in tqdm(range(n-1), desc='Loop within PC', disable=self.disable_progressbar):
            # cis will contain n-1 points
            class_1 = Y[:i+1, :]
            class_2 = Y[i+1:, :]
            ci = compute_average_cluster_index_p_exp(class_1, class_2, p)
            cis.append(ci)

        # get point on the PC that best splits the data
        optimizing_score = ordered_pc_scores[np.argmin(cis)]
        boolean_cluster_labels = (pc_scores > optimizing_score) # note, this follows the original index of the data, not the reordering.
        labels = boolean_cluster_labels.astype(int) + 1  # turns the cluster labels into 1s and 2s
        return min(cis), labels


def compute_average_cluster_index_p_exp(class_1, class_2, p):
    """Compute the average cluster index for the two-class clustering
    given by `labels`, and using the exponent p"""
    n1 = class_1.shape[0]
    n2 = class_2.shape[0]
    if (n1 == 0) or (n2 == 0):
        return np.nan

    class_1_SSE = helper.compute_sum_of_square_distances_to_mean(class_1)
    class_2_SSE = helper.compute_sum_of_square_distances_to_mean(class_2)

    overall_mean = np.concatenate([class_1, class_2]).mean(axis=0)

    numerator = (1/n1)**p * class_1_SSE + (1/n2)**p * class_2_SSE
    denominator = (helper.compute_sum_of_square_distances_to_point(class_1, overall_mean) / (n1**p) +
                   helper.compute_sum_of_square_distances_to_point(class_2, overall_mean) / (n2**p) )

    return numerator/denominator
>>>>>>> master
