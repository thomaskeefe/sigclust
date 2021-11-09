from sklearn.decomposition import PCA
import numpy as np
import sigclust.helper_functions as helper
from tqdm.autonotebook import tqdm


class Avg2Means(object):
    def __init__(self, max_components=1, progressbar=True):
        self.max_components = max_components
        self.disable_progressbar = not progressbar

    def fit(self, X, p=1.0):
        scores = PCA(n_components=self.max_components).fit_transform(X)

        results_per_pc = []
        for j in tqdm(range(self.max_components), desc='Loop over PCs', disable=self.disable_progressbar):
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
