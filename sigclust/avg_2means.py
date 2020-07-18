from sklearn.decomposition import PCA
import numpy as np
import sigclust.helper_functions as helper


class Avg2Means(object):
    def __init__(self):
        pass

    def fit(self, X, p=1.0):
        n = X.shape[0]
        pc1_scores = PCA(n_components=1).fit_transform(X)
        pc1_scores = pc1_scores.reshape(pc1_scores.size)  # reshape to 1-D
        pc1_sort_order = np.argsort(pc1_scores)
        ordered_pc1_scores = pc1_scores[pc1_sort_order]
        Y = np.copy(X)[pc1_sort_order, :]

        cis = []
        for i in range(n-1):  # cis will contain n-1 points
            class_1 = Y[:i+1, :]
            class_2 = Y[i+1:, :]
            ci = compute_average_cluster_index_p_exp(class_1, class_2, p)
            cis.append(ci)

        # get point on the PC1 vector that best splits the data
        optimizing_score = ordered_pc1_scores[np.argmin(cis)]
        boolean_cluster_labels = (pc1_scores > optimizing_score) # note, this follows the original index of the data, not the reordering.
        self.labels = boolean_cluster_labels.astype(int) + 1  # turns the cluster labels into 1s and 2s
        self.cis = cis
        self.ci = min(cis)


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
