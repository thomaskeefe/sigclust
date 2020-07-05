from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import sigclust.helper_functions as helper


class Avg2Means(object):
    def __init__(self):
        self.labels = None
        self.ci = None

    def fit(self, X, p=1.0):
        scores = PCA(n_components=1).fit_transform(X)
        scores_df = pd.DataFrame(scores)

        pc1_scores = scores_df.iloc[:, 0]

        def split_data_by_score(score):
            class_1 = X[pc1_scores <= score]
            class_2 = X[pc1_scores > score]
            return (class_1, class_2)

        def compute_avg_ci_p_by_score(score):
            class_1, class_2 = split_data_by_score(score)
            return compute_average_cluster_index_p_exp(class_1, class_2, p)

        # compute the cluster index corresponding to thresholding at each score
        cis = pc1_scores.map(compute_avg_ci_p_by_score)

        # get the PC1 score that corresponds to minimizing the CI
        score_threshold = pc1_scores[cis.argmin()]

        boolean_cluster_labels = (pc1_scores > score_threshold)
        self.labels = boolean_cluster_labels.astype(int) + 1  # turns the cluster labels into 1s and 2s
        self.labels.name = "labels"  # the default name for this pandas Series is not helpful; we give it a useful name
        self.ci = cis.min()


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
