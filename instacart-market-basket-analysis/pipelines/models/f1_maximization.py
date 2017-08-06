# Based on https://www.kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n by https://www.kaggle.com/mmueller
# and https://www.kaggle.com/cpmpml/f1-score-expectation-maximization-in-o-n/ by https://www.kaggle.com/cpmpml.
# This kernel implements the O(n^2) F1-Score expectation maximization algorithm presented in:
# Ye, N., Chai, K., Lee, W., and Chieu, H.  Optimizing F-measures: A Tale of Two Approaches. In ICML, 2012.

import numpy as np
from numba import jit


@jit
def get_expectations(P):
    expectations = []
    P.sort(reverse=True)
    n = len(P)
    DP_C = np.zeros((n + 2, n + 1))

    DP_C[0][0] = 1.0
    for j in range(1, n):
        DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

    for i in range(1, n + 1):
        DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
        for j in range(i + 1, n + 1):
            DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

    DP_S = np.zeros((2 * n + 1, ))
    for i in range(1, 2 * n + 1):
        DP_S[i] = 1. / (1. * i)
    for k in range(n + 1)[::-1]:
        f1 = 0
        for k1 in range(n + 1):
            f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
        for i in range(1, 2 * k - 1):
            DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
        expectations.append(f1)
    expectations.reverse()
    return np.array(expectations)


def maximize_expected_f1(P):
    expectations = get_expectations(P)
    best_k = expectations.argmax()
    max_f1 = expectations[best_k]
    return best_k, max_f1
