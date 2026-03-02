import numpy as np

def lu_decomposition(A):
    """
    Performs LU decomposition of matrix A such that A = L @ U
    """

    n = A.shape[0]

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):

        # Upper triangular
        for k in range(i, n):
            sum_val = 0
            for j in range(i):
                sum_val += L[i, j] * U[j, k]

            U[i, k] = A[i, k] - sum_val

        # Lower triangular
        L[i, i] = 1

        for k in range(i+1, n):
            sum_val = 0
            for j in range(i):
                sum_val += L[k, j] * U[j, i]

            L[k, i] = (A[k, i] - sum_val) / U[i, i]

    return L, U