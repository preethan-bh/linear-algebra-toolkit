import numpy as np

def qr_decomposition(A):
    """
    Performs QR decomposition using Gram-Schmidt process.
    A = Q @ R

    Returns:
        Q : orthogonal matrix
        R : upper triangular matrix
    """

    n, m = A.shape

    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):

        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)

        if R[j, j] == 0:
            raise ValueError("Matrix has linearly dependent columns")

        Q[:, j] = v / R[j, j]

    return Q, R