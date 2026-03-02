import numpy as np

def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian elimination.

    Parameters:
        A : nxn matrix
        b : nx1 vector

    Returns:
        x : solution vector
    """

    A = A.astype(float)
    b = b.astype(float)

    n = len(b)

    # Forward elimination
    for i in range(n):

        # Pivot
        max_row = i + np.argmax(abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Eliminate
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x