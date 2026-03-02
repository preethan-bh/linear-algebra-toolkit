import numpy as np

def power_method(A, max_iterations=100, tolerance=1e-10):
    """
    Finds dominant eigenvalue and eigenvector using Power Method.

    Parameters:
        A : square matrix
        max_iterations : number of iterations
        tolerance : convergence threshold

    Returns:
        eigenvalue
        eigenvector
    """

    n = A.shape[0]

    # random initial vector
    b = np.random.rand(n)

    # normalize
    b = b / np.linalg.norm(b)

    eigenvalue = 0

    for _ in range(max_iterations):

        # multiply
        b_new = np.dot(A, b)

        # normalize
        b_new = b_new / np.linalg.norm(b_new)

        # Rayleigh quotient gives eigenvalue
        eigenvalue_new = np.dot(b_new, np.dot(A, b_new))

        # check convergence
        if abs(eigenvalue_new - eigenvalue) < tolerance:
            break

        b = b_new
        eigenvalue = eigenvalue_new

    return eigenvalue, b