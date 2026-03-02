import numpy as np

from linalg.gaussian import gaussian_elimination
from linalg.lu import lu_decomposition
from linalg.eigen import power_method


# Test matrix
A = np.array([
    [4, 1],
    [2, 3]
])

b = np.array([9, 8])

# Gaussian
x = gaussian_elimination(A.copy(), b.copy())
print("Gaussian solution:", x)


# LU
L, U = lu_decomposition(A)

print("\nL:")
print(L)

print("\nU:")
print(U)


# Power Method
eigenvalue, eigenvector = power_method(A)

print("\nDominant Eigenvalue:")
print(eigenvalue)

print("\nEigenvector:")
print(eigenvector)

from linalg.qr import qr_decomposition

print("\nQR Decomposition:")

Q, R = qr_decomposition(A)

print("\nQ:")
print(Q)

print("\nR:")
print(R)

print("\nReconstructed A:")
print(Q @ R)