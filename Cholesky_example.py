import numpy as np


A = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]])

eigenvalues = np.linalg.eigvals(A)
is_positive_definite = np.all(eigenvalues > 0)

print(f"eigenvalues: {eigenvalues}")
print(f"is positive definite?: {is_positive_definite}")

try:
    L = np.linalg.cholesky(A)
    print("Cholesky L matrix:\n", L)
except np.linalg.LinAlgError:
    print("Matrix is not positive definite, shouldn't make Cholesky.")
