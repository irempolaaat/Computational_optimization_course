import numpy as np


A1 = np.array([[4, 1, 0], [1, 5, 2], [0, 2, 6]])

A2 = np.array([[1, 2], [2, 1]])

print('\n=== POSITIVE DEFINITE MATRIX CHECK ===')

print('\n1.USING EIGENVALUES')
print('----------------------')
#Eigenvalues - All must be > 0 for PD matrices

eig_A1 = np.linalg.eigvals(A1)
print('\nExample 1 eigenvalues: ')
print(f'{eig_A1}')

if all(eig_A1 > 0):
    print('All eigenvalues > 0 - Matrix is positive definite')
else:
    print('Some eigenvalues ≤ 0 - Matrix is not positive definite')

eig_A2 = np.linalg.eigvals(A2)
print('\nExample 2 eigenvalues: ')
print(f'{eig_A2}')

if all(eig_A2 > 0):
    print('All eigenvalues > 0 - Matrix is positive definite\n')
else:
    print('Some eigenvalues ≤ 0 - Matrix is not positive definite\n')


print('2.USING SYLVESTER''S CRITERION')
print('-------------------------------')
#Sylvester's Criterion - All leading principal minors > 0

def sylvester_criterion(A):
    for i in range(1, A.shape[0] + 1):
        minor = A[:i, :i]
        det = np.linalg.det(minor)
        if det <= 0:
            return False
    
    return True

print('\nExample 1 - Sylvesters Criterion:')

if sylvester_criterion(A1):
    print('All leading principal minors > 0 - Matrix is positive definite')
else:
    print('Some minors ≤ 0 - Matrix is not positive definite')

print('\nExample 2 - Sylvesters Criterion:')

if sylvester_criterion(A2):
    print('All leading principal minors > 0 - Matrix is positive definite')
else:
    print('Some minors ≤ 0 - Matrix is not positive definite')


print('\n3. USING QUADRATIC FORM TEST')
print('-------------------------------')
#Quadratic Form - xᵀAx > 0 for all non-zero x

def test_quadratic_form(A, test_name):
    print(f'{test_name}')
    n = A.shape[0]
    num_tests = 5
    all_positive = True

    for i in range(1, num_tests):
        x = np.random.randn(n, 1)
        quadratic_form = x.T @ A @ x
        print(f'Test {i}: x.TAx = {quadratic_form}')
        if quadratic_form > 0:
            print(' Yes')
        else:
            print(' No')
            all_positive = False

    if all_positive:
        print('All quadratic forms > 0\n')
    else:
        print('Some quadratic forms <= 0\n')

test_quadratic_form(A1, '\nExample 1 - Quadratic Form Test')
test_quadratic_form(A2, '\nExample 2 - Quadratic Form Test')


print('4. USING cholesky() FUNCTION (Most Reliable)')
print('--------------------------------------')
#cholesky() - Most reliable method, fails if matrix is not PD

print('\nExample 1 - Positive Definite:')
try:
    L = np.linalg.cholesky(A1)
    print('cholesky() succeeded - Matrix is positive definite\n')
except np.linalg.LinAlgError:
    print("Matrix is not positive definite, shouldn't make Cholesky.\n")

print('\nExample 2 - Positive Definite:')
try:
    L = np.linalg.cholesky(A2)
    print('cholesky() succeeded - Matrix is positive definite\n')
except np.linalg.LinAlgError:
    print('cholesky() failed - Matrix is not positive definite\n')
