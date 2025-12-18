import math


def golden_section_search(phi, a, b, c, epsilon):
    """
    GOLDENSECTIONSEARCH - Find minimum using golden section search
     Inputs:
       phi     - Function handle to minimize
       a, b, c - Initial bracket [a, b, c] with a < b < c
       epsilon - Convergence tolerance
     Output:
       result  - Estimated minimum location
    """
    tau = (1 + math.sqrt(5))/2
    aa = a
    dd = c

    if (b-a) < (c-b):
        bb = b
        cc = b + (1 - 1/tau) * (c - b)
    
    else:
        bb = b - (1 - 1/tau) * (b - a)
        cc = b

    print('Iter\taa\tbb\tcc\tdd\n')
    iter = 0

    while (dd - aa) > epsilon:
        iter += 1
        print(f'{iter}\t{aa:.4f}\t{bb:.4f}\t{cc:.4f}\t{dd:.4f}\n')

        if phi(cc) < phi(bb):
            dd = cc
            cc = bb
            bb = cc/tau +(1 - 1/tau) * aa
        else:
            aa = bb
            bb = cc
            bb = bb/tau + (1 - 1/tau) * dd

    if phi(cc) < phi(bb):
        result = (bb + cc) / 2
    else:
        result = (bb + cc) / 2

    print(f'Final result: {result} \n')

    return result

def my_function(x):
    return x**2 - 3*x + 2

a, b, c = 0, 1, 5
tol = 1e-5

minimum_x = golden_section_search(my_function, a, b, c, tol)

print(f"minimum value x = {minimum_x:.4f}")
print(f"f({minimum_x:.4f}) = {my_function(minimum_x):.4f}")