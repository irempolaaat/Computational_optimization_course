import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def bisection_search(phi, a, b, epsilon):
    """
     BISECTIONSEARCH - Find critical points using bisection method
     Returns:
      result - Estimated critical point
      dphi   - Function handle to derivative (for plotting)
    """
    if a < b:
        aa = a
        bb = b
    else:
        aa = b
        bb = a

    z = sp.symbols('z')

    if isinstance(phi, sp.Basic):
        phi_sym = phi
    else:
        phi_sym = phi(z)
    
    dphi_sym = sp.diff(phi_sym, z)

    phi_num = sp.lambdify(z, phi_sym, 'numpy')
    dphi_num = sp.lambdify(z, dphi_sym, 'numpy')
    
    u = (aa + bb)/2
    print('Iter\ta\tb\tdphi(u)\n')

    iter = 0
    max_iter = 100

    while (bb - aa) > epsilon and iter < max_iter :
        iter += 1
        y = dphi_num(u)
        
        print(f'{iter}\t{aa}f\t{bb}\t{y}\n')

        if abs(y) <= epsilon:
            break

        if y < 0:
            aa = u
        else:
            bb = u

        u = (aa + bb)/2

    result = u

    print(f'Found critical point at x = {result}\n')

    x_vals = np.linspace(float(a), float(b), 100)

    y_vals_phi = phi_num(x_vals)

    if np.isscalar(y_vals_phi):
      y_vals_phi = np.full_like(x_vals, y_vals_phi)

    y_vals_dphi = dphi_num(x_vals)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x_vals, y_vals_phi, 'b-', label='φ(x)')
    plt.plot(result, phi_num(result), 'ro', label='Critical Point') 
    plt.title(f'Function φ(x) | Critical point at x = {result:.4f}')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x_vals, y_vals_dphi, 'r-', label="φ'(x)")
    plt.axhline(0, color='black', linestyle='--') 
    plt.plot(result, 0, 'ro')
    plt.title("Derivative φ'(x)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout() 
    plt.show()

    return result, dphi_num
z = sp.symbols('z')
phi = z**3 - 2*z**2 + z

[x_crit, dphi] = bisection_search(phi, 0, 2, 1e-6)

print(f"Derivative at solution: {dphi(x_crit)}")


