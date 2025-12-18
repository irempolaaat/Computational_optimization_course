import numpy as np
import matplotlib.pyplot as plt


def modified_secant(g_func, xcurr, xnew, uncert):
    max_iter = 50
    iter = 0

    print('\nIter\tx_current\tx_new\tg(x_new)\n')
    print('------------------------------------------------\n')

    while iter < max_iter:
        g_curr = g_func(xcurr)
        g_new = g_func(xnew)

        print(f'{iter:d}\t{xcurr:.4f}\t{xnew:.4f}\t{g_new:.3e}\n')

        if abs(xnew - xcurr) < abs(xcurr) * uncert:
            break

        if abs(g_new - g_curr) < 1e-15:
            print('Warning: Small function difference, stopping\n')
            break

        xnext = xnew - g_new * (xnew - xcurr) / (g_new- g_curr)

        xcurr = xnew
        xnew = xnext
        iter += 1

    x = xnew
    v = g_func(xnew)

    return x, v
g = lambda x: (2*x - 1)**2 + 4 * (4 - 1024*x)**4

print('=== Finding Root of g(x) = (2x - 1)^2 + 4(4 - 1024x)^4 ===\n\n')

print('Function values at key points:\n')
test_points = [0, 0.001, 0.0039, 0.00390625, 0.004, 0.01, 1]

for i in range(len(test_points)):
    x_val = test_points[i]
    g_val = g(x_val)
    print(f'x = {x_val:.4f}, g(x) = {g_val:.4f}')

true_root = 4/1024
print(f'\nTheoretical minimum: x = 4/1024 = {true_root:.4f}\n')
print(f'g(true_root) = {g(true_root):.4f}')

print('\n=== Using Modified Secant Method ===\n')

x1 = 0.00390624
x2 = 0.00390626

print(f'Initial points: x1 = {x1:.4f}, x2 = {x2:.4f}\n')
print(f'g(x1) = {g(x1):.3e}, g(x2) = {g(x2):.3e}\n')

[x_final, g_final] = modified_secant(g, x1, x2, 1e-12)
print(f'\nFinal result: x = {x_final:.4f}, g(x) = {g_final:.3e}\n')

error = abs(x_final - true_root)
print(f'Error from theoretical root: {error:.3e}\n')

x_plot = np.linspace(0.0039062, 0.0039063, 1000)
g_plot = g(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, g_plot, 'b-', linewidth=2, label='g(x)')
plt.plot(x_final, g_final, 'ro', markersize=8, markerfacecolor='red', label='Secant Solution')
plt.plot(true_root, 0, 'kx', markersize=10, linewidth=2, label='Theoretical Minimum')

plt.axhline(0, color='black', linestyle='--', linewidth=1, label='g(x)=0')

plt.grid(True)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('g(x) near the true minimum')

plt.xlim(0.0039062, 0.0039063)

plt.legend()
plt.show()

