import numpy as np
from steepest_descent import steepest_descent
from conjugate_gradient import conjugate_gradient
from newtons_method import newtons_method


print('=== RASTRIGIN FUNCTION ===\n')
print('f(x) = A*n + Σ[ x_i^2 - A*cos(2*π*x_i) ]\n')
print('Global minimum at (0,0) with f(0,0)=0\n')
print('This is a highly multimodal function with many local minima!\n\n')

A = 10
n = 2

def rastrigin(x):
  return A*n + np.sum(x**2 - A*np.cos(2*np.pi * x))

def gradient_rastrigin(x):
  return 2*x + A * np.sin(2 * np.pi * x) * (2 * np.pi)

def hessian_rastrigin(x):
  diag = 2 + A * (2 * np.pi) * np.cos(2 * np.pi * x) * (2 * np.pi)
  return np.diag(diag)

x0 = np.array([2.5, 3.0])
tolerance = 1e-6
max_iter = 1000

print('Starting point: (%.1f, %.1f)\n', x0[0], x0[1])
print('Starting function value: %.4f\n', rastrigin(x0))


print('1. Implement Conjugate Gradient Method\n')
[x_cg, f_cg, traj_cg, iter_cg] = conjugate_gradient(rastrigin, gradient_rastrigin, x0, tolerance, max_iter)

print('2. Implement Steepest Descent Method\n')
[x_sd, f_sd, traj_sd, iter_sd] = steepest_descent(rastrigin, gradient_rastrigin, x0, tolerance, max_iter) 

print('3. Implement Newton''s Method\n')
[x_nt, f_nt, traj_nt, iter_nt] = newtons_method(rastrigin, gradient_rastrigin, hessian_rastrigin, x0, tolerance, max_iter)






