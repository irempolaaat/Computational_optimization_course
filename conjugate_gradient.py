import numpy as np
from backtracking_line_search import backtracking_line_search

def conjugate_gradient(fun, grad, x0, tol, max_iter):
    
    max_iter = int(max_iter)
    
    x = x0.copy()
    alpha = 1
    g = grad(x)
    d = -g
    f_history = np.zeros((max_iter, 1))
    trajectory = np.zeros((len(x0), max_iter))
    k = 0
    for k in range(max_iter):

        trajectory[:, k] = x
        f_history[k] = fun(x)

        alpha = backtracking_line_search(fun, grad, x, d, alpha)

        x_old = x.copy()
        x += alpha * d
        g_old = g.copy()
        g = grad(x)

        if np.linalg.norm(g) < tol:
            break

        beta = (g.T @ g) / (g_old.T @ g_old)
        d = -g + beta * d

    x_opt = x
    f_history = f_history[:, k+1]
    trajectory = trajectory[:, k+1]
    iterations = k

    return [x_opt, f_history, trajectory, iterations]