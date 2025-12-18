import numpy as np
from backtracking_line_search import backtracking_line_search

def steepest_descent(fun, grad, x0, tol, max_iter):

    x = x0.copy()
    alpha = 1.0

    f_history = np.zeros((max_iter, 1))
    trajectory = np.zeros((len(x0), max_iter))
    
    k = 0
    for k in range(max_iter):
        trajectory[:, k] = x
        f_history[k] = fun(x)

        g = grad(x)
        d = -g

        alpha = backtracking_line_search(fun, grad, x, d, alpha)

        x += alpha * d

        if np.linalg.norm(g) < tol:
           break

    x_opt = x
    f_history = f_history[:, k+1]
    trajectory = trajectory[:, k+1]
    iterations = k + 1

    return [x_opt, f_history, trajectory, iterations]