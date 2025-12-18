import numpy as np
from backtracking_line_search import backtracking_line_search


def steepest_descent(f, grad_f, x0, max_iter, tol):

    x = x0
    alpha = 1.0
    n = len(x0)

    f_history = np.zeros((int(max_iter), 1))
    grad_history = np.zeros((int(max_iter), 1))
    alpha_history = np.zeros((int(max_iter), 1))
    x_history = np.zeros((int(max_iter), n))
    
    iter_idx = -1
    print('Iter\tf(x)\t\t||grad||\talpha\n')
    print('----------------------------------------\n')

    for iter_idx in range(int(max_iter)):

        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        f_val = f(x)

        f_history[iter_idx] = f_val
        grad_history[iter_idx] = grad_norm
        alpha_history[iter_idx] = alpha
        x_history[iter_idx, :] = x.flatten()

        if iter_idx <= 10 or np.mod(iter_idx, 10) == 0 or grad_norm < tol:
            print(f"{iter_idx}\t{f_val:.2e}\t{grad_norm:.2e}\t{alpha:.2e}")
        
        if grad_norm < tol:
            print(f"\n✓ Converged after {iter_idx} iterations (||grad|| < {tol:.0e})")            
            break

        if f_val < 1e-15:
            print(f"\n✓ Converged after {iter_idx} iterations (f(x) < 1e-15)")
            break

        d = -g

        alpha = backtracking_line_search(f, grad_f, x, d, alpha)

        x_prev = x
        x += alpha * d

        if np.linalg.norm(x - x_prev) < 1e-15:
            print("stagnation detected at iteration %d\n", iter)
            break

    final_iter = iter_idx + 1
    f_history = f_history[:final_iter]
    grad_history = grad_history[:final_iter]
    x_history = x_history[:final_iter]

    x_opt = x

    if iter == max_iter:
        print("Maximum iterations reached (||grad|| = %.2e)\n", grad_norm)






    



    return (x_opt, f_history, grad_history, alpha_history, x_history)
