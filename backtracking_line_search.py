def backtracking_line_search(f, grad_f, x, d, alpha_init):
    """
    BACKTRACKING_LINE_SEARCH - Performs backtracking line search

    Inputs:
    f          - Function handle for objective function
    grad_f     - Function handle for gradient
    x          - Current point
    d          - Descent direction
    alpha_init - Initial step size

    Outputs:
    alpha      - Step size satisfying Armijo condition
    """
    rho = 0.5
    c = 1e-4
    alpha = alpha_init

    f_x = f(x)
    grad_f_x = grad_f(x)

    if grad_f_x.T @ d >= 0:
        print("Direction is not a descent direction!")
        alpha = 0
        return alpha
    
    iteration = 0
    max_backtrack_iter = 50

    while f(x + alpha * d) > f_x + c * alpha * grad_f_x.T @ d:
        alpha = rho * alpha
        iteration += 1

        if iteration > max_backtrack_iter:
            print("backtracking line serach reached maximum iterations")
            break

        if alpha < 1e-12:
            print("step size became too small")
            break

    if iteration > 0:
        print(f"Backtracking: {iteration} iterations, final alpha = {alpha:.2e}")
    return alpha
        



