# 
# In this module implement these two functions:
# 1. solve_continuous_are
# 2. solve_ivp
#
# Make sure that they are compatible with their usage
# in modal_lqr.py.
#
# The Gradescope Autograder will call your implementation
# through functions:
#
# 1. simulate_closed_loop
# 2. simulate_open_loop

import numpy as np

class Solution:
    def __init__(self, t, y, success=True):
        self.t = t
        self.y = y
        self.success = success

def solve_ivp(fun, t_span, y0, t_eval=None, rtol=1e-8, atol=1e-10):
    """
    Basic implementation of solve_ivp using RK4.
    Ignores rtol/atol for simplicity.
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 100)
    y0 = np.asarray(y0)
    n_steps = len(t_eval)
    y = np.zeros((len(y0), n_steps))
    y[:, 0] = y0
    for i in range(1, n_steps):
        t = t_eval[i-1]
        h = t_eval[i] - t
        k1 = fun(t, y[:, i-1])
        k2 = fun(t + h/2, y[:, i-1] + h/2 * k1)
        k3 = fun(t + h/2, y[:, i-1] + h/2 * k2)
        k4 = fun(t + h, y[:, i-1] + h * k3)
        y[:, i] = y[:, i-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return Solution(t_eval, y)

def matrix_sign(X):

    tol = 1e-10

    max_iter = 100

    for _ in range(max_iter):

        X_inv = np.linalg.inv(X)

        X_new = (X + X_inv) / 2

        if np.linalg.norm(X_new - X) < tol:

            return X_new

        X = X_new

    return X

def solve_continuous_are(A, B, Q, R):
    """Solve the continuous-time algebraic Riccati equation.

    A^T P + P A - P B R^{-1} B^T P + Q = 0.

    Using Hamiltonian eigen-decomposition so this works without scipy.
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)

    H = np.block(
        [
            [A, -B @ R_inv @ B.T],
            [-Q, -A.T],
        ]
    )

    eigvals, eigvecs = np.linalg.eig(H)
    stable = np.real(eigvals) < 0
    if np.count_nonzero(stable) != n:
        raise np.linalg.LinAlgError("Unable to find stable invariant subspace for CARE")

    U = eigvecs[:, stable]
    Ux = U[:n, :]
    Uy = U[n:, :]

    if np.linalg.matrix_rank(Ux) < n:
        raise np.linalg.LinAlgError("Stable eigenvector subspace is singular for CARE")

    P = np.real(Uy @ np.linalg.inv(Ux))
    P = (P + P.T) / 2
    return P
