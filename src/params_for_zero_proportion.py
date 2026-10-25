from scipy.optimize import brentq
import numpy as np


def find_params(p0_vals: list[float], theta: list[float|int]):
    rng = np.random.default_rng(0)
    N = 1_000_000
    X_ref = np.column_stack([np.ones(N), rng.normal(15, 7, N), rng.binomial(1, 0.2, N)])
    mu_ref = np.exp(X_ref @ np.array(theta))
    m0 = np.mean(np.exp(-mu_ref))

    params = [[], []]
    for p0 in p0_vals:
        pi = (p0 - m0) / (1 - m0)
        alpha = brentq(lambda a: np.mean(np.exp(-np.log1p(a * mu_ref) / a)) - p0, 1e-6, 100)
        params[0].append(float(pi))
        params[1].append(float(alpha))
    return params
