import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


def generate_zi_data(n_samples: int, theta: list[float], pi: float, seed: np.random.SeedSequence):
    """Generates count data with excess zeros (Poisson + share pi of zeros), returns TRAIN and TEST sets."""
    # each replication has its own independent random stream
    rng = np.random.default_rng(seed)
    true_theta = np.array(theta)

    x1 = rng.normal(15, 7, size=n_samples)
    x2 = rng.binomial(1, 0.2, size=n_samples)

    X = pd.DataFrame({'Temp': x1, 'Rain': x2})
    X_const = sm.add_constant(X)

    log_mu = X_const @ true_theta
    mu = np.exp(log_mu)

    y_star = rng.poisson(mu, size=n_samples)

    if pi == 0.0:
        y = y_star
    else:
        is_zero = rng.binomial(1, pi, size=n_samples)
        y = np.where(is_zero == 1, 0, y_star)

    X_train, X_test, y_train, y_test = train_test_split(X_const, y, test_size=0.25, random_state=int(rng.integers(2**31)))

    return X_train, X_test, y_train, y_test


def generate_od_data(n_samples: int, theta: list[float], alpha: float, seed: np.random.SeedSequence):
    """Generates overdispersed count data (NB with parameter alpha), returns TRAIN and TEST sets."""
    # each replication has its own independent random stream
    rng = np.random.default_rng(seed)
    true_theta = np.array(theta)

    x1 = rng.normal(15, 7, size=n_samples)
    x2 = rng.binomial(1, 0.2, size=n_samples)

    X = pd.DataFrame({'Temp': x1, 'Rain': x2})
    X_const = sm.add_constant(X)

    log_mu = X_const @ true_theta
    mu = np.exp(log_mu)

    if alpha < 0:
        raise ValueError('alpha must be non-negative')
    if alpha > 0.0:
        r = 1.0 / alpha
        p = 1.0 / (1 + alpha * mu)

        y = rng.negative_binomial(n=r, p=p)
    else:
        y = rng.poisson(lam= mu)

    X_train, X_test, y_train, y_test = train_test_split(X_const, y, test_size=0.25, random_state=int(rng.integers(2**31)))

    return X_train, X_test, y_train, y_test
