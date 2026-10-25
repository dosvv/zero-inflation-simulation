import numpy as np


def get_nll(model_res, X_test, y_test):
    """Average NLL per observation of the test data."""
    y_arr = np.asarray(y_test).ravel()
    X_arr = np.asarray(X_test)

    dist = model_res.get_distribution(exog=X_arr)
    return -float(np.mean(dist.logpmf(y_arr)))
