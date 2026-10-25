import statsmodels.api as sm
import numpy as np


def fit_poisson(X_train, y_train):
    poisson_model = sm.Poisson(y_train, X_train)
    return poisson_model.fit(disp=0)


def fit_nb(X_train, y_train, poisson_res=None):
    nb_model = sm.NegativeBinomialP(y_train, X_train, p=2)

    # reuse the already fitted Poisson model, otherwise fit it here
    if poisson_res is None:
        poisson_res = fit_poisson(X_train, y_train)

    # start from Poisson, alpha = 0.05 and bound alpha > 1e-6 (same as ZINB)
    start_params = np.append(poisson_res.params, 0.05)
    bounds = [(None, None)] * (len(start_params) - 1) + [(1e-6, None)]

    return nb_model.fit(
        start_params=start_params,
        method="lbfgs",
        maxiter=500,
        bounds=bounds,
        disp=0,
        skip_hessian=True,
        warn_convergence=True,
    )


def fit_ZIP(X_train, y_train):
    zip_model = sm.ZeroInflatedPoisson(y_train, X_train, inflation="logit")
    return zip_model.fit(method="bfgs", maxiter=150, disp=0)


def fit_ZINB(X_train, y_train, zip_res=None):
    zinb_model = sm.ZeroInflatedNegativeBinomialP(y_train, X_train, inflation="logit")

    # reuse the already fitted ZIP model, otherwise fit it here
    if zip_res is None:
        zip_res = fit_ZIP(X_train, y_train)

    # start from ZIP, alpha = 0.05 and bound alpha > 1e-6 (same as NB)
    start_params = np.append(zip_res.params, 0.05)
    bounds = [(None, None)] * (len(start_params) - 1) + [(1e-6, None)]

    return zinb_model.fit(
        start_params=start_params,
        method="lbfgs",
        maxiter=500,
        bounds=bounds,
        disp=0,
        skip_hessian=True,
        warn_convergence=True,
    )
