import warnings

import numpy as np
from scipy.special import expit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.data_fit import fit_poisson, fit_nb, fit_ZIP, fit_ZINB
from src.nll import get_nll
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# row of one fit: metrics + convergence (default values for a fit that raises an exception)
FIT_INFO = {'AIC': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'NLL': np.nan,
            'converged': False, 'valid': False, 'conv_warning': False, 'other_warning': False,
            'optimizer': None, 'alpha_hat': np.nan, 'pi_hat': np.nan, 'error': None}

failure_count = {}
alpha_estimates = {}
_reported = set()


def eval_models_single_run(X_train, X_test, y_train, y_test, source: str, p0: float):
    """Fits the models and collects metrics (AIC, RMSE, MAE, NLL) and convergence info; source = 'ZIP' or 'NB' (data-generating model), p0 = proportion of zeros."""

    results = {}
    models = {
        'Poisson': fit_poisson,
        'NB': fit_nb,
        'ZIP': fit_ZIP,
        'ZINB': fit_ZINB
    }

    # fitted models, NB starts from Poisson and ZINB from ZIP (even if they did not converge)
    fitted = {}

    for name, fit_fn in models.items():
        info = dict(FIT_INFO)

        # catch all warnings raised during the fit
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if name == 'NB':
                    model = fit_nb(X_train, y_train, fitted.get('Poisson'))
                elif name == 'ZINB':
                    model = fit_ZINB(X_train, y_train, fitted.get('ZIP'))
                else:
                    model = fit_fn(X_train, y_train)
                fitted[name] = model

                converged = bool(model.mle_retvals.get("converged", True))

                y_pred = model.predict(X_test)
                aic = model.aic
                rmse = root_mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                nll = get_nll(model, X_test, y_test)

                metrics = [aic, rmse, mae, nll]
                is_valid = converged and all(np.isfinite(m) for m in metrics)

                info.update(AIC=aic, RMSE=rmse, MAE=mae, NLL=nll, converged=converged, valid=is_valid,
                            optimizer=model.mle_settings['optimizer'])
                if 'alpha' in model.params.index:
                    info['alpha_hat'] = model.params['alpha']
                if 'inflate_const' in model.params.index:
                    info['pi_hat'] = expit(model.params['inflate_const'])

                if not is_valid:
                    _record_failure(source, p0, name)
                elif name in ('NB', 'ZINB'):
                    alpha_estimates.setdefault((source, p0, name), []).append(model.params['alpha'])

            except Exception as e:
                _record_failure(source, p0, name, f"{type(e).__name__}: {e}")
                info['error'] = f"{type(e).__name__}: {e}"

        # invalid fits are kept, only marked with valid = False
        info['conv_warning'] = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        info['other_warning'] = any(not issubclass(w.category, ConvergenceWarning) for w in caught)
        results[name] = info

    return results, failure_count, alpha_estimates


def _record_failure(source: str, p0: float, model: str, message: str | None = None) -> None:
    """Increments the failure counter; prints the error message only on the first occurrence."""
    key = (source, p0, model)
    failure_count[key] = failure_count.get(key, 0) + 1

    if message is not None and key not in _reported:
        _reported.add(key)
        print(f"source={source}, p0={p0}, model={model}: {message} (first occurrence, rest suppressed)")


def format_params_summary(p0_values: list[float], pi_values: list[float], alpha_values: list[float]) -> str:
    """Summary of pi and alpha found for each proportion of zeros."""
    lines = ["Params for zero proportion:", f"  {'p0':<6} {'pi':>8} {'alpha':>8}"]

    for p0, pi, alpha in zip(p0_values, pi_values, alpha_values):
        lines.append(f"  {p0:<6.3f} {pi:>8.4f} {alpha:>8.4f}")

    return "\n".join(lines)


def format_failure_summary(failures: dict, n_runs: int | None = None) -> str:
    """Summary of fit failures as a table."""
    if not failures:
        return "No fit failures."

    head = "Fit failures" + (f" (out of {n_runs} runs per p0):" if n_runs else ":")
    lines = [head, f"  {'source':<6} {'p0':<6} {'model':<8} {'failures':>8}"]

    for (source, p0, model), count in sorted(failures.items()):
        lines.append(f"  {source:<6} {p0:<6.3f} {model:<8} {count:>8}")

    return "\n".join(lines)


def format_alpha_summary(alphas: dict) -> str:
    """Summary of estimated alpha for NB and ZINB."""
    lines = ["Alpha (NB, ZINB):", f"  {'source':<6} {'p0':<6} {'model':<8} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}"]

    for (source, p0, model), values in sorted(alphas.items()):
        a = np.array(values)
        lines.append(f"  {source:<6} {p0:<6.3f} {model:<8} {a.mean():>8.4f} {np.median(a):>8.4f} {a.min():>8.4f} {a.max():>8.4f}")

    return "\n".join(lines)
