import pandas as pd

MODELS = ['Poisson', 'NB', 'ZIP', 'ZINB']
METRICS = ['AIC', 'RMSE', 'MAE', 'NLL']
# threshold for pi or alpha estimate "at the boundary" (model reduces to a simpler one)
BOUNDARY = 1e-4


def average_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics by (source, p0, model), valid fits only."""
    valid = df[df['valid']]
    return valid.groupby(['source', 'p0', 'model'], observed=True)[METRICS].mean().reset_index()


def aic_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Proportion of replications in which the model has the smallest AIC, by (source, p0)."""
    valid = df[df['valid']]
    idx_min_aic = valid.groupby(['source', 'p0', 'run'], observed=True)['AIC'].idxmin()
    best_models = valid.loc[idx_min_aic]

    counts = (
        best_models.groupby(['source', 'p0', 'model'], observed=True).size()
        .unstack(fill_value=0)
        .reindex(columns=MODELS, fill_value=0)
    )
    total_runs = df.groupby(['source', 'p0'], observed=True)['run'].nunique()

    return counts.divide(total_runs, axis=0).reset_index()


def convergence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of non-converged fits, fits with warnings and pi/alpha estimates at the boundary by (source, p0, model)."""
    df = df.copy()
    df['not_converged'] = ~df['converged']
    df['pi_boundary'] = df['pi_hat'] < BOUNDARY
    df['alpha_boundary'] = df['alpha_hat'] < BOUNDARY

    return (
        df.groupby(['source', 'p0', 'model'], observed=True)[['not_converged', 'conv_warning', 'other_warning', 'pi_boundary', 'alpha_boundary']]
        .sum()
        .reset_index()
    )
