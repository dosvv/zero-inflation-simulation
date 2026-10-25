import numpy as np
import pandas as pd
from src.data_generator import generate_zi_data, generate_od_data
from src.eval_run import eval_models_single_run, format_failure_summary, format_alpha_summary, format_params_summary
from src.graphs import plot_performance_figures
from src.params_for_zero_proportion import find_params
from src.table import format_table, format_aic_selection_table, format_convergence_table, format_params_table


def main():
    p0_values = [0.035, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    N = 500
    theta = [1, 0.03, -0.5]

    ROOT_SEED = 21

    # for each proportion of zeros p0: pi (zero inflation) and alpha (overdispersion) that produce it
    pi_values, alpha_values = find_params(p0_values, theta)
    print(format_params_summary(p0_values, pi_values, alpha_values))
    format_params_table(p0_values, pi_values, alpha_values)

    # the same proportion of zeros comes either from ZIP data (excess zeros) or from NB data (overdispersion)
    scenarios = {
        'ZIP': (generate_zi_data, pi_values),
        'NB': (generate_od_data, alpha_values),
    }
    seeds = np.random.SeedSequence(ROOT_SEED).spawn(len(scenarios) * len(p0_values) * N)

    all_results = []
    seed_idx = 0

    for source, (generate_data, params) in scenarios.items():
        for p0, param in zip(p0_values, params):
            for run in range(N):
                seed = seeds[seed_idx]
                seed_idx += 1
                X_train, X_test, y_train, y_test = generate_data(500, theta, param, seed)
                res, failure_count, alpha_estimates = eval_models_single_run(X_train, X_test, y_train, y_test, source, p0)

                for model_name, metrics in res.items():
                    all_results.append({
                        'source': source,
                        'p0': p0,
                        'run': run,
                        'model': model_name,
                        **metrics
                    })

    df = pd.DataFrame(all_results)
    # order in tables and figures
    df['source'] = pd.Categorical(df['source'], categories=['ZIP', 'NB'], ordered=True)
    df['model'] = pd.Categorical(df['model'], categories=['Poisson', 'NB', 'ZIP', 'ZINB'], ordered=True)

    format_table(df)
    format_aic_selection_table(df)
    format_convergence_table(df)
    plot_performance_figures(df)
    print(format_failure_summary(failure_count, N))
    print(format_alpha_summary(alpha_estimates))


if __name__ == "__main__":
    main()
