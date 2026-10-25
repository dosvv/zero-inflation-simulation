import re

import pandas as pd
from pathlib import Path

from src.summary import average_metrics, aic_selection, convergence_summary

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "tables"

# A data row starts with the data-generating model (ZIP/NB) and p0, the header starts with \textbf{...}.
DATA_ROW = re.compile(r'^\s*(ZIP|NB)\s*&\s*(\d+\.\d+)\s*&')
# Data row of a table where blocks are split by data source (ZIP/NB) only.
SOURCE_ROW = re.compile(r'^\s*(ZIP|NB)\s*&')

HEADERS = {
    'source': r'\textbf{data}',
    'p0': r'\textbf{$p_0$}',
    'model': r'\textbf{model}',
}


def _split_blocks(latex_code: str, row_pattern: re.Pattern) -> str:
    """Separates blocks of rows with different (source, p0), or source only, by \\midrule."""
    new_lines = []
    prev_key = None

    for line in latex_code.split('\n'):
        match = row_pattern.match(line)
        if match:
            current_key = match.groups()
            if prev_key is not None and current_key != prev_key:
                new_lines.append(r'\midrule')
            prev_key = current_key

        new_lines.append(line)

    return '\n'.join(new_lines)


def _save(latex_code: str, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / name, "w") as f:
        f.write(latex_code)


def format_table(df: pd.DataFrame) -> None:
    """Average AIC, RMSE, MAE, NLL by (data, p0, model)."""
    table = average_metrics(df)
    table['p0'] = table['p0'].map(lambda x: f"{x:.3f}")
    table = table.rename(columns={**HEADERS, **{m: rf'\textbf{{{m}}}' for m in ['AIC', 'RMSE', 'MAE', 'NLL']}})

    latex_code = table.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption='Average model evaluation metrics over Monte Carlo replications '
                '(data = model the data were generated from)',
        label="tab:mc_results",
        float_format="%.3f",
    )

    _save(_split_blocks(latex_code, DATA_ROW), "avg_metrics.tex")


def format_aic_selection_table(df: pd.DataFrame) -> None:
    """Proportion of replications in which the model has the smallest AIC, by (data, p0)."""
    table = aic_selection(df)
    table['p0'] = table['p0'].map(lambda x: f"{x:.3f}")
    table = table.rename(columns={**HEADERS, **{m: rf'\textbf{{{m}}}' for m in ['Poisson', 'NB', 'ZIP', 'ZINB']}})

    latex_code = table.to_latex(
        index=False,
        escape=False,
        caption='Proportion of replications in which the model has the smallest AIC '
                '(data = model the data were generated from)',
        label="tab:aic_selection",
        float_format="%.3f",
        position="htbp",
    )
    latex_code = latex_code.replace(r'\begin{table}[htbp]', '\\begin{table}[htbp]\n\\centering', 1)

    _save(_split_blocks(latex_code, SOURCE_ROW), "aic_selection.tex")


def format_convergence_table(df: pd.DataFrame) -> None:
    """Counts of non-converged fits, fits with warnings and estimates at the boundary by (data, p0, model)."""
    table = convergence_summary(df)
    table['p0'] = table['p0'].map(lambda x: f"{x:.3f}")
    table = table.rename(columns={
        **HEADERS,
        'not_converged': r'\textbf{not conv.}',
        'conv_warning': r'\textbf{conv. warn.}',
        'other_warning': r'\textbf{other warn.}',
        'pi_boundary': r'\textbf{$\hat\pi < 10^{-4}$}',
        'alpha_boundary': r'\textbf{$\hat\alpha < 10^{-4}$}',
    })

    latex_code = table.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption=r'Number of non-converged fits, fits with a ConvergenceWarning or another warning, '
                r'and fits with the estimate $\hat\pi$ or $\hat\alpha$ at the boundary',
        label="tab:convergence",
    )

    _save(_split_blocks(latex_code, DATA_ROW), "convergence.tex")


def format_params_table(p0_values: list[float], pi_values: list[float], alpha_values: list[float]) -> None:
    """Table of pi (ZIP data) and alpha (NB data) found for each proportion of zeros p0."""
    table = pd.DataFrame({
        r'\textbf{$p_0$}': [f"{p0:.3f}" for p0 in p0_values],
        r'\textbf{$\pi$ (ZIP)}': pi_values,
        r'\textbf{$\alpha$ (NB)}': alpha_values,
    })

    latex_code = table.to_latex(
        index=False,
        escape=False,
        caption=r'Parameters $\pi$ and $\alpha$ giving the proportion of zeros $p_0$',
        label="tab:params",
        float_format="%.4f",
        position="htbp",
    )

    # to_latex does not add \centering, the table would otherwise be left-aligned
    latex_code = latex_code.replace(r'\begin{table}[htbp]', '\\begin{table}[htbp]\n\\centering', 1)

    _save(latex_code, "params_table.tex")
