import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.summary import MODELS, average_metrics, aic_selection

STYLE_CONFIG = {
    "Poisson": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "NB": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "ZIP": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "ZINB": {"color": "#d62728", "marker": "d", "linestyle": ":"},
}
# left column = data generated from ZIP, right column = data generated from NB
SOURCES = {"ZIP": "Data generated from ZIP", "NB": "Data generated from NB"}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"


def _plot_models(ax, sub: pd.DataFrame, column) -> None:
    """One curve per model, sub is already filtered to one data source."""
    for model in MODELS:
        if "model" in sub.columns:
            data = sub[sub["model"] == model].sort_values("p0")
            y = data[column]
        else:
            data = sub.sort_values("p0")
            y = data[model]

        ax.plot(
            data["p0"],
            y,
            label=model,
            color=STYLE_CONFIG[model]["color"],
            marker=STYLE_CONFIG[model]["marker"],
            linestyle=STYLE_CONFIG[model]["linestyle"],
            linewidth=1.5,
            markersize=5,
        )

    ax.set_xlabel(r"$p_0$ (proportion of zeros)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)


def _save(fig, name: str) -> None:
    # save as PDF for direct use in LaTeX
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name)
    plt.close(fig)


def plot_aic_selection(df: pd.DataFrame) -> None:
    """Proportion of replications with the smallest AIC vs p0, panels ZIP data | NB data."""
    proportions = aic_selection(df)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)

    for ax, (source, source_title) in zip(axes, SOURCES.items()):
        _plot_models(ax, proportions[proportions["source"] == source], None)
        ax.set_title(source_title, fontsize=11)
        ax.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel("Proportion of smallest AIC", fontsize=10)
    axes[0].legend(frameon=True, fontsize=9)
    _save(fig, "aic_selection.pdf")


def plot_metric_grid(df: pd.DataFrame, metrics: list[str], name: str) -> None:
    """2x2 grid, rows = metrics, columns = ZIP data | NB data."""
    df_avg = average_metrics(df)

    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.0, 6.0))

    for row, metric in enumerate(metrics):
        for col, (source, source_title) in enumerate(SOURCES.items()):
            ax = axes[row, col]
            _plot_models(ax, df_avg[df_avg["source"] == source], metric)
            ax.set_title(source_title, fontsize=11)
        axes[row, 0].set_ylabel(f"Average {metric}", fontsize=10)

    axes[0, 0].legend(frameon=True, fontsize=9)
    _save(fig, name)


def plot_performance_figures(df: pd.DataFrame) -> None:
    plot_aic_selection(df)
    plot_metric_grid(df, ["AIC", "NLL"], "aic_nll.pdf")
    plot_metric_grid(df, ["RMSE", "MAE"], "rmse_mae.pdf")
