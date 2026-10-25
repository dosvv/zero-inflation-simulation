# Research report – simulation code

Monte Carlo simulation for the research report *When Does Explicit Modeling of Zero Inflation Become Useful?*
For each proportion of zeros p0, `find_params` finds the value of π (ZIP) and α (NB)
that give the same proportion of zeros. For both data sources (ZIP and NB), 500 replications are run,
the Poisson, NB, ZIP and ZINB models are fitted and compared by AIC (training data) and RMSE, MAE and NLL
(test data). For every fit, convergence, warnings and the estimates of α and π are recorded.

## Running

```
git clone https://github.com/dosvv/zero-inflation-simulation.git
cd zero-inflation-simulation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

One command runs the simulation and produces the tables and figures (N = 500 takes about 10 minutes).
Settings (N, p0, θ, seed) are changed directly in `main.py`. The script prints π and α for each p0,
the number of fit failures and the estimates of α for NB and ZINB.

## Outputs (`output/`)

Created by `python main.py`; the LaTeX tables are not tracked in the repository.

- `tables/avg_metrics.tex` – average AIC, RMSE, MAE, NLL (data, p0, model)
- `tables/aic_selection.tex` – proportion of replications in which the model has the smallest AIC
- `tables/convergence.tex` – non-converged fits, warnings, estimates of π or α at the boundary (< 1e-4)
- `tables/params_table.tex` – π and α for each p0
- `figures/aic_selection.pdf`, `figures/aic_nll.pdf`, `figures/rmse_mae.pdf` – figures, data from ZIP on the left, from NB on the right

## Structure

- `main.py` – simulation settings and main loop
- `src/params_for_zero_proportion.py` – `find_params`, π and α for a given proportion of zeros
- `src/data_generator.py` – `generate_zi_data` (ZIP) and `generate_od_data` (NB)
- `src/data_fit.py` – model fitting
- `src/eval_run.py` – metrics and convergence for one replication, printed summaries
- `src/nll.py` – NLL on the test data
- `src/summary.py` – averages, AIC selection, convergence summary
- `src/table.py`, `src/graphs.py` – tables and figures
- `report_values.md` – values and settings used in the report
