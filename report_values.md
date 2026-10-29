# Values for the report

## Data-generating model
- True coefficients: b0 = 1, b1 = 0.03 (Temperature), b2 = −0.5 (Rain), log link `ln(mu) = b0 + b1 x1 + b2 x2`
- Explanatory variables: x1 ~ N(15, 7²), x2 ~ Bernoulli(0.2), regenerated in every replication
- Average mu ≈ 4.02 (range ≈ 1.0–11.3), proportion of zeros of the Poisson model alone m0 ≈ 0.0310
- ZIP data: y = 0 with probability π, otherwise Poisson(mu) (`generate_zi_data`)
- NB data: NB2 with parameter α, Var = mu + α mu², no zero inflation (`generate_od_data`)

## Sample size and split
- T = 500 observations per replication
- 75 % / 25 % split (375 training, 125 test) via `sklearn.model_selection.train_test_split(test_size=0.25)`,
  random shuffling, `random_state` is drawn from the rng of the replication (`rng.integers(2**31)`)
- AIC is computed on the training data, RMSE, MAE and NLL (average per observation) on the test data

## Replications and seeds
- N = 500 replications for each combination (data, p0) → 2 × 10 × 500 = 10 000 datasets, 40 000 fits
- `np.random.SeedSequence(21).spawn(2 * 10 * N)`, each replication gets its own independent stream,
  order: ZIP data → NB data, then p0 ascending, then run 0..N−1
- The seed of a replication depends on N (index = (source · 10 + p0_idx) · N + run), so results for N = 5 are not a subset of N = 500

## Calibration of π and α to p0
- p0 ∈ {0.035, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50}
- Reference sample of 10⁶ draws of (x1, x2), `default_rng(0)`, mu_ref = exp(X θ) (Monte Carlo average over the distribution of X)
- π in closed form: π = (p0 − m0) / (1 − m0), where m0 = mean(exp(−mu_ref))
- α numerically: `scipy.optimize.brentq` solves mean((1 + α mu_ref)^(−1/α)) = p0 on the interval [1e-6, 100]

| p0 | π (ZIP) | α (NB) |
|---|---|---|
| 0.035 | 0.0041 | 0.0254 |
| 0.040 | 0.0092 | 0.0561 |
| 0.050 | 0.0196 | 0.1141 |
| 0.075 | 0.0454 | 0.2494 |
| 0.100 | 0.0712 | 0.3802 |
| 0.150 | 0.1228 | 0.6478 |
| 0.200 | 0.1744 | 0.9390 |
| 0.300 | 0.2776 | 1.6450 |
| 0.400 | 0.3808 | 2.6155 |
| 0.500 | 0.4840 | 4.0437 |

## Optimizer settings (statsmodels, ML)
- Poisson: `sm.Poisson`, Newton-Raphson (default), maxiter 35 (default)
- NB: `sm.NegativeBinomialP(p=2)`, L-BFGS-B, maxiter 500, start = Poisson estimate + α = 0.05, bound α ≥ 1e-6, `skip_hessian=True`
- ZIP: `sm.ZeroInflatedPoisson`, logit inflation with a constant only (π does not depend on x), BFGS, maxiter 150, statsmodels default start
- ZINB: `sm.ZeroInflatedNegativeBinomialP` (NB2), logit inflation with a constant only, L-BFGS-B, maxiter 500,
  start = ZIP estimate + α = 0.05, bound α ≥ 1e-6, `skip_hessian=True`
- π̂ = expit(inflate_const)
- Valid fit = `mle_retvals["converged"]` and all metrics finite; averages and AIC selection use valid fits only,
  the selection proportion is divided by the number of all replications

## Versions
- Python 3.13.7
- numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, matplotlib 3.11.2, jinja2 3.1.6
