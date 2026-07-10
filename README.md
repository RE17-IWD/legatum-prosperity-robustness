How Robust Are the Legatum Prosperity Index Rankings?

A Monte Carlo and variance-based sensitivity analysis of weighting, normalization, and aggregation choices in the 2023 Legatum Prosperity Index.

**Authors:** Michael Tarekegn, Adrian Erlikhman, and Ryan Erlikhman

## Overview

The Legatum Prosperity Index ranks 167 countries by an equal-weighted mean of twelve pillar scores. This repository audits how much of the 2023 ranking survives when the equal weights, the normalization method, and the aggregation rule are treated as uncertain modeling choices rather than fixed defaults.

The analysis:

- Exactly replicates the published 2023 index from public pillar-score data (maximum deviation $5 \times 10^{-11}$, all 167 ranks reproduced).
- Propagates weight uncertainty through 10,000 Monte Carlo draws under three priors (uniform, near-equal, and grid).
- Runs a 14-input Sobol' global sensitivity analysis (Saltelli design, 16,384 evaluations) over the twelve pillar weights plus normalization and aggregation selectors.
- Compares each pillar's nominal weight (1/12) to its realized importance (main-effect correlation ratio, η²).

## Repository Structure
├── data/           # Instructions for obtaining the Legatum pillar-score data (not redistributed)
├── figures/        # Generated figures (fig1–fig5)
├── outputs/        # Generated results (CSV/JSON), see below
├── src/            # Analysis pipeline (analysis.py, figures.py)
├── tests/          # Pytest suite: exact replication + Monte Carlo stability
└── LICENSE
├── requirements.txt

## Data

The Legatum Prosperity Index dataset is distributed by the Prosperity Institute at https://index.prosperity.com and is **not redistributed** in this repository, per its terms of use. See `data/README.md` for instructions on obtaining the pillar-score file yourself and placing it in the expected location.

## How to Run

```bash
pip install -r requirements.txt
python src/analysis.py
python src/figures.py
```

Both scripts read from a single fixed seed (42). The full pipeline completes in a few minutes on a laptop.

## What Gets Produced

Running the pipeline populates `outputs/` with:

| File | Contents |
|---|---|
| `rank_uncertainty.csv` | Per-country 90% rank intervals under each of the three weight priors |
| `sobol_indices.csv` | First-order and total-effect Sobol' indices for all fourteen inputs |
| `weights_vs_importance.csv` | Nominal vs. realized importance (η²) for each of the twelve pillars |
| `fragility_by_tier.csv` | Rank-interval width broken out by published-rank tier |
| `summary.json` | Headline statistics (replication deviation, $\bar{R}_S$ per prior, correlations) |

and `figures/` with the five figures used in the paper (rank-shift comparison, per-country uncertainty band, interval-width-vs-rank scatter, Sobol' index bar chart, realized-vs-nominal importance chart).

## Tests

```bash
pytest tests/
```

The test suite verifies exact replication of the published index (score deviation, rank reproduction) and the stability of the headline Monte Carlo statistic ($\bar{R}_S$) across runs.

## Method Summary

Three priors govern the weight-uncertainty exercise: a uniform prior over the weight simplex, a near-equal prior concentrated around 1/12, and a discrete grid prior mimicking how index designers typically adjust weights. A separate joint-perturbation exercise varies all fourteen inputs (twelve weights, one normalization selector, one aggregation selector) simultaneously via Saltelli sampling. Full methodological detail is in the accompanying paper.

## License

Code is released under the MIT License (see `LICENSE`). The Legatum Prosperity Index dataset itself is the property of the Prosperity Institute and is not included in or covered by this license.s
