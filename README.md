# Robustness and Sensitivity Analysis of the Legatum Prosperity Index

**Authors:** Adrian Erlikhman and Ryan Erlikhman (Los Angeles Center for Enriched Studies)

Replication code for the paper *How Robust Are the Legatum Prosperity Index Rankings? A Monte Carlo and Variance-Based Sensitivity Analysis of Weighting, Normalization, and Aggregation Choices.*

The analysis treats the equal weighting of the index's twelve pillars, the normalization method, and the aggregation function as uncertain inputs, propagates them through 10,000 Monte Carlo draws, and decomposes the variance in each country's rank with first-order and total-effect Sobol' indices. It first verifies that an equal-weighted mean of the published pillar scores reproduces all 167 official 2023 ranks exactly.

## Data

This repository does not include the Legatum dataset, which is redistributed under the Prosperity Institute's terms. Download it yourself:

1. Go to the Legatum Prosperity Index resources page at index.prosperity.com.
2. Download the full 2023 dataset Excel file, `Dataset_Legatum_Prosperity_Index_2023.xlsx`.
3. Place it in the `data/` folder, or set `export LPI_DATA=/path/to/file.xlsx`. See `data/README.md`.

## How to run

```
pip install -r requirements.txt
python src/analysis.py     # baseline replication, Monte Carlo, Sobol'; writes outputs/
python src/figures.py      # regenerates all five figures and the refined importance table
```

Results are reproducible because a fixed seed (42) is set in both scripts.

To verify reproducibility, run the test suite (requires the dataset in `data/`):

```
pytest -q
```

The tests confirm exact baseline replication and the stability of the headline Monte Carlo statistic.

## What gets produced

- `outputs/rank_uncertainty.csv` per-country median rank, 5th and 95th percentiles, IQR, 90 percent range, fragility flag
- `outputs/sobol_indices.csv` first-order and total-effect Sobol' indices for all 14 inputs
- `outputs/weights_vs_importance.csv` nominal weight versus realized importance per pillar
- `outputs/fragility_by_tier.csv` width of the 90 percent rank interval by rank tier
- `outputs/summary.json` headline numbers
- `figures/fig1..fig5.png` the paper's figures

## Method summary

- Baseline: equal-weighted arithmetic mean of the 12 pillar scores, verified against published scores and ranks (max deviation 5e-11; Spearman and Kendall both 1.000000).
- Uncertainty analysis: 10,000 draws under three weighting schemes (flat Dirichlet, concentrated Dirichlet, Legatum-discrete on the set 0.5, 1, 1.5, 2); reports median rank, 90 percent intervals, average rank shift, and Spearman and Kendall distributions.
- Sensitivity analysis: Saltelli sampling, base N = 1024, 14 inputs (12 pillar weights plus normalization and aggregation), N(k+2) = 16,384 evaluations, via SALib; Sobol' indices computed per country and variance-weighted across countries.
- Weights versus importance: first-order correlation ratio (eta squared) of each pillar with the overall score.

## Citation

If you use this code, please cite the paper (preprint: [SSRN URL]) and this repository (archived at [ZENODO DOI]).

## License

Code released under the MIT License. The Legatum data is not included and remains subject to the Prosperity Institute's terms of use.
