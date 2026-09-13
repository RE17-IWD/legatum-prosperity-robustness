# How Robust Are the Legatum Prosperity Index Rankings?

Replication code for the revised manuscript *How Robust Are the Legatum Prosperity Index Rankings? A Monte Carlo and Sensitivity Analysis of Weighting, Normalization, and Aggregation* (Adrian Erlikhman and Ryan Erlikhman), resubmitted to the Journal of High School Science.

Every number, table and figure in the revised manuscript is produced by `src/analysis.py` and `src/figures.py` from the two Legatum data files named below, with a fixed seed (42).

## Data (not redistributed)

The Legatum data files are published by the Legatum Institute Foundation and are not included here, under its terms of use. Place them in `data/`, or point to them with environment variables.

| File | SHA-256 | Environment variable |
| :-- | :-- | :-- |
| `Dataset_Legatum_Prosperity_Index_2023.xlsx` | `8c789bd5ab881c12005f83e9b0c55ae81a68548dd18d3acb9fff9bec0eef631c` | `LPI_DATA_2023` |
| `Legatum_Prosperity_Index_2026_Data_Sheet.xlsx` | `54621ff5d00d0cf4afe6bdbcc47008acff2086cd87cf55c3c8777f819eed1718` | `LPI_DATA_2026` |

The 2023 edition has been withdrawn from the publisher's website. A copy can be checked without the authors: `pytest -q tests/` verifies the digest, that the equal-weighted mean of the twelve pillar scores reproduces all 167 published scores and ranks, and that the committed outputs regenerate.

## Run

```
pip install -r requirements.txt    # Python 3.12.10
python src/analysis.py      # all analyses; writes outputs/
python src/figures.py       # figures/fig1..fig8 (300 dpi JPEG)
pytest -q tests/            # reproducibility checks
```

## What is produced

- `outputs/results.json`: every quantity reported in the manuscript.
- `outputs/rank_uncertainty.csv`: per-country published score and rank, median simulated rank, 5th and 95th percentiles and width of the 90 percent rank interval (uniform prior).
- `outputs/sobol_indices.csv`: first-order and total-effect Sobol' indices with 95 percent bootstrap half-widths at N = 8192 and N = 1024.
- `outputs/realized_importance.csv`, `outputs/leverage.csv`: correlation ratios (5, 10, 20 bins and leave-one-out), squared correlations, standard deviations, exact leverage and its two terms.
- `outputs/network_centrality.csv`, `outputs/network_edges.csv`: partial-correlation network, edge tests and centrality.
- `outputs/country_profiles.csv`, `outputs/clusters.csv`: per-country Sobol' shares, local independence and cluster membership.
- `outputs/eigenvalues.csv`: eigenvalues and parallel-analysis thresholds.

## License

Code is released under the MIT License. The Legatum data remain subject to the Legatum Institute Foundation's terms of use.
