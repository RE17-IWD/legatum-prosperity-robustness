"""
Reproducibility tests for the Legatum robustness analysis.

These tests confirm the two claims a reviewer cares about most:
  1. The published 2023 overall score is exactly the equal-weighted mean of
     the twelve pillar scores (baseline replication is exact).
  2. The headline Monte Carlo statistic is stable under the fixed seed.

Run:  pytest -q
The Legatum dataset must be available (see data/README.md); set LPI_DATA or
place the file in data/. If the dataset is absent these tests are skipped.
"""
import os
import numpy as np
import pandas as pd
import pytest

YEAR = 2023
PILLARS = ['Safety and Security', 'Personal Freedom', 'Governance', 'Social Capital',
           'Investment Environment', 'Enterprise Conditions', 'Infrastructure and Market Access',
           'Economic Quality', 'Living Conditions', 'Health', 'Education', 'Natural Environment']

DATA = os.environ.get(
    "LPI_DATA",
    os.path.join(os.path.dirname(__file__), "..", "data",
                 "Dataset_Legatum_Prosperity_Index_2023.xlsx"),
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA),
    reason="Legatum dataset not present; see data/README.md",
)


def _load():
    idx = pd.read_excel(DATA, sheet_name="Prosperity Index")
    pil = pd.read_excel(DATA, sheet_name="Pillars x 12")
    W = pil.pivot(index="area_code", columns="pillar_name", values=f"score_{YEAR}")[PILLARS]
    meta = idx.set_index("area_code")[[f"score_{YEAR}", f"rank_{YEAR}"]]
    meta.columns = ["off_score", "off_rank"]
    return meta.join(W)


def _ranks(x):
    return pd.Series(x).rank(ascending=False, method="min").to_numpy().astype(int)


def test_dataset_shape():
    df = _load()
    assert df.shape[0] == 167
    assert df[PILLARS].isna().sum().sum() == 0


def test_baseline_replication_exact():
    """Equal-weighted mean of the 12 pillars must reproduce the official score and rank."""
    df = _load()
    S = df[PILLARS].to_numpy()
    recon_score = S.mean(axis=1)
    recon_rank = _ranks(recon_score)
    max_dev = np.abs(recon_score - df["off_score"].to_numpy()).max()
    assert max_dev < 1e-6, f"score deviation too large: {max_dev}"
    assert (recon_rank == df["off_rank"].to_numpy()).all(), "ranks do not match official"


def test_monte_carlo_headline_stable():
    """Under the fixed seed, the average rank shift (flat Dirichlet) is the published value."""
    df = _load()
    S = df[PILLARS].to_numpy()
    base_rank = _ranks(S.mean(axis=1))
    rng = np.random.default_rng(42)
    M = 10000
    Wts = rng.dirichlet(np.ones(12), size=M)
    sc = S @ Wts.T
    ranks = np.empty_like(sc, dtype=int)
    for m in range(sc.shape[1]):
        ranks[:, m] = _ranks(sc[:, m])
    avg_shift = float(np.mean(np.abs(ranks - base_rank[:, None])))
    # published headline: 5.73 (allow a small tolerance for platform RNG differences)
    assert 5.5 < avg_shift < 5.95, f"avg rank shift outside expected range: {avg_shift}"
