"""
Reproducibility tests for the revised Legatum robustness analysis.

  pytest -q tests/

They check, against the two Legatum data files, the claims the paper rests on:
  1. the files are byte-identical to the ones analysed (SHA-256);
  2. the equal-weighted mean of the twelve published pillar scores reproduces every
     published 2023 score and rank;
  3. the leverage identity (eq4) and the effective-weight constraint (eq17) hold on the data;
  4. the three-domain hierarchy is algebraically flat equal weighting;
  5. the committed outputs match a fresh computation (pillar statistics, correlation ratios,
     and the uniform-prior headline, which regenerates exactly under seed 42).
If a data file is absent the tests that need it are skipped.
"""
import hashlib, json, os
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _find(name, env):
    for p in (os.environ.get(env, ""), os.path.join(ROOT, "data", name), os.path.join(ROOT, name)):
        if p and os.path.exists(p):
            return p
    return None
DATA23 = _find("Dataset_Legatum_Prosperity_Index_2023.xlsx", "LPI_DATA_2023")
DATA26 = _find("Legatum_Prosperity_Index_2026_Data_Sheet.xlsx", "LPI_DATA_2026")
SHA23 = "8c789bd5ab881c12005f83e9b0c55ae81a68548dd18d3acb9fff9bec0eef631c"
SHA26 = "54621ff5d00d0cf4afe6bdbcc47008acff2086cd87cf55c3c8777f819eed1718"
OUT = os.path.join(ROOT, "outputs")
PILLARS = ['Safety and Security', 'Personal Freedom', 'Governance', 'Social Capital',
           'Investment Environment', 'Enterprise Conditions', 'Infrastructure and Market Access',
           'Economic Quality', 'Living Conditions', 'Health', 'Education', 'Natural Environment']
need23 = pytest.mark.skipif(DATA23 is None, reason="2023 Legatum data file not present")
need26 = pytest.mark.skipif(DATA26 is None, reason="2026 Legatum data file not present")

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _load():
    idx = pd.read_excel(DATA23, sheet_name="Prosperity Index")
    pil = pd.read_excel(DATA23, sheet_name="Pillars x 12")
    W = pil.pivot(index="area_code", columns="pillar_name", values="score_2023")[PILLARS]
    meta = idx.set_index("area_code")[["score_2023", "rank_2023"]]
    df = meta.join(W)
    return df[PILLARS].to_numpy(float), df["score_2023"].to_numpy(float), df["rank_2023"].to_numpy(int)

def _ranks(x):
    return rankdata(-x, method="min", axis=0).astype(int)

@need23
def test_sha256_2023():
    assert _sha(DATA23) == SHA23

@need26
def test_sha256_2026():
    assert _sha(DATA26) == SHA26

@need23
def test_baseline_replication_exact():
    S, score, rank = _load()
    assert S.shape == (167, 12) and not np.isnan(S).any()
    Y = S.mean(1)
    assert np.abs(Y - score).max() < 1e-9
    assert (_ranks(Y) == rank).all()

@need23
def test_leverage_identity_and_effective_weight_constraint():
    S, _, _ = _load()
    Y = S.mean(1)
    rho = np.array([np.corrcoef(S[:, j], Y)[0, 1] for j in range(12)])
    c = S.std(0, ddof=1) / Y.std(ddof=1)
    L_direct = np.var(S - Y[:, None], axis=0, ddof=1) / Y.var(ddof=1)
    assert np.allclose((c - rho) ** 2 + (1 - rho ** 2), L_direct, atol=1e-12)
    assert abs((rho * c).mean() - 1) < 1e-12

@need23
def test_three_domains_are_flat_equal_weighting():
    S, _, rank = _load()
    dom = np.column_stack([S[:, 0:4].mean(1), S[:, 4:8].mean(1), S[:, 8:12].mean(1)])
    assert np.abs(dom.mean(1) - S.mean(1)).max() < 1e-12
    assert (_ranks(dom.mean(1)) == rank).all()

@need23
def test_committed_pillar_statistics_match():
    S, _, _ = _load()
    Y = S.mean(1)
    lev = pd.read_csv(os.path.join(OUT, "leverage.csv")).set_index("pillar").loc[PILLARS]
    assert np.allclose(lev["sigma"], S.std(0, ddof=1), atol=1e-9)
    rho2 = np.array([np.corrcoef(S[:, j], Y)[0, 1] ** 2 for j in range(12)])
    assert np.allclose(lev["rho2"], rho2, atol=1e-9)

@need23
def test_uniform_prior_headline_regenerates_exactly():
    """analysis.py draws the near-equal, grid and uniform priors in that order from one generator."""
    S, _, rank = _load()
    rng = np.random.default_rng(42)
    rng.dirichlet(np.ones(12) * 50.0, 10000)
    rng.choice([0.5, 1.0, 1.5, 2.0], size=(10000, 12))
    W = rng.dirichlet(np.ones(12), 10000)
    avg_shift = np.abs(_ranks(S @ W.T) - rank[:, None]).mean()
    reported = json.load(open(os.path.join(OUT, "results.json")))["ua"]["uniform"]["avg_shift"]
    assert abs(avg_shift - reported) < 1e-9
