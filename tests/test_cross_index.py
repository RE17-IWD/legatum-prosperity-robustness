"""
Tests for the cross-index comparison of Section 8.3 (src/cross_index.py).

  pytest -q tests/test_cross_index.py

They check that the four external files are the ones analysed (SHA-256), that each index's
published construction is reproduced from its dimension scores before any reweighting, that
the Legatum row reproduces the first-principal-component row of Table 5, that a fresh computation
equals the committed outputs/cross_index.json in every field, and that the values printed in
Table 7 and Appendix Table A4 are the ones the code produces. Tests are skipped if a file is absent.
"""
import json, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
SHA = {
    "HDI": "61ed82e5b66c88dfca8ff9fac775c63981ecab6a254862af97acacc41c143117",
    "GII": "f7ab52221b82d15645eacdaf6f0ca0c1ce7b3686f814eb15f9bed8c7d9d88126",
    "WGI": "8e4754bda8b2be0b36d172c18a4caa634cb36fddd54145d16ab7316d22b3542c",
    "SDG": "ac34ef41945cf3cb69d420841e66eef6bf4d43dc5a76186e0c5a2ed658c7b892",
    "LPI": "8c789bd5ab881c12005f83e9b0c55ae81a68548dd18d3acb9fff9bec0eef631c",
}
try:
    import cross_index as ci
except FileNotFoundError:
    ci = None
need = pytest.mark.skipif(ci is None, reason="cross-index data files not present (see data/README.md)")
LOADERS = ("load_hdi", "load_wgi", "load_gii", "load_lpi", "load_sdg")


@need
@pytest.mark.parametrize("key", list(SHA))
def test_sha256(key):
    assert ci.sha(ci.FILES[key]) == SHA[key]


@need
def test_hdi_rebuilt_from_goalposts():
    ix = ci.load_hdi()
    assert ix["X"].shape == (193, 3)
    # the published HDI has three decimals; the rebuilt geometric mean must round to it for every country
    assert (np.round(ix["official_rebuilt"], 3) == np.round(ix["official"], 3)).all()


@need
def test_gii_hierarchy_reproduces_published_score():
    ix = ci.load_gii()
    assert ix["X"].shape == (133, 7)
    assert np.abs(ix["official"] - ix["official_rebuilt"]).max() < 1e-9


@need
def test_sdg_index_is_equal_weight_mean():
    ix = ci.load_sdg()
    assert ix["X"].shape == (114, 17) and ix["n_ranked"] == 169
    assert np.abs(ix["official"] - ix["X"].mean(1)).max() < 1e-9


@need
def test_legatum_matches_table5_pc1_row():
    r = ci.analyse(ci.load_lpi())
    t5 = json.load(open(os.path.join(ROOT, "outputs", "results.json")))["framings"]["First principal component"]
    assert abs(r["shift_E_A"]["mean"] - t5["mean"]) < 1e-12 and r["shift_E_A"]["max"] == t5["max"]
    assert r["check_A_equals_pc1_score"] > 1 - 1e-12


def _cmp(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _cmp(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for x, y in zip(a, b):
            _cmp(x, y, path)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert abs(a - b) < 1e-10, path
    else:
        assert a == b, path


@need
def test_committed_outputs_regenerate():
    committed = json.load(open(os.path.join(ROOT, "outputs", "cross_index.json")))["indices"]
    for name in LOADERS:
        r = ci.analyse(getattr(ci, name)())
        _cmp(json.loads(json.dumps(r)), committed[r["short"]], r["short"])


# values as printed in Table 7 and Appendix Table A4 of the manuscript
PRINTED = {
    #           n   k  PC1%   SD_A  CV_A  E-A   max  per100 negs  O-E    O-A    SD_B  E-B   R_EA*1e3 R_EB*1e3
    "HDI":     (193, 3, "89.4", "7.61", "0.23", "2.39", 11, "1.24", 0, "0.94", "3.10", "0.45", "0.13", "2.48", "0.01"),
    "WGI":     (206, 6, "85.1", "1.79", "0.11", "0.98", 6, "0.48", 0, None, None, "0.97", "0.73", "0.26", "0.19"),
    "GII":     (133, 7, "81.3", "2.02", "0.14", "1.38", 8, "1.04", 0, "3.40", "2.86", "0.69", "0.66", "1.38", "0.18"),
    "Legatum": (167, 12, "74.5", "1.85", "0.22", "1.81", 10, "1.08", 0, "0.00", "1.81", "0.77", "1.09", "1.42", "0.59"),
    "SDG":     (114, 17, "56.4", "6.85", "1.17", "5.67", 41, "4.97", 3, "0.00", "5.67", "6.36", "5.77", "199.45", "218.02"),
}


@need
def test_printed_table_values():
    out = json.load(open(os.path.join(ROOT, "outputs", "cross_index.json")))["indices"]
    for k, (n, kk, pc1, sda, cva, ea, mx, p100, neg, oe, oa, sdb, eb, rea, reb) in PRINTED.items():
        r = out[k]
        assert (r["n"], r["k"], r["n_negative"], int(r["shift_E_A"]["max"])) == (n, kk, neg, mx), k
        assert f"{100*r['pc1_share']:.1f}" == pc1, k
        assert f"{r['dispersion_A']['sd_pp']:.2f}" == sda and f"{r['dispersion_A']['cv']:.2f}" == cva, k
        assert f"{r['shift_E_A']['mean']:.2f}" == ea and f"{r['shift_E_A']['per100']:.2f}" == p100, k
        assert f"{r['dispersion_B']['sd_pp']:.2f}" == sdb and f"{r['shift_E_B']['mean']:.2f}" == eb, k
        assert f"{1000*r['rank_changing_share_A']:.2f}" == rea and f"{1000*r['rank_changing_share_B']:.2f}" == reb, k
        assert r["retained_pa"] == 1, k
        if oe is not None:
            assert f"{r['official_vs_E']['mean']:.2f}" == oe and f"{r['official_vs_A']['mean']:.2f}" == oa, k
