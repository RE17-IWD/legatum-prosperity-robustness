"""
Cross-index comparison for the second JHSS revision (Section 8.3, Table 7, Figure 9, Appendix Table A4).

The same procedure is applied to five composite indices:
  Legatum Prosperity Index 2023 (12 pillars), Human Development Index 2023 (3 dimension indices),
  Global Innovation Index 2024 (7 pillars), Worldwide Governance Indicators 2025 (6 dimensions),
  SDG Index 2026 (17 goals).

For each index, on the complete-case matrix X (n x k) of its published dimension scores:
  1. PCA of the correlation matrix (population z-scores, as in Section 4): PC1 and PC2 shares,
     components retained by parallel analysis (95th percentile, 1,000 standard-normal matrices),
     PC1 loadings and their signs.
  2. E   = equal-weight arithmetic mean of X (the controlled baseline).
     O   = the official index where it differs from E (HDI geometric mean, GII input/output hierarchy).
  3. PCA weights, two definitions:
     A (primary; the "First principal component" row of Table 5): the PC1 score of the z-scored
       dimensions written as weights on the original scores, w_j proportional to v_j / sigma_j;
       the composite X @ w_A orders countries exactly as the PC1 score does.
     B (secondary): normalized PC1 loadings applied to the original scores, w_j proportional to loading_j.
  4. Mean absolute rank shift (eq6 with M = 1, average ranks for ties, also for the 3-decimal published HDI), maximum shift, shift per 100
     countries and Spearman correlation for O vs E, E vs A and E vs B.
  5. Dispersion of each weight vector: population SD (percentage points) and CV = k * SD,
     i.e. the SD relative to the equal weight 1/k.
  6. The rank-changing share of the reweighting implied by Section 6: the variance of X @ (w - 1/k)
     left after regressing it on E, divided by the variance of E.
Writes outputs/cross_index.csv and outputs/cross_index.json. Seed 42.
"""
import os, json, hashlib
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)


def _data(name, env):
    for p in (os.environ.get(env, ""), os.path.join(ROOT, "data", name), os.path.join(ROOT, name)):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found: set {env} or place it in data/ (see data/README.md)")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


FILES = dict(
    LPI=_data("Dataset_Legatum_Prosperity_Index_2023.xlsx", "LPI_DATA_2023"),
    HDI=_data("HDR25_Composite_indices_complete_time_series.csv", "HDI_DATA"),
    GII=_data("wipo-pub-2000-2024-tech1.xlsx", "GII_DATA"),
    WGI=_data("WGI_2026_Governance_Estimates_and_Scores_1996-2025.xlsx", "WGI_DATA"),
    SDG=_data("database_2026.xlsx", "SDG_DATA"),
)


# ------------------------------------------------------------------ loaders
def load_lpi():
    pil = pd.read_excel(FILES["LPI"], sheet_name="Pillars x 12")
    idx = pd.read_excel(FILES["LPI"], sheet_name="Prosperity Index").set_index("area_code")
    X = pil.pivot(index="area_code", columns="pillar_name", values="score_2023").dropna()
    names = idx.loc[X.index, "area_name"].to_numpy()
    off = idx.loc[X.index, "score_2023"].to_numpy(float)
    return dict(name="Legatum Prosperity Index", short="Legatum", edition="2023", X=X.to_numpy(float),
                dims=list(X.columns), units=names, official=off, official_rule="Equal-weight arithmetic mean of 12 pillars")


def load_hdi():
    h = pd.read_csv(FILES["HDI"], encoding="latin1")
    h = h[~h["iso3"].astype(str).str.startswith("ZZ")]                     # drop regional and group aggregates
    h = h.dropna(subset=["hdi_2023", "le_2023", "eys_2023", "mys_2023", "gnipc_2023"])
    lei = ((h["le_2023"] - 20) / (85 - 20)).clip(upper=1.0)               # UNDP goalposts; values above 85 years are capped
    ei = (np.minimum(h["eys_2023"], 18) / 18 + np.minimum(h["mys_2023"], 15) / 15) / 2
    gni = h["gnipc_2023"].clip(100, 75000)
    ii = (np.log(gni) - np.log(100)) / (np.log(75000) - np.log(100))
    X = np.column_stack([lei, ei, ii]).astype(float)
    geo = np.cbrt(X.prod(1))
    return dict(name="Human Development Index", short="HDI", edition="2023 (HDR 2025)", X=X,
                dims=["Health (life expectancy index)", "Education index", "Income index"],
                units=h["country"].to_numpy(), official=h["hdi_2023"].to_numpy(float), official_rebuilt=geo,
                official_rule="Geometric mean of 3 dimension indices")


def load_gii():
    st = pd.read_excel(FILES["GII"], sheet_name="Index Structure")
    pillars = st.loc[st["LEVEL"] == "Pillar", ["NUM", "NAME"]]
    d = pd.read_excel(FILES["GII"], sheet_name="Data")
    W = d[d["NUM"].isin(pillars["NUM"])].pivot(index="ISO3", columns="NUM", values="SCORE")[list(pillars["NUM"])].dropna()
    tot = d[d["NUM"].isna() & (d["NAME"] == "Global Innovation Index")].set_index("ISO3")["SCORE"]
    nm = d.drop_duplicates("ISO3").set_index("ISO3")["ECONOMY_NAME"]
    X = W.to_numpy(float)
    inp = [c.startswith("IN") for c in W.columns]
    rebuilt = 0.5 * X[:, inp].mean(1) + 0.5 * X[:, [not b for b in inp]].mean(1)
    return dict(name="Global Innovation Index", short="GII", edition="2024", X=X, dims=list(pillars["NAME"]),
                units=nm.loc[W.index].to_numpy(), official=tot.loc[W.index].to_numpy(float), official_rebuilt=rebuilt,
                official_rule="Mean of input sub-index (5 pillars) and output sub-index (2 pillars)")


def load_wgi(year=2025):
    cols = []
    for dim in ["va", "pv", "ge", "rq", "rl", "cc"]:
        s = pd.read_excel(FILES["WGI"], sheet_name=dim)
        s = s[s["Year"] == year].set_index("Economy (code)")
        cols.append(s["Governance score (0-100)"].rename(dim))
        nm = s["Economy (name)"]
    W = pd.concat(cols, axis=1).dropna()
    return dict(name="Worldwide Governance Indicators", short="WGI", edition=f"{year} (2026 release)",
                X=W.to_numpy(float), units=nm.reindex(W.index).to_numpy(),
                dims=["Voice and Accountability", "Political Stability", "Government Effectiveness",
                      "Regulatory Quality", "Rule of Law", "Control of Corruption"],
                official=None, official_rule="No official composite")


def load_sdg():
    d = pd.read_excel(FILES["SDG"], sheet_name="SDR2026 Data")
    gc = [f"Goal {i} Score" for i in range(1, 18)]
    d = d.dropna(subset=["2026 SDG Index Score", "2026 SDG Index Rank"])    # the 169 officially ranked countries
    n_ranked = len(d)
    d = d.dropna(subset=gc)
    return dict(name="SDG Index", short="SDG", edition="2026", X=d[gc].to_numpy(float), dims=[f"SDG {i}" for i in range(1, 18)],
                units=d["country"].to_numpy(), official=d["2026 SDG Index Score"].to_numpy(float),
                official_rule="Equal-weight arithmetic mean of 17 goals", n_ranked=int(n_ranked))


# ------------------------------------------------------------------ analysis
def ranks(s):
    return rankdata(-np.asarray(s), method="average")


def shift(a, b):
    ra, rb = ranks(a), ranks(b); d = np.abs(ra - rb); n = len(ra)
    return dict(mean=float(d.mean()), max=float(d.max()), per100=float(100 * d.mean() / n),
                spearman=float(spearmanr(ra, rb).correlation))


def parallel_analysis(n, k, reps=1000, q=95):
    rng = np.random.default_rng(SEED)
    ev = np.empty((reps, k))
    for r in range(reps):
        Z = rng.standard_normal((n, k))
        ev[r] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    return np.percentile(ev, q, axis=0)


def disp(w):
    k = len(w); sd = float(np.std(w))
    return dict(min=float(w.min()), max=float(w.max()), sd_pp=100 * sd, cv=k * sd)


def rank_changing_share(X, w, E):
    k = X.shape[1]
    dY = X @ (w - 1.0 / k)
    beta = np.cov(dY, E, bias=True)[0, 1] / np.var(E)
    resid = dY - dY.mean() - beta * (E - E.mean())
    return float(np.var(resid) / np.var(E))


def analyse(ix):
    X = ix["X"]; n, k = X.shape
    sig = X.std(0)
    Z = (X - X.mean(0)) / sig
    Rm = np.corrcoef(Z.T)
    ev, V = np.linalg.eigh(Rm); o = np.argsort(ev)[::-1]; ev, V = ev[o], V[:, o]
    E = X.mean(1)
    v = V[:, 0] * np.sign(np.corrcoef(Z @ V[:, 0], E)[0, 1])           # orient PC1 with the equal-weight mean
    load = v * np.sqrt(ev[0])
    pa = parallel_analysis(n, k)
    wA = (v / sig) / (v / sig).sum()
    wB = load / load.sum()
    YA, YB = X @ wA, X @ wB
    pc1_score = Z @ v
    res = dict(index=ix["name"], short=ix["short"], edition=ix["edition"], n=int(n), k=int(k),
               official_rule=ix["official_rule"], equal_weight=1.0 / k,
               pc1_share=float(ev[0] / k), pc2_share=float(ev[1] / k), retained_pa=int((ev > pa).cumprod().sum()),
               retained_kaiser=int((ev > 1).sum()),
               loadings={d: float(x) for d, x in zip(ix["dims"], load)},
               loading_min=float(load.min()), loading_max=float(load.max()), n_negative=int((load < 0).sum()),
               weights_A={d: float(x) for d, x in zip(ix["dims"], wA)},
               weights_B={d: float(x) for d, x in zip(ix["dims"], wB)},
               dispersion_A=disp(wA), dispersion_B=disp(wB),
               shift_E_A=shift(E, YA), shift_E_B=shift(E, YB),
               check_A_equals_pc1_score=float(spearmanr(YA, pc1_score).correlation),
               rank_changing_share_A=rank_changing_share(X, wA, E), rank_changing_share_B=rank_changing_share(X, wB, E),
               sigma_ratio=float(sig.max() / sig.min()))
    if ix.get("official") is not None:
        res["official_vs_E"] = shift(ix["official"], E)
        res["official_vs_A"] = shift(ix["official"], YA)
        res["max_abs_official_minus_E"] = float(np.abs(ix["official"] - E).max())
    if ix.get("official_rebuilt") is not None:
        res["max_abs_official_minus_rebuilt"] = float(np.abs(ix["official"] - ix["official_rebuilt"]).max())
        res["rebuilt_rank_spearman"] = float(spearmanr(ranks(ix["official"]), ranks(ix["official_rebuilt"])).correlation)
    if "n_ranked" in ix:
        res["n_ranked"] = ix["n_ranked"]
    top = np.argsort(-np.abs(ranks(E) - ranks(YA)), kind="stable")[:5]
    res["largest_shifts_A"] = [dict(unit=str(ix["units"][i]), rank_E=float(ranks(E)[i]), rank_A=float(ranks(YA)[i])) for i in top]
    return res


if __name__ == "__main__":
    out = dict(seed=SEED, files={k: dict(file=os.path.basename(p), sha256=sha(p)) for k, p in FILES.items()}, indices={})
    rows = []
    for loader in (load_hdi, load_wgi, load_gii, load_lpi, load_sdg):
        ix = loader(); r = analyse(ix); out["indices"][r["short"]] = r
        rows.append(dict(index=r["index"], edition=r["edition"], n=r["n"], k=r["k"], pc1_pct=100 * r["pc1_share"],
                         pc2_pct=100 * r["pc2_share"], retained_pa=r["retained_pa"], loading_min=r["loading_min"],
                         loading_max=r["loading_max"], n_negative=r["n_negative"],
                         wA_min_pct=100 * r["dispersion_A"]["min"], wA_max_pct=100 * r["dispersion_A"]["max"],
                         wA_sd_pp=r["dispersion_A"]["sd_pp"], wA_cv=r["dispersion_A"]["cv"],
                         wB_sd_pp=r["dispersion_B"]["sd_pp"], wB_cv=r["dispersion_B"]["cv"],
                         shift_E_A=r["shift_E_A"]["mean"], shift_E_A_max=r["shift_E_A"]["max"], shift_E_A_per100=r["shift_E_A"]["per100"],
                         spearman_E_A=r["shift_E_A"]["spearman"], shift_E_B=r["shift_E_B"]["mean"], shift_E_B_max=r["shift_E_B"]["max"],
                         shift_O_E=r.get("official_vs_E", {}).get("mean", np.nan), shift_O_E_max=r.get("official_vs_E", {}).get("max", np.nan),
                         shift_O_A=r.get("official_vs_A", {}).get("mean", np.nan),
                         rcs_A=r["rank_changing_share_A"], rcs_B=r["rank_changing_share_B"]))
        print(f"[{r['short']}] n={r['n']} k={r['k']} PC1={100*r['pc1_share']:.2f}% PA={r['retained_pa']} neg={r['n_negative']} "
              f"A: sd={r['dispersion_A']['sd_pp']:.2f}pp cv={r['dispersion_A']['cv']:.3f} shift={r['shift_E_A']['mean']:.2f} (max {r['shift_E_A']['max']:.1f}) "
              f"B: sd={r['dispersion_B']['sd_pp']:.2f}pp shift={r['shift_E_B']['mean']:.2f} O-E={r.get('official_vs_E', {}).get('mean', float('nan')):.2f} "
              f"rcsA={r['rank_changing_share_A']:.5f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cross_index.csv"), index=False)
    json.dump(out, open(os.path.join(OUT, "cross_index.json"), "w"), indent=1)
