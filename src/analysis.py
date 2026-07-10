"""
Robustness and sensitivity analysis of the Legatum Prosperity Index 2023.
Reproducible: fixed seed. Produces all numbers/figures used in the paper.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, gmean
from scipy.stats.qmc import Sobol as QMCSobol
from SALib.sample import sobol as salib_sample
from SALib.analyze import sobol as salib_analyze
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os

SEED = 42
rng = np.random.default_rng(SEED)
DATA = os.environ.get("LPI_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "Dataset_Legatum_Prosperity_Index_2023.xlsx"))
OUT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(OUT, "figures")
YEAR = 2023
PILLARS = ['Safety and Security','Personal Freedom','Governance','Social Capital',
           'Investment Environment','Enterprise Conditions','Infrastructure and Market Access',
           'Economic Quality','Living Conditions','Health','Education','Natural Environment']
K = len(PILLARS)

plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,
                     "grid.alpha":0.3,"figure.dpi":150})

# ---------------------------------------------------------------- load + baseline
idx = pd.read_excel(DATA, sheet_name="Prosperity Index")
pil = pd.read_excel(DATA, sheet_name="Pillars x 12")
W = pil.pivot(index="area_code", columns="pillar_name", values=f"score_{YEAR}")[PILLARS]
meta = idx.set_index("area_code")[["area_name","area_group",f"score_{YEAR}",f"rank_{YEAR}"]]
meta.columns = ["area_name","region","off_score","off_rank"]
df = meta.join(W)
S = df[PILLARS].to_numpy()                     # 167 x 12 pillar scores
names = df["area_name"].to_numpy()
codes = df.index.to_numpy()
off_rank = df["off_rank"].to_numpy()
N = S.shape[0]

def scores_to_ranks(x):
    # rank 1 = best (highest score). ties -> min rank
    order = pd.Series(x).rank(ascending=False, method="min")
    return order.to_numpy().astype(int)

base_score = S.mean(axis=1)
base_rank = scores_to_ranks(base_score)
assert (base_rank == off_rank).all(), "baseline replication failed"
print("[baseline] equal-weight mean reproduces official rank exactly:", bool((base_rank==off_rank).all()))
print("[baseline] max |score diff| vs official: %.2e" % np.abs(base_score-df['off_score'].to_numpy()).max())

# ---------------------------------------------------------------- normalization variants
# Each maps the 12 pillar scores (already 0-100) onto an alternative scale, per country-pillar,
# computed column-wise across the 167 countries.
def norm_identity(M):          # Legatum native distance-to-frontier (already applied)
    return M.copy()
def norm_minmax(M):
    mn, mx = M.min(0), M.max(0)
    return 100*(M-mn)/(mx-mn)
def norm_zscore(M):
    return (M-M.mean(0))/M.std(0, ddof=0)
def norm_rank(M):
    # percentile rank within each pillar, 0-100
    out = np.empty_like(M, dtype=float)
    for j in range(M.shape[1]):
        out[:,j] = pd.Series(M[:,j]).rank(pct=True).to_numpy()*100
    return out
NORMS = {"identity":norm_identity,"minmax":norm_minmax,"zscore":norm_zscore,"rank":norm_rank}

def aggregate(Mn, w, mode):
    w = w/ w.sum()
    if mode == "additive":
        return Mn @ w
    # geometric: requires positivity -> shift each column to be strictly positive
    Mp = Mn.copy()
    cmin = Mp.min(0)
    shift = np.where(cmin <= 0, -cmin + 1e-6, 0.0)
    Mp = Mp + shift
    logM = np.log(Mp)
    return np.exp(logM @ w)   # weighted geometric mean

# ================================================================ PART 1: UA over weights
def sample_weights(n, scheme):
    if scheme == "dirichlet_flat":           # uniform on simplex (max dispersion)
        return rng.dirichlet(np.ones(K), size=n)
    if scheme == "dirichlet_conc":           # concentrated near equal weights
        return rng.dirichlet(np.ones(K)*50.0, size=n)
    if scheme == "legatum_discrete":         # weights in {0.5,1,1.5,2} then renormalised
        draws = rng.choice([0.5,1.0,1.5,2.0], size=(n,K))
        return draws/ draws.sum(1, keepdims=True)
    raise ValueError(scheme)

M_UA = 10000
ua = {}
for scheme in ["dirichlet_flat","dirichlet_conc","legatum_discrete"]:
    Wts = sample_weights(M_UA, scheme)
    sc = S @ Wts.T                            # 167 x M  (additive, identity norm)
    # ranks per simulation
    ranks = np.empty_like(sc, dtype=int)
    for m in range(sc.shape[1]):
        ranks[:,m] = scores_to_ranks(sc[:,m])
    rank_med = np.median(ranks,1)
    rank_lo, rank_hi = np.percentile(ranks,5,1), np.percentile(ranks,95,1)
    rank_iqr = np.percentile(ranks,75,1)-np.percentile(ranks,25,1)
    # average shift in rank (Saisana et al. 2005): mean abs deviation from baseline across all
    avg_shift = np.mean(np.abs(ranks - base_rank[:,None]))
    # spearman / kendall distribution vs baseline
    rhos = np.array([spearmanr(sc[:,m], base_score).correlation for m in range(0,sc.shape[1],50)])
    taus = np.array([kendalltau(ranks[:,m], base_rank).correlation for m in range(0,sc.shape[1],50)])
    ua[scheme] = dict(ranks=ranks, rank_med=rank_med, rank_lo=rank_lo, rank_hi=rank_hi,
                      rank_iqr=rank_iqr, avg_shift=float(avg_shift),
                      rho_med=float(np.median(rhos)), rho_p5=float(np.percentile(rhos,5)),
                      tau_med=float(np.median(taus)), tau_p5=float(np.percentile(taus,5)))
    print(f"[UA:{scheme}] avg rank shift={avg_shift:.2f}  rho_med={np.median(rhos):.4f}  tau_med={np.median(taus):.4f}")

# fragility: published rank outside own 90% interval
prim = ua["dirichlet_flat"]
fragile = (off_rank < prim["rank_lo"]) | (off_rank > prim["rank_hi"])
df_out = df.reset_index()[["area_code","area_name","region","off_score","off_rank"]].copy()
df_out["rank_median"]=prim["rank_med"]; df_out["rank_p5"]=prim["rank_lo"]
df_out["rank_p95"]=prim["rank_hi"]; df_out["rank_iqr"]=prim["rank_iqr"]
df_out["rank_range_90"]=prim["rank_hi"]-prim["rank_lo"]; df_out["fragile"]=fragile
df_out = df_out.sort_values("off_rank")
df_out.to_csv(f"{OUT}/outputs/rank_uncertainty.csv", index=False)
print(f"[UA] fragile countries (off rank outside 90% band): {int(fragile.sum())}/{N}")
print(f"[UA] median 90% rank-range: {np.median(prim['rank_hi']-prim['rank_lo']):.1f} places")

# ================================================================ PART 2: Sobol' SA
# Uncertain inputs (triggers):
#   x0..x11 : 12 pillar weights (continuous, mapped via -log to exponential -> Dirichlet-like)
#   x12     : normalization method (categorical: identity/minmax/zscore/rank)
#   x13     : aggregation mode (categorical: additive/geometric)
problem = {
    "num_vars": K+2,
    "names": [f"w_{p}" for p in PILLARS] + ["normalization","aggregation"],
    "bounds": [[0.0,1.0]]*K + [[0.0,1.0],[0.0,1.0]],
}
def map_weights(u):
    # turn 12 uniforms into a weight vector on the simplex (Dirichlet(1) via -log)
    g = -np.log(np.clip(u,1e-12,1.0))
    return g/g.sum()
def pick(u, options):
    i = min(int(u*len(options)), len(options)-1)
    return options[i]
NORM_OPTS=["identity","minmax","zscore","rank"]; AGG_OPTS=["additive","geometric"]

def model_eval_ranks(X):
    """Return rank matrix (N x n_samples) for SALib design X."""
    n = X.shape[0]
    out = np.empty((N,n))
    cache = {nm:NORMS[nm](S) for nm in NORM_OPTS}
    for i in range(n):
        w = map_weights(X[i,:K])
        nm = pick(X[i,K], NORM_OPTS); ag = pick(X[i,K+1], AGG_OPTS)
        sc = aggregate(cache[nm], w, ag)
        out[:,i] = scores_to_ranks(sc)
    return out

Nsob = 1024
Xs = salib_sample.sample(problem, Nsob, calc_second_order=False, seed=SEED)
R = model_eval_ranks(Xs)              # N x n_design
print(f"[SA] design rows: {Xs.shape[0]} (= N*(k+2))  model evals done")

# analyze per-country rank variance, then aggregate Si/STi across countries (mean weighted by variance)
Si_all = np.zeros((N,K+2)); STi_all = np.zeros((N,K+2)); var_all=np.zeros(N)
for c in range(N):
    y = R[c,:].astype(float)
    if y.std()==0:
        continue
    res = salib_analyze.analyze(problem, y, calc_second_order=False, seed=SEED, print_to_console=False)
    Si_all[c]=np.clip(res["S1"],0,None); STi_all[c]=np.clip(res["ST"],0,None); var_all[c]=y.var()
wts = var_all/var_all.sum()
Si = (Si_all*wts[:,None]).sum(0); STi=(STi_all*wts[:,None]).sum(0)
sa_df = pd.DataFrame({"input":problem["names"],"S1":Si,"ST":STi}).sort_values("ST",ascending=False)
sa_df.to_csv(f"{OUT}/outputs/sobol_indices.csv", index=False)
print("[SA] variance-weighted Sobol' (top):")
print(sa_df.head(14).to_string(index=False))

# ================================================================ PART 3: weights != importance
# main effect (first-order correlation ratio) of each pillar on the baseline score,
# contrasted with nominal equal weight (1/12). Pearson corr of pillar vs overall as proxy importance.
imp = np.array([np.corrcoef(S[:,j], base_score)[0,1]**2 for j in range(K)])  # R^2 share style
imp_share = imp/imp.sum()
wi = pd.DataFrame({"pillar":PILLARS,"nominal_weight":1/K,"importance_share":imp_share}
                  ).sort_values("importance_share",ascending=False)
wi.to_csv(f"{OUT}/outputs/weights_vs_importance.csv", index=False)
print("[importance] nominal vs realized importance (top 5):")
print(wi.head(5).to_string(index=False))

summary = dict(seed=SEED, n_countries=int(N), n_pillars=K, M_UA=M_UA, N_sobol=Nsob,
               baseline_exact=bool((base_rank==off_rank).all()),
               ua={k:{kk:vv for kk,vv in v.items() if kk in
                      ['avg_shift','rho_med','rho_p5','tau_med','tau_p5']} for k,v in ua.items()},
               fragile_count=int(fragile.sum()),
               median_90_range=float(np.median(prim['rank_hi']-prim['rank_lo'])),
               sobol=sa_df.to_dict("records"),
               importance=wi.to_dict("records"))
json.dump(summary, open(f"{OUT}/outputs/summary.json","w"), indent=2)
print("\n[done] wrote outputs/ and will plot next")
np.save(f"{OUT}/outputs/_ranks_flat.npy", prim["ranks"])
df_out.to_pickle(f"{OUT}/outputs/_df_out.pkl")
