"""
Full analysis for the JHSS revision of
"How Robust Are the Legatum Prosperity Index Rankings?"
Regenerates every number in the manuscript from the two data files. Seed 42.
"""
import os, json, hashlib, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr, rankdata, norm
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score
from SALib.sample import sobol as salib_sample
warnings.filterwarnings("ignore")

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
def _data(name, env):   # environment variable, then data/, then the repository root
    for p_ in (os.environ.get(env, ""), os.path.join(ROOT, "data", name), os.path.join(ROOT, name)):
        if p_ and os.path.exists(p_): return p_
    raise FileNotFoundError(f"{name} not found: set {env} or place it in data/ (see data/README.md)")
DATA23 = _data("Dataset_Legatum_Prosperity_Index_2023.xlsx", "LPI_DATA_2023")
DATA26 = _data("Legatum_Prosperity_Index_2026_Data_Sheet.xlsx", "LPI_DATA_2026")
PILLARS = ['Safety and Security','Personal Freedom','Governance','Social Capital',
           'Investment Environment','Enterprise Conditions','Infrastructure and Market Access',
           'Economic Quality','Living Conditions','Health','Education','Natural Environment']
DOMAINS = {"Inclusive Societies": PILLARS[0:4], "Open Economies": PILLARS[4:8], "Empowered People": PILLARS[8:12]}
K = 12
R = {}   # everything reported in the paper -> outputs/results.json

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
R["sha256_2023"] = sha(DATA23); R["sha256_2026"] = sha(DATA26)

# ------------------------------------------------------------------ 1. data + baseline
idx = pd.read_excel(DATA23, sheet_name="Prosperity Index")
pil = pd.read_excel(DATA23, sheet_name="Pillars x 12")
Wide = pil.pivot(index="area_code", columns="pillar_name", values="score_2023")[PILLARS]
meta = idx.set_index("area_code")[["area_name","area_group","score_2023","rank_2023"]]
meta.columns = ["name","region","off_score","off_rank"]
df = meta.join(Wide)
S = df[PILLARS].to_numpy(float); names = df["name"].to_numpy(); region = df["region"].to_numpy()
off_rank = df["off_rank"].to_numpy(int); off_score = df["off_score"].to_numpy(float)
N = len(df)

def ranks_of(scores):  # rank 1 = best, ties -> min; works on (N,) or (N,m)
    return rankdata(-scores, method="min", axis=0).astype(int)

Y0 = S.mean(1); base_rank = ranks_of(Y0)
R["baseline"] = dict(n=int(N), max_abs_score_dev=float(np.abs(Y0-off_score).max()),
                     ranks_exact=bool((base_rank==off_rank).all()),
                     spearman=float(spearmanr(base_rank, off_rank).correlation),
                     kendall=float(kendalltau(base_rank, off_rank).correlation))
print("[baseline]", R["baseline"])

# ------------------------------------------------------------------ 2. normalizations / aggregations
def n_identity(M): return M.copy()
def n_minmax(M): return 100*(M-M.min(0))/(M.max(0)-M.min(0))
def n_zscore(M): return (M-M.mean(0))/M.std(0)
def n_rank(M): return rankdata(M, axis=0)/M.shape[0]*100
NORMS = {"identity":n_identity,"minmax":n_minmax,"zscore":n_zscore,"rank":n_rank}
NORM_OPTS = list(NORMS); AGG_OPTS = ["additive","geometric"]
def positive(M):
    cmin = M.min(0); shift = np.where(cmin <= 0, -cmin + 1e-6, 0.0); return M + shift
def agg_scores(Mn, Wt, mode):            # Mn: N x 12 ; Wt: m x 12 rows summing to 1 -> N x m
    if mode == "additive":
        return Mn @ Wt.T
    return np.exp(np.log(positive(Mn)) @ Wt.T)

# ------------------------------------------------------------------ 3. weight-uncertainty (3 priors)
M_UA = 10000
def draw(prior, n):
    if prior == "uniform": return rng.dirichlet(np.ones(K), n)
    if prior == "near_equal": return rng.dirichlet(np.ones(K)*50.0, n)
    if prior == "grid":
        d = rng.choice([0.5,1.0,1.5,2.0], size=(n,K)); return d/d.sum(1, keepdims=True)
ua = {}; ranks_uniform = None
for prior in ["near_equal","grid","uniform"]:
    Wt = draw(prior, M_UA); rk = ranks_of(S @ Wt.T)          # N x M
    shift_per_draw = np.abs(rk - off_rank[:,None]).mean(0)
    sub = range(0, M_UA, 50)
    rhos = np.array([spearmanr(rk[:,m], off_rank).correlation for m in sub])
    taus = np.array([kendalltau(rk[:,m], off_rank).correlation for m in sub])
    ua[prior] = dict(avg_shift=float(shift_per_draw.mean()), shift_p5=float(np.percentile(shift_per_draw,5)),
                     shift_p95=float(np.percentile(shift_per_draw,95)),
                     rho_med=float(np.median(rhos)), rho_p5=float(np.percentile(rhos,5)),
                     tau_med=float(np.median(taus)), tau_p5=float(np.percentile(taus,5)))
    if prior == "uniform": ranks_uniform = rk
    print(f"[UA {prior}]", ua[prior])
R["ua"] = ua
lo, hi = np.percentile(ranks_uniform,5,1), np.percentile(ranks_uniform,95,1)
width = hi - lo
cty = pd.DataFrame(dict(area_code=df.index, country=names, region=region, off_score=off_score, off_rank=off_rank,
                        rank_median=np.median(ranks_uniform,1), rank_p5=lo, rank_p95=hi,
                        rank_iqr=np.percentile(ranks_uniform,75,1)-np.percentile(ranks_uniform,25,1), width_90=width))
cty["fragile"] = (off_rank < lo) | (off_rank > hi)
cty = cty.sort_values("off_rank"); cty.to_csv(f"{OUT}/rank_uncertainty.csv", index=False)
tiers = [(1,33),(34,67),(68,100),(101,134),(135,167)]
tier_tab = []
for a_, b_ in tiers:
    m_ = (off_rank>=a_)&(off_rank<=b_)
    tier_tab.append(dict(tier=f"{a_}–{b_}", mean=float(width[m_].mean()), median=float(np.median(width[m_])), max=int(width[m_].max())))
NAMED = ("Denmark","South Sudan","Turkmenistan","Belarus","China","Norway","Russia")
R["intervals"] = dict(median_width=float(np.median(width)), fragile=int(cty["fragile"].sum()), tiers=tier_tab,
                      widest=cty.sort_values("width_90",ascending=False).head(8)[["country","off_rank","rank_p5","rank_p95","width_90","region"]].to_dict("records"),
                      narrowest=cty.sort_values("width_90").head(4)[["country","off_rank","rank_p5","rank_p95","width_90","region"]].to_dict("records"),
                      named={n: dict(rank=int(off_rank[i]), lo=int(lo[i]), hi=int(hi[i])) for i,n in enumerate(names) if n in NAMED})
print("[intervals] median width", R["intervals"]["median_width"], "tiers", tier_tab)

# ------------------------------------------------------------------ 4. Sobol' design + estimators
problem = {"num_vars": K+2, "names": [f"w_{p}" for p in PILLARS]+["normalization","aggregation"], "bounds": [[0,1]]*(K+2)}
INPUTS = PILLARS + ["Normalization selector","Aggregation selector"]
def map_weights(U):  # m x 12 uniforms -> simplex (flat Dirichlet)
    g = -np.log(np.clip(U,1e-12,1)); return g/g.sum(1,keepdims=True)
def evaluate(X):
    """rank matrix N x rows for a SALib design X"""
    m = X.shape[0]; out = np.empty((N,m), dtype=np.int16)
    W = map_weights(X[:,:K]); nsel = np.minimum((X[:,K]*4).astype(int),3); asel = np.minimum((X[:,K+1]*2).astype(int),1)
    cache = {nm: NORMS[nm](S) for nm in NORM_OPTS}
    for ni,nm in enumerate(NORM_OPTS):
        for ai,ag in enumerate(AGG_OPTS):
            sel = np.where((nsel==ni)&(asel==ai))[0]
            for c0 in range(0, len(sel), 16384):
                cols = sel[c0:c0+16384]
                out[:,cols] = ranks_of(agg_scores(cache[nm], W[cols], ag))
    return out
def sobol_indices(Rk, k, Nb, idx=None):
    """Saltelli (2010) / Jansen estimators for every country at once. Rk: N x Nb*(k+2)."""
    if idx is None: idx = np.arange(Nb)
    blk = Rk.reshape(N, Nb, k+2).astype(float)[:, idx, :]
    A = blk[:,:,0]; B = blk[:,:,k+1]; AB = blk[:,:,1:k+1]
    allY = np.concatenate([A,B],1); V = allY.var(1); mu = allY.mean(1)
    Ac = A - mu[:,None]; Bc = B - mu[:,None]; ABc = AB - mu[:,None,None]
    S1 = (Bc[:,:,None]*(ABc - Ac[:,:,None])).mean(1) / V[:,None]
    ST = 0.5*((Ac[:,:,None]-ABc)**2).mean(1) / V[:,None]
    return S1, ST, V
def run_sobol(Nb, nboot=200):
    X = salib_sample.sample(problem, Nb, calc_second_order=False, seed=SEED)
    Rk = evaluate(X)
    S1, ST, V = sobol_indices(Rk, K+2, Nb)
    pi = V/V.sum()
    agg_S1 = (S1*pi[:,None]).sum(0); agg_ST = (ST*pi[:,None]).sum(0)
    brng = np.random.default_rng(SEED)
    bS1 = np.empty((nboot,K+2)); bST = np.empty((nboot,K+2)); cS1 = np.empty((nboot,N,K+2)); cST = np.empty((nboot,N,K+2))
    for b in range(nboot):
        ii = brng.integers(0, Nb, Nb)
        s1,st,v = sobol_indices(Rk, K+2, Nb, ii); p = v/v.sum()
        bS1[b] = (s1*p[:,None]).sum(0); bST[b] = (st*p[:,None]).sum(0); cS1[b]=s1; cST[b]=st
    hw = lambda arr: (np.percentile(arr,97.5,0)-np.percentile(arr,2.5,0))/2
    return dict(S1=agg_S1, ST=agg_ST, S1_hw=hw(bS1), ST_hw=hw(bST), S1_c=S1, ST_c=ST, V=V,
                S1_c_hw=hw(cS1), ST_c_hw=hw(cST), ST_c_boot=cST, n_eval=int(X.shape[0]))
sob_small = run_sobol(1024); sob = run_sobol(8192)
print("[sobol] evaluations", sob["n_eval"])
sa = pd.DataFrame(dict(input=INPUTS, S1=sob["S1"], S1_hw=sob["S1_hw"], ST=sob["ST"], ST_hw=sob["ST_hw"],
                       S1_N1024=sob_small["S1"], ST_N1024=sob_small["ST"], S1_hw_N1024=sob_small["S1_hw"], ST_hw_N1024=sob_small["ST_hw"]))
sa = sa.sort_values("ST", ascending=False); sa.to_csv(f"{OUT}/sobol_indices.csv", index=False)
print(sa[["input","S1","S1_hw","ST","ST_hw"]].to_string(index=False))
order_big = list(sa["input"]); order_small = list(sa.sort_values("ST_N1024",ascending=False)["input"])
R["sobol"] = dict(n_eval=sob["n_eval"], n_eval_small=sob_small["n_eval"], table=sa.to_dict("records"),
                  sum_S1=float(sob["S1"].sum()), sum_S1_small=float(sob_small["S1"].sum()),
                  max_change_vs_1024=float(np.abs(sob["ST"]-sob_small["ST"]).max()),
                  max_change_S1_vs_1024=float(np.abs(sob["S1"]-sob_small["S1"]).max()),
                  order_8192=order_big, order_1024=order_small,
                  hw_ratio_ST=float((sob_small["ST_hw"][:K]/sob["ST_hw"][:K]).mean()),
                  hw_ratio_S1=float((sob_small["S1_hw"][:K]/sob["S1_hw"][:K]).mean()))
ST_w = sob["ST"][:K]; S1_w = sob["S1"][:K]
top2 = np.sort(ST_w)[::-1]; ST_sorted_all = np.sort(sob["ST"])[::-1]
R["sobol"]["top2_sum"] = float(top2[:2].sum()); R["sobol"]["next3_sum"] = float(ST_sorted_all[2:5].sum()); R["sobol"]["next4_sum"] = float(ST_sorted_all[2:6].sum())
R["sobol"]["agg_exceeds_n_weights"] = int((ST_w < sob["ST"][K+1]).sum()); R["sobol"]["norm_exceeds_n_weights"] = int((ST_w < sob["ST"][K]).sum())
R["sobol"]["norm_interaction_frac"] = float(1 - sob["S1"][K]/sob["ST"][K]); R["sobol"]["agg_interaction_frac"] = float(1 - sob["S1"][K+1]/sob["ST"][K+1])
R["sobol"]["norm_small"] = dict(S1=float(sob_small["S1"][K]), hw=float(sob_small["S1_hw"][K]))
R["sobol"]["norm_big"] = dict(S1=float(sob["S1"][K]), hw=float(sob["S1_hw"][K]))

# ------------------------------------------------------------------ 5. realized importance: eta^2, rho^2, leave-one-out
def eta2(x, y, B):
    bins = pd.qcut(x, B, labels=False, duplicates="drop"); ybar = y.mean(); num = 0.0
    for b in np.unique(bins):
        m = bins==b; num += m.sum()*(y[m].mean()-ybar)**2
    return num/((y-ybar)**2).sum()
rho = np.array([pearsonr(S[:,j], Y0)[0] for j in range(K)]); rho2 = rho**2
eta = {B: np.array([eta2(S[:,j], Y0, B) for j in range(K)]) for B in (5,10,20)}
eta_loo = np.array([eta2(S[:,j], np.delete(S,j,1).mean(1), 10) for j in range(K)])
share = eta[10]/eta[10].sum()
imp = pd.DataFrame(dict(pillar=PILLARS, rho2=rho2, eta2_B5=eta[5], eta2_B10=eta[10], eta2_B20=eta[20], eta2_loo=eta_loo, share=share,
                        S1=S1_w, ST=ST_w)).sort_values("eta2_B10", ascending=False)
imp.to_csv(f"{OUT}/realized_importance.csv", index=False)
R["importance"] = dict(table=imp.to_dict("records"),
                       spearman_ST_eta=float(spearmanr(ST_w, eta[10]).correlation), spearman_S1_eta=float(spearmanr(S1_w, eta[10]).correlation),
                       spearman_ST_rho2=float(spearmanr(ST_w, rho2).correlation),
                       spearman_B5_B20=float(spearmanr(eta[5], eta[20]).correlation), max_bin_move=float(np.abs(eta[20]-eta[5]).max()),
                       spearman_eta_loo=float(spearmanr(eta[10], eta_loo).correlation),
                       loo_drop_min=float((eta[10]-eta_loo).min()), loo_drop_max=float((eta[10]-eta_loo).max()),
                       loo_drop_argmax=PILLARS[int(np.argmax(eta[10]-eta_loo))],
                       eta_below_rho2=[PILLARS[j] for j in range(K) if eta[10][j] < rho2[j]])
print("[importance] Spearman(ST, eta2)=%.3f  (S1)=%.3f" % (R["importance"]["spearman_ST_eta"], R["importance"]["spearman_S1_eta"]))

# ------------------------------------------------------------------ 6. leverage identity
sig = S.std(0, ddof=1); sigY = Y0.std(ddof=1); c = sig/sigY
L = (c-rho)**2 + (1-rho**2); scale_term = (c-rho)**2; info_term = 1-rho**2; L5 = (1-rho**2)/rho**2; a = rho*c
lev = pd.DataFrame(dict(pillar=PILLARS, sigma=sig, rho2=rho2, eta2=eta[10], pc1=np.nan, L=L, scale_term=scale_term, info_term=info_term, L_eq5=L5, a=a, ST=ST_w, ST_hw=sob["ST_hw"][:K]))
def named(p): return PILLARS.index(p)
pf, sc_ = named("Personal Freedom"), named("Social Capital")
R["leverage"] = dict(sigma_Y=float(sigY), sigma_min=float(sig.min()), sigma_min_pillar=PILLARS[int(sig.argmin())], sigma_max=float(sig.max()), sigma_max_pillar=PILLARS[int(sig.argmax())],
                     mean_a=float(a.mean()), sd_a=float(a.std(ddof=1)), identity_resid=float(np.abs(L - np.var(S-Y0[:,None],axis=0,ddof=1)/sigY**2).max()),
                     pearson_L_ST=float(pearsonr(L,ST_w)[0]), pearson_info_ST=float(pearsonr(info_term,ST_w)[0]), pearson_eq5_ST=float(pearsonr(L5,ST_w)[0]), pearson_eta_ST=float(pearsonr(eta[10],ST_w)[0]),
                     spearman_L_ST=float(spearmanr(L,ST_w).correlation), spearman_info_ST=float(spearmanr(info_term,ST_w).correlation), spearman_eq5_ST=float(spearmanr(L5,ST_w).correlation),
                     spearman_rho2_ST=float(spearmanr(rho2,ST_w).correlation), spearman_sigma_ST=float(spearmanr(sig,ST_w).correlation),
                     eq5_pct_error={PILLARS[j]: float(100*(L5[j]-L[j])/L[j]) for j in range(K)},
                     scale_share={PILLARS[j]: float(100*scale_term[j]/L[j]) for j in range(K)},
                     pair=dict(rho2_PF=float(rho2[pf]), rho2_SC=float(rho2[sc_]), rho2_ratio=float(rho2[pf]/rho2[sc_]), sigma_PF=float(sig[pf]), sigma_SC=float(sig[sc_]), sigma_ratio=float(sig[pf]/sig[sc_]),
                               L_PF=float(L[pf]), L_SC=float(L[sc_]), L_ratio=float(L[pf]/L[sc_]), ST_PF=float(ST_w[pf]), ST_SC=float(ST_w[sc_]), ST_ratio=float(ST_w[pf]/ST_w[sc_]),
                               eq5_PF=float(L5[pf]), eq5_SC=float(L5[sc_]), eq5_ratio=float(L5[pf]/L5[sc_]), eta_PF=float(eta[10][pf]), eta_SC=float(eta[10][sc_]),
                               scale_share_PF=float(100*scale_term[pf]/L[pf]), scale_share_SC=float(100*scale_term[sc_]/L[sc_])))
R["leverage"]["pair"]["pred_err_pct"] = float(100*abs(R["leverage"]["pair"]["L_ratio"]-R["leverage"]["pair"]["ST_ratio"])/R["leverage"]["pair"]["ST_ratio"])
print("[leverage] pair", R["leverage"]["pair"])
def eq5_study(sd_a, reps=2000):
    g = np.random.default_rng(1); out=[]
    for _ in range(reps):
        rr = g.uniform(0.5,0.98,K); aa = 1 + sd_a*g.standard_normal(K); aa = aa - aa.mean() + 1; cc = aa/rr
        out.append(spearmanr((cc-rr)**2+(1-rr**2), (1-rr**2)/rr**2).correlation)
    return float(np.mean(out))
R["leverage"]["eq5_study"] = {str(s): eq5_study(s) for s in (0.05,0.1,0.2,0.3,0.4,0.5)}

# ------------------------------------------------------------------ 7. latent structure: PCA, parallel analysis, VIF, network
C = np.corrcoef(S.T); evals, evecs = np.linalg.eigh(C); order = np.argsort(evals)[::-1]; evals = evals[order]; evecs = evecs[:,order]
loadings = evecs*np.sqrt(evals)
if loadings[:,0].sum() < 0: loadings[:,0] *= -1; evecs[:,0] *= -1
if loadings[named("Personal Freedom"),1] > 0: loadings[:,1] *= -1
vif = np.zeros(K)
for j in range(K):
    Xj = np.column_stack([np.ones(N), np.delete(S,j,1)]); beta = np.linalg.lstsq(Xj, S[:,j], rcond=None)[0]
    r2 = 1 - ((S[:,j]-Xj@beta)**2).sum()/((S[:,j]-S[:,j].mean())**2).sum(); vif[j] = 1/(1-r2)
lev["pc1"] = loadings[:,0]
prng = np.random.default_rng(SEED); pa = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(prng.standard_normal((N,K)).T)))[::-1] for _ in range(1000)])
pa95 = np.percentile(pa,95,0)
R["pca"] = dict(condition_number=float(evals[0]/evals[-1]), eigenvalues=evals.tolist(), var_explained=(evals/K).tolist(), pa95=pa95.tolist(),
                retained_pa=int((evals>pa95).sum()), retained_kaiser=int((evals>1).sum()),
                pc1_loadings={PILLARS[j]: float(loadings[j,0]) for j in range(K)}, pc2_loadings={PILLARS[j]: float(loadings[j,1]) for j in range(K)},
                vif={PILLARS[j]: float(vif[j]) for j in range(K)}, all_positive=bool((C>0).all()),
                spearman_pc1_eta=float(spearmanr(loadings[:,0], eta[10]).correlation), spearman_pc1_ST=float(spearmanr(loadings[:,0], ST_w).correlation))
print("[pca] cond", R["pca"]["condition_number"], "PC1 %.3f PC2 %.3f" % (evals[0]/K, evals[1]/K), "PA retains", R["pca"]["retained_pa"], "Kaiser", R["pca"]["retained_kaiser"])
P = np.linalg.inv(C); d = np.sqrt(np.diag(P)); pcor = -P/np.outer(d,d); np.fill_diagonal(pcor, 0)
iu = np.triu_indices(K,1); r = pcor[iu]; z = np.arctanh(r)*np.sqrt(N-(K-2)-3); p = 2*(1-norm.cdf(np.abs(z)))
o = np.argsort(p); m = len(p); bh = np.zeros(m,bool); thr = 0.05*np.arange(1,m+1)/m
ok = np.where(p[o] <= thr)[0]
if len(ok): bh[o[:ok.max()+1]] = True
pc_sel = np.zeros_like(pcor); pc_sel[iu] = np.where(bh, r, 0); pc_sel = pc_sel + pc_sel.T
strength = np.abs(pc_sel).sum(1); ei = pc_sel.sum(1)
net = pd.DataFrame(dict(pillar=PILLARS, strength=strength, expected_influence=ei, n_edges=(pc_sel!=0).sum(1),
                        strength_unthresholded=np.abs(pcor).sum(1))).sort_values("strength", ascending=False)
net.to_csv(f"{OUT}/network_centrality.csv", index=False)
edges = pd.DataFrame(dict(a=[PILLARS[i] for i in iu[0]], b=[PILLARS[j] for j in iu[1]], partial_r=r, p=p, kept=bh)); edges.to_csv(f"{OUT}/network_edges.csv", index=False)
R["network"] = dict(table=net.to_dict("records"), n_edges_kept=int(bh.sum()), n_edges=int(m), unregularised_min_abs=float(np.abs(r).min()),
                    spearman_strength_ST=float(spearmanr(strength, ST_w).correlation))
print("[network] edges kept", bh.sum(), "/", m); print(net.to_string(index=False))

# ------------------------------------------------------------------ 8. per-country analyses
STc = sob["ST_c"]; V = sob["V"]
shares = STc[:,:K]/STc[:,:K].sum(1,keepdims=True)              # 12 pillar-weight S_T normalized to sum to one per country (not fractions of rank variance)
resid = np.empty((N,K))
for j in range(K):
    Xj = np.column_stack([np.ones(N), np.delete(S,j,1)]); beta = np.linalg.lstsq(Xj, S[:,j], rcond=None)[0]
    e = S[:,j]-Xj@beta; resid[:,j] = np.abs(e/e.std(ddof=K))
unus = resid.mean(1)
per_pillar_corr = [float(spearmanr(resid[:,j], STc[:,j]).correlation) for j in range(K)]
lead = np.argmax(shares,1); lead_counts = {PILLARS[j]: int((lead==j).sum()) for j in range(K)}
hhi = (shares**2).sum(1); dist_med = np.abs(off_rank - np.median(off_rank))
nm = {n:i for i,n in enumerate(names)}
def top_of(country): i = nm[country]; return (PILLARS[int(lead[i])], float(shares[i].max()))
R["country"] = dict(most_unusual=[(str(names[i]), float(unus[i])) for i in np.argsort(unus)[::-1][:5]], most_ordinary=[(str(names[i]), float(unus[i])) for i in np.argsort(unus)[:3]],
                    spearman_unus_width=float(spearmanr(unus, width).correlation), per_pillar_corr={PILLARS[j]: per_pillar_corr[j] for j in range(K)},
                    per_pillar_median=float(np.median(per_pillar_corr)), n_positive=int(sum(x>0 for x in per_pillar_corr)),
                    lead_counts=lead_counts, china_top=top_of("China"), denmark_top=top_of("Denmark"), turkmenistan_top=top_of("Turkmenistan"),
                    belarus_top=top_of("Belarus"), south_sudan_top=top_of("South Sudan"),
                    hhi_min=float(hhi.min()), hhi_max=float(hhi.max()), spearman_hhi_width=float(spearmanr(hhi, width).correlation),
                    spearman_distmed_width=float(spearmanr(dist_med, width).correlation))
print("[country]", {k:v for k,v in R["country"].items() if k in ("spearman_unus_width","per_pillar_median","n_positive","spearman_hhi_width","spearman_distmed_width")})
pd.DataFrame(shares, columns=PILLARS, index=names).assign(unusualness=unus, hhi=hhi, width_90=width).to_csv(f"{OUT}/country_profiles.csv")
# clustering
prof = shares
Z = linkage(prof, "ward"); sil = {k: float(silhouette_score(prof, fcluster(Z,k,"maxclust"))) for k in range(2,8)}
kbest = max(sil, key=sil.get); lab = fcluster(Z, kbest, "maxclust")
crng = np.random.default_rng(SEED); aris=[]
for _ in range(100):
    ii = np.unique(crng.integers(0,N,N)); lb = fcluster(linkage(prof[ii],"ward"), kbest, "maxclust"); aris.append(adjusted_rand_score(lab[ii], lb))
clusters=[]
for cl in range(1,kbest+1):
    mm = lab==cl; mean_sh = prof[mm].mean(0); j = int(mean_sh.argmax())
    reg = pd.Series(region[mm]).value_counts().to_dict()
    tier = pd.Series(pd.cut(off_rank[mm],[0,33,67,100,134,167],labels=[t["tier"] for t in tier_tab])).value_counts().to_dict()
    clusters.append(dict(cluster=cl, n=int(mm.sum()), dominant=PILLARS[j], dominant_share=float(mean_sh[j]), regions={str(k):int(v) for k,v in reg.items()}, tiers={str(k):int(v) for k,v in tier.items()},
                         members=[str(x) for x in names[mm]]))
R["clusters"] = dict(silhouette=sil, k=int(kbest), stability_ari=float(np.mean(aris)), clusters=clusters)
print("[clusters] k", kbest, "sil", sil[kbest], "ARI", np.mean(aris)); [print("   ", c["n"], c["dominant"], round(c["dominant_share"],3)) for c in clusters]
pd.DataFrame(dict(country=names, cluster=lab)).to_csv(f"{OUT}/clusters.csv", index=False)

# ------------------------------------------------------------------ 9. structural reframings (2023)
def shift_stats(rk): return dict(mean=float(np.abs(rk-off_rank).mean()), max=int(np.abs(rk-off_rank).max()), argmax=str(names[int(np.abs(rk-off_rank).argmax())]), spearman=float(spearmanr(rk,off_rank).correlation))
dom = np.column_stack([S[:,[named(p) for p in ps]].mean(1) for ps in DOMAINS.values()])
fr = {}
fr["Three domains, equal weight"] = dict(**shift_stats(ranks_of(dom.mean(1))), max_dev=float(np.abs(dom.mean(1)-Y0).max()))
fr["Three domains, geometric mean across domains"] = shift_stats(ranks_of(np.exp(np.log(dom).mean(1))))
fr["Pillars, min–max normalization"] = shift_stats(ranks_of(n_minmax(S).mean(1)))
fr["Pillars, geometric mean"] = shift_stats(ranks_of(np.exp(np.log(S).mean(1))))
Zs = n_zscore(S); pc1 = Zs @ evecs[:,0]; pc1 = pc1*np.sign(np.corrcoef(pc1,Y0)[0,1])
fr["First principal component"] = shift_stats(ranks_of(pc1))
fr["Pillars, z-score normalization"] = shift_stats(ranks_of(Zs.mean(1)))
fr["Pillars, percentile-rank normalization"] = shift_stats(ranks_of(n_rank(S).mean(1)))
R["framings"] = fr; print("[framings]"); [print("   %-48s %.2f %2d %.4f %s" % (k,v["mean"],v["max"],v["spearman"],v["argmax"])) for k,v in fr.items()]

# ------------------------------------------------------------------ 10. 2026 edition
x26 = pd.ExcelFile(DATA26); ranks26 = x26.parse("ranks").set_index("Iso 3"); def_rank = ranks26["Prosperity Index"]
v26 = {}
for tab,label in [("ranks_uni_dtf+","Uniform normalization function"),("ranks_uni_p","Uniform aggregation exponent"),("ranks_uni_wghts","Uniform weights"),
                  ("ranks_uni_wghts_uni_dtf+","Uniform weights and normalization"),("ranks_uni_p_uni_dtf+","Uniform aggregation exponent and normalization"),
                  ("ranks_uni_wghts_uni_p","Uniform weights and aggregation exponent"),("ranks_uni_wghts_uni_p_uni_dtf+","All three uniform")]:
    alt = x26.parse(tab).set_index("Iso 3")["Prosperity Index"].reindex(def_rank.index)
    v26[label] = dict(mean=float(np.abs(alt-def_rank).mean()), max=int(np.abs(alt-def_rank).max()), spearman=float(spearmanr(alt,def_rank).correlation))
sc26 = x26.parse("scores_uni_wghts_uni_p").set_index("Iso 3")
DOM26 = ["Development","Freedom","Society"]; LEAF26 = ["Standard of Living","Health","Education","Civil Peace","Freedom of Dissent","Economic Freedom","Integrity of Families","Integrity of Communities","Integrity of Society","Integrity of State"]
Yd = sc26[DOM26].to_numpy(float); Yc = sc26["Prosperity Index"].to_numpy(float)
def identity_check(M, Y):
    rr = np.array([pearsonr(M[:,j],Y)[0] for j in range(M.shape[1])]); cc = M.std(0,ddof=1)/Y.std(ddof=1)
    Lx = (cc-rr)**2+(1-rr**2); Ld = np.var(M-Y[:,None],axis=0,ddof=1)/Y.var(ddof=1)
    return dict(max_resid=float(np.abs(Lx-Ld).max()), mean_a=float((rr*cc).mean()))
R["y2026"] = dict(n=int(len(sc26)), variants=v26, composite_is_domain_mean=float(np.abs(Yd.mean(1)-Yc).max()),
                  domain_identity=identity_check(Yd, Yc), leaf_identity=identity_check(sc26[LEAF26].to_numpy(float), sc26[LEAF26].to_numpy(float).mean(1)))
print("[2026]", R["y2026"]["composite_is_domain_mean"], R["y2026"]["domain_identity"], R["y2026"]["leaf_identity"]); [print("   %-48s %.2f %2d %.4f" % (k,v["mean"],v["max"],v["spearman"])) for k,v in v26.items()]

# ------------------------------------------------------------------ 11. rank leverage, setting-matched Sobol', reachability, positivity repair, density
from scipy.optimize import linprog
Rrank = c ** 2 * (1 - rho2)   # rank-relevant leverage: Var of the part of s_j not explained by Y, over Var(Y)
neutral = (a - 1) ** 2        # rank-neutral rescaling part of L (L = neutral + Rrank)
R["rank_leverage"] = dict(identity_resid=float(np.abs(L - (neutral + Rrank)).max()),
                          by_pillar={PILLARS[j]: dict(R=float(Rrank[j]), neutral=float(neutral[j]), L=float(L[j]), a=float(a[j]), c=float(c[j]), rho2=float(rho2[j])) for j in range(K)})

def weights_only_sobol(M, mode="additive", Nb=8192, nboot=200):
    """Sobol' total effects of the twelve weight coordinates with normalization and aggregation held fixed."""
    prob = {"num_vars": K, "names": PILLARS, "bounds": [[0, 1]] * K}
    Xw = salib_sample.sample(prob, Nb, calc_second_order=False, seed=SEED)
    Ww = map_weights(Xw)
    rk = np.empty((N, Xw.shape[0]), dtype=np.int16)
    for c0 in range(0, Xw.shape[0], 16384):
        rk[:, c0:c0 + 16384] = ranks_of(agg_scores(M, Ww[c0:c0 + 16384], mode))
    _, st, v = sobol_indices(rk, K, Nb)
    STx = (st * (v / v.sum())[:, None]).sum(0)
    brng = np.random.default_rng(SEED); bs = np.empty((nboot, K))
    for b in range(nboot):
        ii = brng.integers(0, Nb, Nb); _, stb, vb = sobol_indices(rk, K, Nb, ii)
        bs[b] = (stb * (vb / vb.sum())[:, None]).sum(0)
    return STx, (np.percentile(bs, 97.5, 0) - np.percentile(bs, 2.5, 0)) / 2, bs

cells = {}
for nm in NORM_OPTS:
    M = NORMS[nm](S); Ym = M.mean(1)
    cm = M.std(0, ddof=1) / Ym.std(ddof=1); rm = np.array([pearsonr(M[:, j], Ym)[0] for j in range(K)])
    Rm = cm ** 2 * (1 - rm ** 2); Lm = (cm - rm) ** 2 + 1 - rm ** 2
    STx, hwx, bs = weights_only_sobol(M, nboot=200 if nm == "identity" else 100)
    pr = bs[:, pf] / bs[:, sc_]
    cells[nm] = dict(ST=STx.tolist(), ST_hw=hwx.tolist(), R=Rm.tolist(), L=Lm.tolist(), c=cm.tolist(),
                     pearson_R_ST=float(pearsonr(Rm, STx)[0]), spearman_R_ST=float(spearmanr(Rm, STx).correlation),
                     pearson_L_ST=float(pearsonr(Lm, STx)[0]), spearman_L_ST=float(spearmanr(Lm, STx).correlation),
                     pearson_info_ST=float(pearsonr(1 - rm ** 2, STx)[0]), spearman_info_ST=float(spearmanr(1 - rm ** 2, STx).correlation),
                     spearman_eta_ST=float(spearmanr(eta[10], STx).correlation), spearman_rho2_ST=float(spearmanr(rho2, STx).correlation),
                     pair_ratio_ST=float(STx[pf] / STx[sc_]), pair_ratio_ST_ci=[float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))],
                     pair_ratio_R=float(Rm[pf] / Rm[sc_]), pair_ratio_L=float(Lm[pf] / Lm[sc_]))
    print(f"[weights-only {nm}] Pearson(R,ST)={cells[nm]['pearson_R_ST']:.3f} Spearman(R,ST)={cells[nm]['spearman_R_ST']:.3f} "
          f"PF/SC ST={cells[nm]['pair_ratio_ST']:.2f} R={cells[nm]['pair_ratio_R']:.2f} Spearman(eta2,ST)={cells[nm]['spearman_eta_ST']:.3f}")
STn = np.array(cells["identity"]["ST"])
def concord(x, y): return int(sum((x[i] - x[j]) * (y[i] - y[j]) > 0 for i in range(K) for j in range(i + 1, K)))
close = sorted((abs(rho2[i] - rho2[j]), i, j) for i in range(K) for j in range(i + 1, K))[:4]
R["weights_only"] = dict(cells=cells,
                         closest_pairs=[dict(pair=[PILLARS[i], PILLARS[j]], d_rho2=float(d), R_ratio=float(Rrank[i] / Rrank[j]), L_ratio=float(L[i] / L[j]), ST_ratio=float(STn[i] / STn[j])) for d, i, j in close],
                         pairs_ordered=dict(R=concord(Rrank, STn), L=concord(L, STn), neg_rho2=concord(-rho2, STn), neg_eta2=concord(-eta[10], STn), total=K * (K - 1) // 2))
pd.DataFrame(dict(pillar=PILLARS, c=c, rho2=rho2, rank_leverage=Rrank, L=L, rank_neutral=neutral, ST_native_weights_only=STn,
                  ST_native_weights_only_hw=cells["identity"]["ST_hw"], ST_joint_design=ST_w)).to_csv(f"{OUT}/rank_leverage.csv", index=False)

# local rank displacement at the published weights, step h along (e_j - w0)
w0 = np.ones(K) / K; loc = {}
for h in (0.02, 0.05):
    shv = np.array([np.abs(ranks_of(S @ (w0 + h * (np.eye(K)[j] - w0))) - off_rank).mean() for j in range(K)])
    loc[str(h)] = dict(shift=shv.tolist(), pearson_R=float(pearsonr(shv, Rrank)[0]), spearman_R=float(spearmanr(shv, Rrank).correlation),
                       pearson_L=float(pearsonr(shv, L)[0]), spearman_L=float(spearmanr(shv, L).correlation))
R["local_rank_shift"] = loc

# Table 5 framings: can some weight vector on the published scores reproduce the ranking? (max separating margin > 0)
def max_margin(score):
    order_ = np.argsort(-score); rows = [S[v] - S[u] for u, v in zip(order_[:-1], order_[1:]) if score[u] > score[v] + 1e-9]
    A_ = np.array(rows); m_ = len(A_)
    res_ = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.hstack([A_, np.ones((m_, 1))]), b_ub=np.zeros(m_),
                   A_eq=np.r_[np.ones(K), 0.0][None, :], b_eq=[1.0], bounds=[(0, None)] * K + [(None, None)], method="highs")
    return float(-res_.fun)
fr_scores = {"Three domains, geometric mean across domains": np.exp(np.log(dom).mean(1)), "Pillars, min–max normalization": n_minmax(S).mean(1),
             "Pillars, geometric mean": np.exp(np.log(S).mean(1)), "First principal component": pc1, "Pillars, z-score normalization": Zs.mean(1),
             "Pillars, percentile-rank normalization": n_rank(S).mean(1)}
for k_, scv in fr_scores.items():
    mm_ = max_margin(scv); R["framings"][k_]["max_margin"] = mm_; R["framings"][k_]["reachable_by_reweighting"] = bool(mm_ > 0)
print("[framings reachable]", {k_: (R["framings"][k_]["reachable_by_reweighting"], round(R["framings"][k_]["max_margin"], 3)) for k_ in fr_scores})

# sensitivity of the joint design to the geometric-mean positivity repair (shifted minimum = frac x column range)
def evaluate_repair(X, frac):
    m_ = X.shape[0]; out = np.empty((N, m_), dtype=np.int16)
    W = map_weights(X[:, :K]); nsel = np.minimum((X[:, K] * 4).astype(int), 3); asel = np.minimum((X[:, K + 1] * 2).astype(int), 1)
    for ni, nm in enumerate(NORM_OPTS):
        M = NORMS[nm](S); cmin = M.min(0); rngc = M.max(0) - cmin
        Mg = M + np.where(cmin <= 0, -cmin + frac * rngc, 0.0)
        for ai, ag in enumerate(AGG_OPTS):
            sel = np.where((nsel == ni) & (asel == ai))[0]
            for c0 in range(0, len(sel), 16384):
                cols = sel[c0:c0 + 16384]
                out[:, cols] = ranks_of(M @ W[cols].T if ag == "additive" else np.exp(np.log(Mg) @ W[cols].T))
    return out
Xp = salib_sample.sample(problem, 8192, calc_second_order=False, seed=SEED)
rep = {}
for frac in (0.01, 0.05):
    _, st, v = sobol_indices(evaluate_repair(Xp, frac), K + 2, 8192)
    STr = (st * (v / v.sum())[:, None]).sum(0)
    rep[str(frac)] = dict(ST_aggregation=float(STr[K + 1]), ST_normalization=float(STr[K]), agg_exceeds=int((STr[:K] < STr[K + 1]).sum()),
                          norm_exceeds=int((STr[:K] < STr[K]).sum()), top4_weights=[PILLARS[j] for j in np.argsort(-STr[:K])[:4]],
                          max_abs_change_weights=float(np.abs(STr[:K] - sob["ST"][:K]).max()))
R["positivity_repair"] = dict(baseline=dict(ST_aggregation=float(sob["ST"][K + 1]), ST_normalization=float(sob["ST"][K]),
                                            agg_exceeds=R["sobol"]["agg_exceeds_n_weights"], norm_exceeds=R["sobol"]["norm_exceeds_n_weights"],
                                            top4_weights=[PILLARS[j] for j in np.argsort(-sob["ST"][:K])[:4]]), variants=rep)
print("[repair]", R["positivity_repair"])

# score density around each country (published scores within one point) and interval width
dens = np.array([int((np.abs(off_score - off_score[i]) <= 1.0).sum() - 1) for i in range(N)])
R["density"] = dict(spearman_density_width=float(spearmanr(dens, width).correlation),
                    tier_mean_neighbours=[float(dens[(off_rank >= a_) & (off_rank <= b_)].mean()) for a_, b_ in tiers])

# network centrality compared with loadings, eta2 and sensitivities
R["network"]["compare"] = {nm_: dict(strength=float(spearmanr(strength, v_).correlation), expected_influence=float(spearmanr(ei, v_).correlation))
                           for nm_, v_ in [("PC1 loading", loadings[:, 0]), ("eta2", eta[10]), ("ST joint design", ST_w),
                                           ("ST native weights-only", STn), ("rank leverage", Rrank)]}
# high PC1 loading and sensitivity
R["pca"]["high_loading"] = [dict(pillar=PILLARS[j], pc1=float(loadings[j, 0]), ST_joint_rank=int((ST_w > ST_w[j]).sum() + 1),
                                 ST_native_rank=int((STn > STn[j]).sum() + 1)) for j in np.argsort(-loadings[:, 0])[:4]]
# heterogeneity of country-level profiles, and profiles by tier
R["country"]["share_spread"] = {PILLARS[j]: dict(min=float(shares[:, j].min()), median=float(np.median(shares[:, j])), max=float(shares[:, j].max()),
                                                 n_below_0_02=int((shares[:, j] < 0.02).sum()), n_above_0_25=int((shares[:, j] > 0.25).sum())) for j in range(K)}
tp = []
for (a_, b_), t_ in zip(tiers, tier_tab):
    msk = (off_rank >= a_) & (off_rank <= b_); ms = shares[msk].mean(0)
    tp.append(dict(tier=t_["tier"], mean_hhi=float(hhi[msk].mean()), lead_pillar=PILLARS[int(ms.argmax())], lead_share=float(ms.max()),
                   median_rank_variance=float(np.median(V[msk]))))
R["country"]["tier_profile"] = tp

# ------------------------------------------------------------------ 12. interval widths under every prior; implied weights of the linear framings
rng2 = np.random.default_rng(SEED)          # same draw order as section 3, so the uniform widths reproduce exactly
prior_widths = {}
for prior in ["near_equal", "grid", "uniform"]:
    if prior == "near_equal": Wt = rng2.dirichlet(np.ones(K) * 50.0, M_UA)
    elif prior == "grid":
        d_ = rng2.choice([0.5, 1.0, 1.5, 2.0], size=(M_UA, K)); Wt = d_ / d_.sum(1, keepdims=True)
    else: Wt = rng2.dirichlet(np.ones(K), M_UA)
    rk_ = ranks_of(S @ Wt.T); wd = np.percentile(rk_, 95, 1) - np.percentile(rk_, 5, 1)
    prior_widths[prior] = dict(median=float(np.median(wd)), max=float(wd.max()), argmax=str(names[int(wd.argmax())]), n_over_10=int((wd > 10).sum()),
                               tier_median=[float(np.median(wd[(off_rank >= a_) & (off_rank <= b_)])) for a_, b_ in tiers])
assert abs(prior_widths["uniform"]["median"] - R["intervals"]["median_width"]) < 1e-9
R["prior_widths"] = prior_widths
print("[prior widths]", {k_: (v_["median"], v_["max"], v_["argmax"]) for k_, v_ in prior_widths.items()})
implied = {"Pillars, min–max normalization": 1 / (S.max(0) - S.min(0)), "Pillars, z-score normalization": 1 / S.std(0), "First principal component": evecs[:, 0] / S.std(0)}
for k_, wv in implied.items():
    wv = wv / wv.sum()
    R["framings"][k_]["implied_weights_range"] = [float(wv.min()), float(wv.max())]
    R["framings"][k_]["ranks_equal_reweighting"] = bool((ranks_of(fr_scores[k_]) == ranks_of(S @ wv)).all())
print("[implied weights]", {k_: (R["framings"][k_]["implied_weights_range"], R["framings"][k_]["ranks_equal_reweighting"]) for k_ in implied})
keep_ = np.array([j != pf for j in range(K)])
R["weights_only"]["excluding_personal_freedom"] = dict(pearson_R_ST=float(pearsonr(Rrank[keep_], STn[keep_])[0]), spearman_R_ST=float(spearmanr(Rrank[keep_], STn[keep_]).correlation),
                                                       pearson_L_ST=float(pearsonr(L[keep_], STn[keep_])[0]), spearman_L_ST=float(spearmanr(L[keep_], STn[keep_]).correlation))
vif_arr = np.array([R["pca"]["vif"][p_] for p_ in PILLARS])
R["importance"]["lowest_vif_are_top_ST"] = bool(set(np.argsort(vif_arr)[:4]) == set(np.argsort(-ST_w)[:4]))
print("[excl PF]", R["weights_only"]["excluding_personal_freedom"], "| lowest-VIF four = top-four joint S_T:", R["importance"]["lowest_vif_are_top_ST"])

# ------------------------------------------------------------------ 13. leave-one-out correlation as an exact function of a_j and rank leverage
loo = {}
for nm in NORM_OPTS:
    M = NORMS[nm](S); Ym = M.mean(1); cm = M.std(0, ddof=1) / Ym.std(ddof=1)
    rm = np.array([pearsonr(M[:, j], Ym)[0] for j in range(K)]); am = rm * cm; Rm = cm ** 2 * (1 - rm ** 2)
    direct = np.array([pearsonr(Ym, np.delete(M, j, 1).mean(1))[0] for j in range(K)])
    f_ = (K - am) / (K - 1); formula = f_ / np.sqrt(f_ ** 2 + Rm / (K - 1) ** 2)
    loo[nm] = dict(max_resid=float(np.abs(direct - formula).max()), a_min=float(am.min()), a_max=float(am.max()),
                   spearman_1minus_corr_R=float(spearmanr(1 - direct, Rm).correlation))
R["leave_one_out"] = dict(cells=loo, max_resid_all=float(max(v["max_resid"] for v in loo.values())))
print("[leave-one-out]", R["leave_one_out"])

# ------------------------------------------------------------------ 14. quantities quoted in Sections 4, 5.4, 7.7 and 7.9
bs_sh = sob["ST_c_boot"][:, :, :K] / sob["ST_c_boot"][:, :, :K].sum(2, keepdims=True)
share_hw = (np.percentile(bs_sh, 97.5, 0) - np.percentile(bs_sh, 2.5, 0)) / 2
pd.DataFrame(share_hw, columns=PILLARS, index=names).to_csv(f"{OUT}/country_share_hw.csv")
R["country"]["share_hw"] = dict(median=float(np.median(share_hw)), p95=float(np.percentile(share_hw, 95)), max=float(share_hw.max()))
R["country"]["share_extremes"] = {PILLARS[j]: dict(min_country=str(names[int(shares[:, j].argmin())]), min=float(shares[:, j].min()),
                                                   max_country=str(names[int(shares[:, j].argmax())]), max=float(shares[:, j].max())) for j in range(K)}
R["country"]["every_weight_below_0_02_for_two_and_above_0_20_for_one"] = bool(all(((shares[:, j] < 0.02).sum() >= 2) and ((shares[:, j] > 0.20).sum() >= 1) for j in range(K)))
R["country"]["max_local_independence"] = {PILLARS[j]: [str(names[int(resid[:, j].argmax())]), float(resid[:, j].max())] for j in range(K)}
for t_, (a_, b_) in zip(R["country"]["tier_profile"], tiers):
    msk = (off_rank >= a_) & (off_rank <= b_); t_["median_share"] = {PILLARS[j]: float(np.median(shares[msk, j])) for j in range(K)}
for cl in R["clusters"]["clusters"]:
    mm = lab == cl["cluster"]; j = PILLARS.index(cl["dominant"])
    cl["li_members"] = float(resid[mm, j].mean()); cl["li_others"] = float(resid[~mm, j].mean())
R["clusters"]["region_totals"] = {str(k_): int(v_) for k_, v_ in pd.Series(region).value_counts().items()}
clr_ = np.log(prof) - np.log(prof).mean(1, keepdims=True)
R["clusters"]["ari_clr_vs_euclidean"] = float(adjusted_rand_score(lab, fcluster(linkage(clr_, "ward"), kbest, "maxclust")))
STu = sob["ST_c"].mean(0); pi_ = sob["V"] / sob["V"].sum(); order_pi = np.argsort(-pi_)
n_half = int(np.searchsorted(np.cumsum(pi_[order_pi]), 0.5) + 1)
R["pooling"] = dict(unweighted_ST=STu.tolist(), unweighted_agg_exceeds=int((STu[:K] < STu[K + 1]).sum()), unweighted_norm_exceeds=int((STu[:K] < STu[K]).sum()),
                    unweighted_top2=[PILLARS[j] for j in np.argsort(-STu[:K])[:2]], n_countries_half_weight=n_half,
                    half_weight_ranks_34_134=int(((off_rank[order_pi[:n_half]] >= 34) & (off_rank[order_pi[:n_half]] <= 134)).sum()))
ebq = {}
for q_ in (0.01, 0.05, 0.10):
    ok_ = np.where(p[o] <= q_ * np.arange(1, m + 1) / m)[0]; kept_ = np.zeros(m, bool)
    if len(ok_): kept_[o[:ok_.max() + 1]] = True
    sel_ = np.zeros((K, K)); sel_[iu] = np.where(kept_, r, 0); sel_ = sel_ + sel_.T
    ebq[str(q_)] = dict(n_kept=int(kept_.sum()), zero_strength=[PILLARS[j] for j in range(K) if np.abs(sel_[j]).sum() == 0], min_edges=int((sel_ != 0).sum(1).min()))
R["network"]["edges_by_q"] = ebq
Wg = np.random.default_rng(SEED).dirichlet(np.ones(K), 1000)
gm_ = ranks_of(agg_scores(n_minmax(S), Wg, "geometric")); gz_ = ranks_of(agg_scores(n_zscore(S), Wg, "geometric"))
R["positivity_repair"]["geo_minmax_vs_zscore_mean_abs_rank_diff"] = float(np.abs(gm_ - gz_).mean())
print("[section 14]", R["pooling"], R["network"]["edges_by_q"], R["clusters"]["ari_clr_vs_euclidean"], R["country"]["share_hw"],
      R["positivity_repair"]["geo_minmax_vs_zscore_mean_abs_rank_diff"], R["country"]["every_weight_below_0_02_for_two_and_above_0_20_for_one"])

# ------------------------------------------------------------------ save
np.save(f"{OUT}/_ranks_uniform.npy", ranks_uniform); np.save(f"{OUT}/_ST_country.npy", STc); np.save(f"{OUT}/_S1_country.npy", sob["S1_c"])
lev.sort_values("L", ascending=False).to_csv(f"{OUT}/leverage.csv", index=False)
pd.DataFrame(dict(eigenvalue=evals, pa95=pa95)).to_csv(f"{OUT}/eigenvalues.csv", index=False)
json.dump(R, open(f"{OUT}/results.json","w"), indent=1, default=float)
print("[done]")
