"""Regenerates the eight figures from outputs/. 300 dpi JPEG, no titles, keys inside the figure,
title-case labels, units on axes where the quantity has units, per the JHSS author guidelines."""
import os, json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.environ.get("LPI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs")
FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(f"{OUT}/results.json"))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300})
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.jpg", dpi=300, format="jpeg", bbox_inches="tight", pad_inches=0.05, pil_kwargs={"quality": 95}); plt.close(fig)

# Figure 1: eigenvalues vs parallel analysis
ev = pd.read_csv(f"{OUT}/eigenvalues.csv"); k = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(k, ev["eigenvalue"], "o-", color="black", label="Observed Eigenvalue")
ax.plot(k, ev["pa95"], "s--", color="grey", label="Parallel Analysis, 95th Percentile")
ax.axhline(1, color="grey", lw=0.8, ls=":", label="Kaiser Criterion (Eigenvalue = 1)")
ax.set_yscale("log"); ax.set_xlabel("Principal Component"); ax.set_ylabel("Eigenvalue (Dimensionless, Log Scale)"); ax.set_xticks(k); ax.legend(frameon=False)
save(fig, "fig1_eigenvalues")

# Figure 2: average absolute rank shift by prior
ua = R["ua"]; labels = ["Near-Equal", "Grid {0.5, 1, 1.5, 2}", "Uniform"]; keys = ["near_equal", "grid", "uniform"]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
m = [ua[x]["avg_shift"] for x in keys]; lo = [ua[x]["avg_shift"]-ua[x]["shift_p5"] for x in keys]; hi = [ua[x]["shift_p95"]-ua[x]["avg_shift"] for x in keys]
ax.bar(labels, m, color=["#bbbbbb", "#888888", "#444444"], yerr=[lo, hi], capsize=4, label="Mean Over 10,000 Draws (Bars: 5th–95th Percentile)")
ax.set_xlabel("Weighting Prior"); ax.set_ylabel("Average Absolute Rank Shift (Places)")
for i, v in enumerate(m): ax.text(i + 0.22, v + 0.1, f"{v:.2f}", ha="left", va="bottom")
ax.legend(frameon=False, loc="upper left", fontsize=7)
save(fig, "fig2_rank_shift")

# Figure 3: per-country 90 % intervals
c = pd.read_csv(f"{OUT}/rank_uncertainty.csv").sort_values("off_rank")
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.fill_between(c["off_rank"], c["rank_p5"], c["rank_p95"], color="#bbbbbb", label="90 Percent Rank Interval")
ax.plot(c["off_rank"], c["rank_median"], ".", color="black", ms=3, label="Median Simulated Rank")
ax.plot(c["off_rank"], c["off_rank"], "-", color="grey", lw=0.8, label="Published Rank")
ax.set_xlabel("Published Rank, 2023 (Places)"); ax.set_ylabel("Simulated Rank (Places)"); ax.invert_yaxis(); ax.legend(frameon=False, loc="upper right")
save(fig, "fig3_rank_intervals")

# Figure 4: interval width vs published rank, with tier means ± 2 s.e.
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.scatter(c["off_rank"], c["width_90"], s=10, color="black", alpha=0.6, label="Country")
bins = pd.cut(c["off_rank"], [0, 33, 67, 100, 134, 167]); g = c.groupby(bins, observed=True)["width_90"]
xs = [b.mid for b in g.mean().index]; ax.errorbar(xs, g.mean(), yerr=2*g.std()/np.sqrt(g.count()), fmt="s-", color="grey", capsize=3, label="Tier Mean ± 2 Standard Errors")
ax.set_xlabel("Published Rank, 2023 (Places)"); ax.set_ylabel("Width of 90 Percent Rank Interval (Places)"); ax.legend(frameon=False, loc="upper left")
save(fig, "fig4_interval_width")

# Figure 5: Sobol' indices with bootstrap bars
sa = pd.read_csv(f"{OUT}/sobol_indices.csv").sort_values("ST")
fig, ax = plt.subplots(figsize=(6.2, 4.4)); y = np.arange(len(sa))
ax.barh(y - 0.2, sa["S1"], 0.4, xerr=sa["S1_hw"], color="#999999", label="First-Order Index $S$ (95% Bootstrap Interval)", capsize=2)
ax.barh(y + 0.2, sa["ST"], 0.4, xerr=sa["ST_hw"], color="#333333", label="Total-Effect Index $S_T$ (95% Bootstrap Interval)", capsize=2)
ax.set_yticks(y); ax.set_yticklabels([x.replace(" weight", "").replace("selector", "Selector") for x in sa["input"]]); ax.set_xlabel("Variance-Weighted Sobol' Index (Dimensionless)"); ax.legend(frameon=False, loc="lower right", fontsize=7)
save(fig, "fig5_sobol")

# Figure 6: realized importance share vs nominal, with resampling bars
imp = pd.read_csv(f"{OUT}/realized_importance.csv")
data23 = next(p_ for p_ in (os.environ.get("LPI_DATA_2023", ""), os.path.join(ROOT, "data", "Dataset_Legatum_Prosperity_Index_2023.xlsx"), os.path.join(ROOT, "Dataset_Legatum_Prosperity_Index_2023.xlsx")) if p_ and os.path.exists(p_))
pil = pd.read_excel(data23, sheet_name="Pillars x 12")
W = pil.pivot(index="area_code", columns="pillar_name", values="score_2023")[list(imp["pillar"])].to_numpy(float)
def eta2(x, y, B=10):
    b = pd.qcut(x, B, labels=False, duplicates="drop"); ybar = y.mean()
    return sum((b == u).sum()*(y[b == u].mean()-ybar)**2 for u in np.unique(b))/((y-ybar)**2).sum()
rng = np.random.default_rng(42); boots = []
for _ in range(1000):
    ii = rng.integers(0, len(W), len(W)); Wb = W[ii]; Yb = Wb.mean(1); e = np.array([eta2(Wb[:, j], Yb) for j in range(12)]); boots.append(e/e.sum())
boots = np.array(boots); err = np.abs(np.percentile(boots, [2.5, 97.5], 0) - imp["share"].to_numpy())
fig, ax = plt.subplots(figsize=(6.2, 3.8)); y = np.arange(12)[::-1]
ax.barh(y, imp["share"], color="#555555", xerr=err, capsize=2, label="Realized Importance Share ($\\eta^2$, 95% Resampling Interval)")
ax.axvline(1/12, color="black", ls="--", lw=0.9, label="Nominal Share (1/12)")
ax.set_yticks(y); ax.set_yticklabels(imp["pillar"]); ax.set_xlabel("Share of Total Realized Importance (Dimensionless)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, fontsize=7)
save(fig, "fig6_importance")

# Figure 7: rank leverage c^2(1 - rho^2) vs weights-only total effect on the published scores, matched pair joined
lev = pd.read_csv(f"{OUT}/rank_leverage.csv").drop(columns=["L"]).rename(columns={"rank_leverage": "L", "ST_native_weights_only": "ST", "ST_native_weights_only_hw": "ST_hw"})
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.errorbar(lev["L"], lev["ST"], yerr=lev["ST_hw"], fmt="o", color="black", ms=4, capsize=2, label="Pillar (95% Bootstrap Interval)")
short = {"Infrastructure and Market Access": "Infrastructure", "Investment Environment": "Investment Env.", "Enterprise Conditions": "Enterprise Cond.",
         "Safety and Security": "Safety & Security", "Economic Quality": "Econ. Quality"}
offs = {"Living Conditions": (6, -10, "left"), "Personal Freedom": (-6, 3, "right")}
# crowded lower-left cluster: labels placed in free space with thin leader lines
placed = {"Natural Environment": (0.02, 0.107),"Social Capital": (0.33, 0.091), "Governance": (0.34, 0.079),
          "Enterprise Cond.": (0.27, 0.060), "Health": (0.27, 0.048), "Econ. Quality": (0.02, 0.093),
          "Infrastructure": (0.02, 0.079), "Investment Env.": (0.02, 0.046)}
for _, r in lev.iterrows():
    lab_ = short.get(r["pillar"], r["pillar"])
    if lab_ in placed:
        ax.annotate(lab_, (r["L"], r["ST"]), xytext=placed[lab_], textcoords="data", fontsize=6.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, shrinkA=1, shrinkB=3))
    else:
        dx, dy, ha = offs.get(r["pillar"], (4, 3, "left"))
        ax.annotate(lab_, (r["L"], r["ST"]), fontsize=6.5, xytext=(dx, dy), textcoords="offset points", ha=ha)
ax.set_xlim(0.0, 1.02); ax.set_ylim(0.04, 0.30)
pair = lev.set_index("pillar").loc[["Social Capital", "Personal Freedom"]]
ax.plot(pair["L"], pair["ST"], ls="--", color="grey", lw=1, zorder=0, label="Matched Pair (Equal Alignment, Unequal Spread)")
ax.set_xlabel("Rank Leverage $c_j^2(1-\\rho_j^2)$ From (eq5), Dimensionless"); ax.set_ylabel("Total-Effect Index $S_T$, Weights Only"); ax.legend(frameon=False, loc="upper left", fontsize=7)
save(fig, "fig7_leverage")

# Figure 8: structural reframings vs weight priors (priors carry 5th-95th percentile bars)
fr = R["framings"]
TITLE = {"Three domains, equal weight": "Three Domains, Equal Weight (As Published)",
         "Three domains, geometric mean across domains": "Three Domains, Geometric Mean",
         "Pillars, min–max normalization": "Pillars, Min–Max Normalization",
         "Pillars, geometric mean": "Pillars, Geometric Mean",
         "First principal component": "First Principal Component",
         "Pillars, z-score normalization": "Pillars, Z-Score Normalization",
         "Pillars, percentile-rank normalization": "Pillars, Percentile-Rank Normalization"}
items = [(TITLE.get(k, k), v["mean"], None) for k, v in fr.items()]
items += [(f"Weight Prior: {l}", ua[x]["avg_shift"], (ua[x]["avg_shift"]-ua[x]["shift_p5"], ua[x]["shift_p95"]-ua[x]["avg_shift"])) for l, x in zip(labels, keys)]
items = sorted(items, key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(6.6, 3.9)); y = np.arange(len(items))
for yi, (name, val, err) in zip(y, items):
    if err is None:
        ax.barh(yi, val, color="#999999")
    else:
        ax.barh(yi, val, color="#333333", xerr=[[err[0]], [err[1]]], capsize=2)
ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in items]); ax.set_xlabel("Mean Absolute Rank Shift (Places)")
ax.legend(handles=[Patch(color="#999999", label="Alternative Construction (Deterministic)"), Patch(color="#333333", label="Weight Prior (Bars: 5th–95th Percentile)")], frameon=False, loc="lower right", fontsize=7)
save(fig, "fig8_structural")
print("figures written to", FIG, sorted(os.listdir(FIG)))
