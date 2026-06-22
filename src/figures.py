"""Figures + refined 'weights vs importance' using first-order correlation ratio (main effect)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEED=42; rng=np.random.default_rng(SEED)
OUT=os.path.join(os.path.dirname(__file__), ".."); FIG=os.path.join(OUT, "figures"); YEAR=2023
DATA=os.environ.get("LPI_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "Dataset_Legatum_Prosperity_Index_2023.xlsx"))
PILLARS=['Safety and Security','Personal Freedom','Governance','Social Capital',
         'Investment Environment','Enterprise Conditions','Infrastructure and Market Access',
         'Economic Quality','Living Conditions','Health','Education','Natural Environment']
SHORT={'Safety and Security':'Safety','Personal Freedom':'Pers. Freedom','Governance':'Governance',
       'Social Capital':'Soc. Capital','Investment Environment':'Investment','Enterprise Conditions':'Enterprise',
       'Infrastructure and Market Access':'Infrastructure','Economic Quality':'Econ. Quality',
       'Living Conditions':'Living Cond.','Health':'Health','Education':'Education','Natural Environment':'Nat. Env.'}
K=len(PILLARS)
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})

idx=pd.read_excel(DATA,sheet_name="Prosperity Index")
pil=pd.read_excel(DATA,sheet_name="Pillars x 12")
W=pil.pivot(index="area_code",columns="pillar_name",values=f"score_{YEAR}")[PILLARS]
meta=idx.set_index("area_code")[["area_name","area_group",f"score_{YEAR}",f"rank_{YEAR}"]]
meta.columns=["area_name","region","off_score","off_rank"]
df=meta.join(W); S=df[PILLARS].to_numpy(); names=df["area_name"].to_numpy()
off_rank=df["off_rank"].to_numpy(); base=S.mean(1); N=len(S)
df_out=pd.read_pickle(f"{OUT}/outputs/_df_out.pkl")
ranks_flat=np.load(f"{OUT}/outputs/_ranks_flat.npy")
summary=json.load(open(f"{OUT}/outputs/summary.json"))

# ---- refined importance: first-order correlation ratio eta^2 of overall score on pillar bins
def corr_ratio(x,y,bins=10):
    q=pd.qcut(x,bins,duplicates="drop"); g=pd.DataFrame({"y":y,"q":q})
    ybar=y.mean(); ss_between=sum(len(gr)* (gr["y"].mean()-ybar)**2 for _,gr in g.groupby("q",observed=True))
    ss_tot=((y-ybar)**2).sum(); return ss_between/ss_tot
eta=np.array([corr_ratio(S[:,j],base) for j in range(K)]); eta_share=eta/eta.sum()
wi=pd.DataFrame({"pillar":PILLARS,"nominal_weight":1/K,"main_effect_eta2":eta,
                 "importance_share":eta_share}).sort_values("importance_share",ascending=False)
wi.to_csv(f"{OUT}/outputs/weights_vs_importance.csv",index=False)

# ============ FIG 1: rank uncertainty intervals (all 167, ordered by official rank)
d=df_out.sort_values("off_rank")
fig,ax=plt.subplots(figsize=(7,9))
y=np.arange(len(d))
ax.fill_betweenx(y,d["rank_p5"],d["rank_p95"],alpha=0.25,color="#2c7fb8",label="90% interval")
ax.plot(d["rank_median"],y,".",ms=2.5,color="#08519c",label="median rank")
ax.plot(d["off_rank"],y,"|",ms=3,color="#cb181d",label="published rank")
ax.set_xlabel("Rank (1 = most prosperous)"); ax.set_ylabel("Country (ordered by published rank)")
ax.set_ylim(len(d),-1); ax.invert_xaxis()
ax.legend(loc="lower right",fontsize=8)
ax.set_title("Rank uncertainty under agnostic pillar weights\n(flat Dirichlet, 10,000 draws)",fontsize=11)
plt.savefig(f"{FIG}/fig1_rank_intervals.png"); plt.close()

# ============ FIG 2: width of 90% rank interval vs official rank (where is the index fragile?)
fig,ax=plt.subplots(figsize=(7,4.2))
rng90=d["rank_p95"]-d["rank_p5"]
ax.scatter(d["off_rank"],rng90,s=14,color="#238b45",alpha=0.8)
ax.set_xlabel("Published rank"); ax.set_ylabel("Width of 90% rank interval (places)")
ax.set_title("The middle of the table is where rankings are least determinate",fontsize=11)
# annotate a few
for _,r in d.iterrows():
    if r["off_rank"] in (1,167) or (r["rank_p95"]-r["rank_p5"])>=rng90.quantile(.985):
        ax.annotate(r["area_name"],(r["off_rank"],r["rank_p95"]-r["rank_p5"]),
                    fontsize=6,xytext=(3,3),textcoords="offset points")
plt.savefig(f"{FIG}/fig2_interval_width.png"); plt.close()

# ============ FIG 3: Sobol' first-order vs total-effect
sa=pd.DataFrame(summary["sobol"])
order=sa.sort_values("ST",ascending=True)
lab=[SHORT.get(n.replace("w_",""),n) if n.startswith("w_") else n for n in order["input"]]
fig,ax=plt.subplots(figsize=(7,5.2)); yy=np.arange(len(order))
ax.barh(yy-0.2,order["S1"],0.4,label="First-order $S_i$",color="#6baed6")
ax.barh(yy+0.2,order["ST"],0.4,label="Total-effect $S_{Ti}$",color="#08519c")
ax.set_yticks(yy); ax.set_yticklabels(lab,fontsize=8); ax.legend(fontsize=8)
ax.set_xlabel("Share of rank variance"); ax.set_title("Sobol' sensitivity of country ranks to methodological choices",fontsize=11)
plt.savefig(f"{FIG}/fig3_sobol.png"); plt.close()

# ============ FIG 4: nominal weight vs realized importance
w2=wi.sort_values("importance_share",ascending=True)
fig,ax=plt.subplots(figsize=(7,5)); yy=np.arange(len(w2))
ax.barh(yy,w2["importance_share"],color="#807dba",label="Realized importance (share of $\\eta^2$)")
ax.axvline(1/K,color="#cb181d",ls="--",lw=1.2,label=f"Equal nominal weight = {1/K:.3f}")
ax.set_yticks(yy); ax.set_yticklabels([SHORT[p] for p in w2["pillar"]],fontsize=8)
ax.set_xlabel("Share"); ax.legend(fontsize=8)
ax.set_title("Equal nominal weights do not imply equal influence",fontsize=11)
plt.savefig(f"{FIG}/fig4_weights_vs_importance.png"); plt.close()

# ============ FIG 5: distribution of avg rank shift across the 3 weighting schemes
fig,ax=plt.subplots(figsize=(7,4))
schemes=["dirichlet_conc","legatum_discrete","dirichlet_flat"]
labels=["Concentrated\n(near equal)","Legatum discrete\n{0.5,1,1.5,2}","Flat Dirichlet\n(agnostic)"]
vals=[summary["ua"][s]["avg_shift"] for s in schemes]
ax.bar(labels,vals,color=["#bdd7e7","#6baed6","#2171b5"])
for i,v in enumerate(vals): ax.text(i,v+0.05,f"{v:.2f}",ha="center",fontsize=9)
ax.set_ylabel("Average shift in rank (places)")
ax.set_title("How far ranks move depends on how agnostic the weights are",fontsize=11)
plt.savefig(f"{FIG}/fig5_rank_shift.png"); plt.close()

# refresh summary importance + fragility-by-tier table
d["tier"]=pd.cut(d["off_rank"],[0,33,67,100,134,167],
                 labels=["1-33","34-67","68-100","101-134","135-167"])
tier=d.groupby("tier",observed=True).agg(mean_90_range=("rank_range_90","mean"),
        median_90_range=("rank_range_90","median"),max_90_range=("rank_range_90","max")).reset_index()
tier.to_csv(f"{OUT}/outputs/fragility_by_tier.csv",index=False)
json.dump({"importance":wi.to_dict("records"),
           "tier_fragility":tier.to_dict("records")},
          open(f"{OUT}/outputs/summary_extra.json","w"),indent=2)
print("FIGURES + refined importance done")
print(tier.to_string(index=False))
print()
print(wi.to_string(index=False))
print("\nmedian 90% range:",summary["median_90_range"])
print("avg shift flat:",summary["ua"]["dirichlet_flat"]["avg_shift"])
