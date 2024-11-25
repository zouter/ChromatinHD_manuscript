# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Do CRISPR

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm

from chromatinhd_manuscript import crispri

from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
chd.set_default_device("cuda:0")

# %%
dataset_name = "hspc"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")
splitter = "5x5"
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)

model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / splitter / "magic" / "v33"
# model_folder = chd.get_output() / "pred" / dataset_name / "500k500k" / splitter / "magic" / "v34"
# models = chd.models.pred.model.better.Models(model_folder)

regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
    model_folder / "scoring" / "crispri" / "fulco_2019" / "regionmultiwindow",
)


folder = chd.get_output() / "data" / "crispri" / "fulco_2019"
data = pd.read_csv(folder / "data.tsv", sep="\t", index_col = 0)

# %%
# dataset_name = "hspc_focus"
# transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
# fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "500k500k")
# splitter = "5x5"
# folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)

# %% [markdown]
# ### Get genes

# %%
symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)
genes_oi = transcriptome.gene_id(["GATA1", "H1FX", "KLF1", "CALR"])

data = data.loc[data["Gene"].isin(symbols_oi)].copy()
data["gene"] = transcriptome.gene_id(data["Gene"]).values

# %%
# regionpairwindow = chd.models.pred.interpret.RegionPairWindow(models.path / "scoring" / "regionpairwindow")
# for gene_oi in genes_oi:
#     windows_oi = regionmultiwindow.design.query("window_size == 100").index
#     windows_oi = windows_oi[regionmultiwindow.scores["deltacor"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") < -0.0005]
#     # windows_oi = windows_oi[regionmultiwindow.scores["lost"].sel_xr(gene_oi).sel(phase = "test").sel(window = windows_oi.tolist()).mean("fold") > 1e-3]
#     # windows_oi = windows_oi
#     design = regionmultiwindow.censorer.design.loc[["control"] + windows_oi.tolist()]

#     censorer = chd.models.pred.interpret.censorers.WindowCensorer(fragments.regions.window)
#     censorer.design = design
#     design.shape

#     regionpairwindow.score(models, regions = [gene_oi], censorer = censorer)

# %%
# regionmultiwindow.interpolate()

# %% [markdown]
# ## Check enrichment

# %%
genes_oi = genes_oi[regionmultiwindow.scores["scored"].sel_xr(genes_oi).all("fold").values]

# %%
joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 100)
# joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 100, regionpairwindow=regionpairwindow)
joined_observed = joined_all.loc[~pd.isnull(joined_all["HS_LS_logratio"])].copy()

# %%
deltacor_cutoff = -0.001
lfc_cutoff = np.log(1.5)

# %%
joined_observed["significant_expression"] = joined_observed["HS_LS_logratio"].abs() > lfc_cutoff
joined_observed["significant_chd"] = joined_observed["deltacor"] < deltacor_cutoff

# %%
genescores = []
for gene_oi, joined_oi in joined_observed.groupby("gene"):
    confirmed_n  = (joined_oi["significant_expression"] & joined_oi["significant_chd"]).sum()
    total_n = joined_oi["significant_chd"].sum()
    confirmed = confirmed_n / total_n

    randoms = []
    for i in range(1000):
        randoms.append((joined_oi.iloc[np.random.permutation(np.arange(len(joined_oi)))]["significant_expression"].values & joined_oi["significant_chd"]).sum() / joined_oi["significant_chd"].sum())
    randoms = np.array(randoms)

    p = (randoms > confirmed).mean()

    genescores.append({
        "gene": gene_oi,
        "symbol": transcriptome.symbol(gene_oi),
        "observed": confirmed,
        "random": randoms.mean(),
        "p": p
    })

    print(f"{transcriptome.symbol(gene_oi)} Observed {confirmed:.2%} Random {randoms.mean():.2%} p-value {p:.2f}")
genescores = pd.DataFrame(genescores).set_index("gene")
genescores["dispersions_norm"] = transcriptome.var.loc[genescores.index, "dispersions_norm"].values
genescores["dispersions"] = transcriptome.var.loc[genescores.index, "dispersions"].values

genescores.style.bar()

# %%
confirmed_n  = (joined_observed["significant_expression"] & joined_observed["significant_chd"]).sum()
total_n = joined_observed["significant_chd"].sum()
confirmed = confirmed_n / total_n

randoms = []
for i in range(5000):
    randoms.append((joined_observed.iloc[np.random.permutation(np.arange(len(joined_observed)))]["significant_expression"].values & joined_observed["significant_chd"]).sum() / joined_observed["significant_chd"].sum())
randoms = np.array(randoms)

fig, ax =plt.subplots(figsize = (5, 2))
ax.hist(randoms, bins = np.linspace(0, 1, 20), density = True)
ax.axvline(confirmed, color = "red")
(randoms >= confirmed).mean(), f"{confirmed:.2%}", f"{randoms.mean():.2%}"

# %%
fig, ax =plt.subplots(figsize = (5, 2))
colors = pd.Series(sns.color_palette("tab20", len(joined_observed["gene"].unique())), index = joined_observed["gene"].unique())
ax.scatter(joined_observed["deltacor_positive"], joined_observed["HS_LS_logratio"], c = colors[joined_observed["gene"]], s = 2)
for gene, color in colors.items():
    ax.scatter([], [], c = color, label = transcriptome.symbol(gene))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axvline(deltacor_cutoff, color = "red", dashes = (2, 2))
ax.axhline(0, color = "#333", lw = 1, zorder = 0)
ax.axhline(-lfc_cutoff, color = "red", dashes = (2, 2))
ax.set_xlabel("$\Delta$ cor")
ax.set_ylabel("Log-fold change")
ax.set_yticks([-lfc_cutoff, 0, lfc_cutoff])
ax.set_yticklabels(["1/2", "1", "2"])
ax.set_ylabel(f"CRISPRi\nfold enrichment\nhigh vs low RNA", rotation = 0, ha = "right", va = "center")
ax.set_yticks(np.log([0.125, 0.25, 0.5, 1, 2]))
ax.set_yticklabels(["⅛", "¼", "½", 1, 2])
ax.set_xlim(ax.get_xlim())
ax.axvspan(ax.get_xlim()[0], deltacor_cutoff, color = "grey", zorder = -1, alpha = 0.1)
ax.annotate("{} High $\Delta$ cor+ low CRISPRi depletion".format(total_n - confirmed_n), (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top", fontsize = 8)
ax.annotate("{} High $\Delta$ cor+ high CRISPRi depletion".format(confirmed_n), (0.05, 0.05), xycoords = "axes fraction", ha = "left", va = "bottom", fontsize = 8)

# %%
import fisher
contingency = pd.crosstab(joined_observed["significant_expression"], joined_observed["significant_chd"])
print(fisher.pvalue(*contingency.values.flatten()))
odds = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / (contingency.iloc[1, 0] * contingency.iloc[0, 1])
print(odds)

# %% [markdown]
# ## Focus on one gene

# %%
gene_oi = transcriptome.gene_id("H1FX")
# gene_oi = transcriptome.gene_id("KLF1")
# gene_oi = transcriptome.gene_id("GATA1")
# gene_oi = transcriptome.gene_id("CALR")
# gene_oi = transcriptome.gene_id("GATA1")
# gene_oi = transcriptome.gene_id("MYO1D")
# gene_oi = transcriptome.gene_id("HDAC6")
# gene_oi = transcriptome.gene_id("CNBP")
# gene_oi = transcriptome.gene_id("NFE2")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

joined = joined_all.query("gene == @gene_oi").copy()
joined["score"] = joined["HS_LS_logratio"] * joined["deltacor"]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.1))

# binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]
binwidth = 100

# gene_oi = transcriptome.gene_id("GATA1")
# window = [-10000, 20000]  # GATA1 region oi
# arrows = [{"position": 14000, "orientation": "right", "y": 0.5}]

# window = [-20000, 20000]  # HDAC6 region oi
# arrows = [
    # {"position": 14000, "orientation": "right", "y": 0.5}
# ]

# window = [-260000, -250000]
# arrows = []

# gene_oi = transcriptome.gene_id("KLF1")
# window = [-10000, 10000] # KLF1 TSS
# arrows = [
#     # {"position": -3500, "orientation": "left", "y": 0.5},
#     {"position": 1000, "orientation": "left", "y": 0.5},
#     {"position": -650, "orientation": "right", "y": 0.5},
# ]

# window = [-70000, -50000] # KLF1 upstream
# arrows = []

# window = [-65000, -40000] # CALR upstream
# arrows = []

# window = [-10000, 20000] # whatever
# arrows = []

window = fragments.regions.window  # all
arrows = []

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width=10, window=window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(
    chd.models.pred.plot.Pileup(regionmultiwindow.get_plotdata(gene_oi), window=window, width=10)
)

panel, ax = fig.main.add_under(
    chdm.plotting.Peaks(
        region,
        chd.get_output() / "peaks" / dataset_name,
        window=fragments.regions.window,
        width=10,
        peakcallers=["cellranger", "macs2_improved", "macs2_leiden_0.1", "macs2_leiden_0.1_merged", "encode_screen"],
    )
)
ax.set_xlim(*window)
ax.set_xticks([])

panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 1)))
ax.bar(
    joined["window_mid"],
    joined["HS_LS_logratio"],
    lw=0,
    width=binwidth,
    color="#333",
)
ax.step(
    joined["window_mid"],
    joined["HS_LS_logratio"].fillna(0.0),
    lw=0.5,
    color="#AAA",
    where="mid",
    zorder=0,
    linestyle="--",
)
ax.set_ylim(*ax.get_ylim()[::-1])
ax.set_xlim(*window)
# ax.set_ylabel(f"CRISPRi\nfold enrichment\ngrowth selection", rotation=0, ha="right", va="center")
ax.set_ylabel(f"CRISPRi\nfold enrichment\nhigh vs low {transcriptome.symbol(gene_oi)}", rotation = 0, ha = "right", va = "center")
ax.set_xticks([])
ax.set_yticks(np.log([0.125, 0.25, 0.5, 1, 2]))
ax.set_yticklabels(["⅛", "¼", "½", 1, 2])

panel, ax = fig.main.add_under(
    chd.models.pred.plot.Predictivity(regionmultiwindow.get_plotdata(gene_oi), window=window, width=10, limit = -0.1)
)
for arrow in arrows:
    panel.add_arrow(**arrow)

# interaction
# panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 0.5)))
# plotdata = pd.DataFrame(
#     {
#         "deltacor":regionpairwindow.scores[gene_oi].mean("fold")["deltacor"].to_pandas().values,
#         "deltadeltacor":regionpairwindow.interaction[gene_oi].mean("fold").mean("window1").to_pandas()
#     }
# )
# plotdata.index.name = "window"
# plotdata = regionmultiwindow.design.query("window_size == 100.").join(plotdata)
# plotdata = plotdata.fillna(0.)
# ax.plot(
#     plotdata["window_mid"],
#     plotdata["deltadeltacor"],
#     color="#333",
# )
# ax.set_xlim(*window)

# panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 0.5)))
# # # !wget https://www.encodeproject.org/files/ENCFF010PHG/@@download/ENCFF010PHG.bigWig
# import pyBigWig
# file = pyBigWig.open("ENCFF010PHG.bigWig")
# plotdata = pd.DataFrame({"value":file.values(region["chrom"], region["start"], region["end"])})
# plotdata["position"] = np.arange(*fragments.regions.window)[::int(region["strand"])]
# ax.plot(plotdata["position"], plotdata["value"], color = "#333")
# ax.set_xlim(*window)
# ax.set_ylim(0, 20)
# ax.set_ylabel("H3K27ac\nsignal", rotation = 0, ha = "right", va = "center")

# panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 0.5)))
# # # !wget https://www.encodeproject.org/files/ENCFF814IYI/@@download/ENCFF814IYI.bigWig
# import pyBigWig
# file = pyBigWig.open("ENCFF814IYI.bigWig")
# plotdata = pd.DataFrame({"value":file.values(region["chrom"], region["start"], region["end"])})
# plotdata["position"] = np.arange(*fragments.regions.window)[::int(region["strand"])]
# ax.plot(plotdata["position"], plotdata["value"], color = "#333")
# ax.set_xlim(*window)
# ax.set_ylim(0, 20)
# ax.set_ylabel("H3K4me3\nsignal", rotation = 0, ha = "right", va = "center")

# panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 0.5)))
# # # !wget https://www.encodeproject.org/files/ENCFF242ENK/@@download/ENCFF242ENK.bigWig
# import pyBigWig
# file = pyBigWig.open("ENCFF242ENK.bigWig")
# plotdata = pd.DataFrame({"value":file.values(region["chrom"], region["start"], region["end"])})
# plotdata["position"] = np.arange(*fragments.regions.window)[::int(region["strand"])]
# ax.plot(plotdata["position"], plotdata["value"], color = "#333")
# ax.set_xlim(*window)
# ax.set_ylim(0, 20)
# ax.set_ylabel("H3K27me\nsignal", rotation = 0, ha = "right", va = "center")

fig.plot()

# %%
cummean = np.cumsum(~pd.isnull(joined_all.sort_values("deltacor")["HS_LS_logratio"])) / np.arange(1, len(joined_all) + 1)
fig, ax = plt.subplots()
plotdata = pd.DataFrame({
    "deltacor":joined_all.sort_values("deltacor")["deltacor"].values,
    "perc_crispr":cummean.values,
}).iloc[::100]
plotdata = plotdata.loc[plotdata["deltacor"] < -1e-5]
ax.plot(plotdata["deltacor"], plotdata["perc_crispr"])
ax.set_xscale("symlog", linthresh = 1e-5)

# %%
deltacor_bins = [-0.01, -0.005, -0.002, -0.001, -0.0005, -0.0002, -0.0001]
joined["deltacor_bin"] = pd.cut(joined["deltacor"], deltacor_bins, labels = deltacor_bins[:-1])
joined["no_crispr"] = np.isnan(joined["HS_LS_logratio"])

fig, ax = plt.subplots()
plotdata = joined.groupby("deltacor_bin").mean()
ax.plot(plotdata.index.astype(str), plotdata["no_crispr"], color = "#333")

# %%
import scanpy as sc
symbols_oi2 = ["HDAC6"]
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(symbols_oi2), title = symbols_oi2, use_raw = False, layer = "normalized")

# %% [markdown]
# ## Link CRISPRi and Chd/peak-calling

# %%
import sklearn.metrics

# %%
genes_oi = transcriptome.var.loc[genes_oi].query("dispersions_norm > 0").index
transcriptome.symbol(genes_oi)

# %%
window_size = 50
# window_size = 100
# window_size = 500

# joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = window_size, regionpairwindow=regionpairwindow)
joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = window_size)
windows_oi = regionmultiwindow.design.query("window_size == @window_size").index
design = regionmultiwindow.design.loc[windows_oi]

# remove promoters
# joined_all.loc[(joined_all["window_mid"] > -1000) & (joined_all["window_mid"] < 1000), "HS_LS_logratio"] = np.nan

# lfc_cutoff = np.log(1.2)
lfc_cutoff = np.log(1.5)
# lfc_cutoff = np.log(2.0)
joined_all["significant_expression"] = joined_all["HS_LS_logratio"].abs() > lfc_cutoff

joined_genes = {k:v for k, v in joined_all.groupby("gene")}

# %%
import sklearn.metrics
import chromatinhd.data.peakcounts
import scipy.stats

# %%
joined_all.dropna()["significant_expression"].mean()

# %%
genescores = {}
slicescores = {}
allslicescores = {}

# %%
# ws = [0, 1, 2, 3, 5, 7, 10 ,20,25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
ws = [0, 1, 5, 10, 20]

# %%
deltacor_cutoff = -0.001

method_names = [
    "accessibility",
    "v33",
]
for method_name in method_names:
    print(method_name)
    genescores[method_name] = []
    slicescores[method_name] = []
    allslicescores[method_name] = []
    for gene_oi in tqdm.tqdm(genes_oi):
        for w in ws:
            joined_oi = joined_genes[gene_oi]
            joined_oi["significant_cor"] = joined_oi["deltacor"] < deltacor_cutoff

            joined_oi["score"] = -joined_oi["deltacor"]
            if method_name in ["v30", "v33", "v34"]:
                joined_oi["score"] = crispri.rolling_max(-joined_oi["deltacor_positive"].values, w)
            if method_name.endswith("normalized"):
                joined_oi["score"] = crispri.rolling_max(-joined_oi["deltacor_positive"].values / (joined_oi["lost"]+0.00001), w)
            if method_name.endswith("positive"):
                joined_oi["score"] = crispri.rolling_max(-np.where(joined_oi["effect"] > 0, joined_oi["deltacor"].values, 0), w)
            elif method_name.endswith("negative"):
                joined_oi["score"] = crispri.rolling_max(-np.where(joined_oi["effect"] < 0, joined_oi["deltacor"].values, 0), w)
            elif method_name.endswith("pure_effect"):
                joined_oi["score"] = crispri.rolling_max(joined_oi["effect"].values, w)
            elif method_name.endswith("interaction_effect"):
                joined_oi["score"] = crispri.rolling_max(joined_oi["effect"].values * (joined_oi["interaction"].values), w)
            elif method_name.endswith("effect"):
                joined_oi["score"] = crispri.rolling_max(joined_oi["effect"].values * (-joined_oi["deltacor"].values), w)
            elif method_name.endswith("interaction"):
                joined_oi["score"] = crispri.rolling_max(joined_oi["interaction_abs"], w)
            elif method_name.endswith("interaction_abs"):
                joined_oi["score"] = crispri.rolling_max(joined_oi["effect"].values * (joined_oi["interaction_abs"].values), w)
            elif method_name.endswith("deltacor2"):
                joined_oi["score"] = crispri.rolling_max(joined_oi["effect"].values * (-joined_oi["deltacor2"].values), w)
            elif method_name == "accessibility":
                joined_oi["score"] = crispri.rolling_max(joined_oi["lost"].values, w)
            # else:
            #     raise ValueError("Unknown method: {}".format(method_name))

            joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
            
            aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
            auroc = sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], joined_oi["score"])
            
            joined_oi["score_untied"] = joined_oi["score"] + np.random.normal(scale = 1e-5, size = len(joined_oi))
            cor = np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0, 1]
            spearman = scipy.stats.spearmanr(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0]

            confirmed = (joined_oi["significant_expression"] & joined_oi["significant_cor"]).sum() / joined_oi["significant_cor"].sum()

            joined_oi["score_rank"] = (joined_oi["score"] + np.random.normal(0., 1e-5, len(joined_oi))).rank(ascending = False) / joined_oi.shape[0]

            allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, w = w))

            joined_chd = joined_oi.loc[joined_oi["significant_cor"]]

            genescores[method_name].append({
                "gene": gene_oi,
                "method": method_name,
                "lfc_mean": joined_chd["HS_LS_logratio"].abs().mean(),
                "n": joined_chd.shape[0],
                "aupr":aupr,
                "auroc":auroc,
                "cor":cor,
                "spearman":spearman,
                "confirmed":confirmed,
                "w":w,
            })

            slicescores[method_name].append(pd.DataFrame({
                "gene": gene_oi,
                "method": method_name,
                "lfc": joined_chd["HS_LS_logratio"].abs(),
                "score": joined_chd["score"],
                "slice":joined_chd.index,
                "w":w,
            }))

# %%
gene_oi = transcriptome.gene_id("GATA1")
joined_oi = joined_genes[gene_oi]
joined_oi["significant_cor"] = joined_oi["deltacor"] < deltacor_cutoff

joined_oi["score"] = -joined_oi["deltacor"]
joined_oi["score_original"] = joined_oi["score"]
joined_oi["score"] = crispri.rolling_max(-joined_oi["deltacor"].values, 10)
joined_oi_all = joined_oi.copy()

joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()

fig, ax = plt.subplots(figsize = (10, 2))
ax.plot(joined_oi_all["window_mid"], np.abs(joined_oi_all["HS_LS_logratio"]))
ax.axhline(lfc_cutoff, color = "blue", linestyle = "--", lw = 1)

ax2 = ax.twinx()
ax2.plot(joined_oi_all["window_mid"], -joined_oi_all["deltacor"], color = "green")
ax2.plot(joined_oi_all["window_mid"], joined_oi_all["score"], color = "red")

# ax.set_xlim([-10000, 20000])
ax.set_xlim([-5000, 5000])

# %%
peakcallers = [
    "encode_screen",
    "macs2_leiden_0.1_merged",
    "macs2_summit",
    "macs2_summits",
    "macs2_improved",
    "cellranger",
    "rolling_500",
    "rolling_100",
    "rolling_50",
]

# %%
gene = transcriptome.var.loc[gene_oi]

# %%
import statsmodels as sm
for peakcaller in peakcallers:
    peakcounts = chd.flow.Flow.from_path(
        path=chromatinhd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / "100k100k"
        # path=chromatinhd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / "500k500k"
    )

    transcriptome.var["ix"] = np.arange(transcriptome.var.shape[0])

    for method_name in [peakcaller + "/linear"]:
    # for method_name in [peakcaller + "/linear", peakcaller + "/linear_test"]:
        genescores[method_name] = []
        slicescores[method_name] = []
        allslicescores[method_name] = []

        for gene_oi in tqdm.tqdm(genes_oi):
            gene = transcriptome.var.loc[gene_oi]

            peak_gene_links_oi, x = peakcounts.get_peak_counts(gene_oi, densify = False)
            peak_gene_links_oi = chd.data.peakcounts.peakcounts.center_peaks(peak_gene_links_oi, fragments.regions.coordinates.loc[gene_oi])
            y = transcriptome.layers["magic"][:, transcriptome.var.index.get_loc(gene_oi)]

            x = x.T
            y = y[:, None].T

            cors = np.corrcoef(x, y)[:-1, -1]
            cors[np.isnan(cors)] = 0

            if method_name.endswith("test"):
                n = x.shape[1]
                t = (cors * np.sqrt(n-2))/np.sqrt(1-cors**2)
                p = scipy.stats.t.sf(np.abs(t), n-2)*2
                q = chd.utils.fdr(p)
                cors[q > 0.05] = 0
            elif method_name.endswith("cutoff"):
                cors[np.abs(cors) < 0.05] = 0.05
            peakscores = cors
            
            peak_gene_links_oi["cor"] = cors

            joined = joined_genes[gene_oi]

            cor = np.zeros(len(joined))
            joined["cor"] = 0.
            for peak_id, peak_gene_link in peak_gene_links_oi.iterrows():
                cor[np.arange(np.searchsorted(design["window_start"], peak_gene_link.start), np.searchsorted(design["window_start"], peak_gene_link.end))] = peak_gene_link["cor"]
            joined["cor"] = cor

            for w in ws:
                joined_oi = joined.copy()

                if method_name.endswith("linear_absolute"):
                    joined_oi["score"] = crispri.rolling_max(np.abs(joined_oi["cor"].abs().values), w)
                elif method_name.endswith("linear_positive"):
                    joined_oi["score"] = crispri.rolling_max(np.where(joined_oi["cor"] > 0, joined_oi["cor"].values, 0), w)
                else:
                    joined_oi["score"] = crispri.rolling_max(joined_oi["cor"].values, w)

                joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()

                aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
                auroc = sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], joined_oi["score"])

                joined_oi["score_untied"] = joined_oi["score"] + np.random.normal(scale = 1e-3, size = len(joined_oi))
                cor = np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0, 1]
                spearman = scipy.stats.spearmanr(joined_oi["HS_LS_logratio"].abs(), joined_oi["score_untied"])[0]

                joined_oi["score_rank"] = (joined_oi["score"] + np.random.normal(0., 1e-5, len(joined_oi))).rank(ascending = False) / joined_oi.shape[0]

                allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, w = w))

                joined_cre = joined_oi.loc[joined_oi["significant_cor"]]
                
                genescores[method_name].append({
                    "gene": gene_oi,
                    "method": method_name,
                    "lfc_mean": joined_cre["HS_LS_logratio"].abs().mean(),
                    "n": joined_cre.shape[0],
                    "aupr":aupr,
                    "auroc":auroc,
                    "cor":cor,
                    "spearman":spearman,
                    "w":w,
                })

                slicescores[method_name].append(pd.DataFrame({
                    "gene": gene_oi,
                    "method": method_name,
                    "lfc": joined_cre["HS_LS_logratio"].abs(),
                    "slice":joined_cre.index,
                    "score": joined_cre["cor"].abs(),
                    "w":w,
                }))

# %%
method_name = "all"
genescores[method_name] = []
slicescores[method_name] = []
allslicescores[method_name] = []
for gene_oi in tqdm.tqdm(genes_oi):
    joined_oi = joined_genes[gene_oi]
    joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
    
    aupr = []
    auroc = []
    for i in range(100):
        aupr.append(sklearn.metrics.average_precision_score(joined_oi["significant_expression"], np.random.rand(joined_oi.shape[0])))
        auroc.append(sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], np.random.rand(joined_oi.shape[0])))
    aupr = np.mean(aupr)
    auroc = np.mean(auroc)

    joined_oi["score"] = np.random.rand(joined_oi.shape[0])
    joined_oi["score_rank"] = joined_oi["score"].rank(ascending = False) / joined_oi.shape[0]

    genescores[method_name].append({
        "gene": gene_oi,
        "method": method_name,
        "lfc_mean": joined_oi["HS_LS_logratio"].abs().mean(),
        "n": joined_chd.shape[0],
        "aupr":aupr,
        "auroc":auroc,
        "cor":np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score"] + np.random.normal(0, 1e-5, size = len(joined_oi)))[0, 1],
    })
    slicescores[method_name].append(pd.DataFrame({
        "gene": gene_oi,
        "method": method_name,
        "lfc": joined_oi["HS_LS_logratio"].abs(),
        "score":0.,
        "slice":joined_oi.index,
    }))

    allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, score = np.random.rand(joined_oi.shape[0]), w = 0))

# %% [markdown]
# ## Compare quantitatively

# %% [markdown]
# ### All genes together

# %%
allslicescores_stacked = pd.concat([pd.concat(allslicescores[method_name]) for method_name in allslicescores.keys()], ignore_index=True)
slicescores_stacked = pd.concat([pd.concat(slicescores[method_name]) for method_name in slicescores.keys()], ignore_index=True)

# %%
w_oi = 10

# %%
fig = polyptich.grid.Figure(polyptich.grid.Wrap())

w_oi = 10.
for method_name, slicescores_oi in allslicescores_stacked.query("w == @w_oi").groupby("method"):
    panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))

    # x = slicescores_oi["score"].abs()
    x = np.clip(slicescores_oi["score"].abs(), 0, np.inf)
    # if method_name == "rolling_500/linear":
        # x = np.clip(slicescores_oi["score"], 0.1, np.inf)
    # x = -slicescores_oi["score_rank"]
    # x = scipy.stats.rankdata(slicescores_oi["score"] + np.random.normal(scale = 1e-5, size = len(slicescores_oi)))
    y = -slicescores_oi["HS_LS_logratio"]
    # y = scipy.stats.rankdata(slicescores_oi["HS_LS_logratio"])

    cor = np.corrcoef(x, y)[0, 1]
    r2 = cor**2
    ax.annotate(f"r2 = {r2*100:.0f}, cor = {cor*100:.0f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top")
    ax.scatter(x, y, s = 1)
    ax.set_title(method_name)

    # loess
    # import statsmodels.api as sm
    # lowess = sm.nonparametric.lowess
    # z = lowess(y, x, frac=0.1)
    # ax.plot(z[:,0], z[:,1], color = "red")

    # lm
    import statsmodels.formula.api as smf
    model = smf.ols(formula='y ~ x', data=slicescores_oi)
    results = model.fit()
    ax.plot(x, results.predict(), color = "green")

fig.plot()

# %%
eps = 1e-5

methodscores = []
for (method_name, w), slicescores_oi in allslicescores_stacked.groupby(["method", "w"]):
    aupr = sklearn.metrics.average_precision_score(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    auroc = sklearn.metrics.roc_auc_score(slicescores_oi["significant_expression"], slicescores_oi["score"])
    aupr = sklearn.metrics.average_precision_score(slicescores_oi["significant_expression"], -slicescores_oi["score_rank"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    auroc = sklearn.metrics.roc_auc_score(slicescores_oi["significant_expression"], -slicescores_oi["score_rank"])

    x = np.clip(slicescores_oi["score"], 0, np.inf)

    methodscores.append({
        "method": method_name,
        "aupr": aupr,
        "auroc": auroc,
        "cor": np.corrcoef(-slicescores_oi["HS_LS_logratio"], x)[0, 1],
        "w":w
    })
methodscores = pd.DataFrame(methodscores)
methodscores["r2"] = methodscores["cor"]**2

# %%
w_oi = 10
methodscores.loc[(methodscores["method"] == "all") | (methodscores["w"] == w_oi)].style.bar()

# %%
prediction_methods = chdm.methods.prediction_methods
prediction_methods.loc["all"] = {
    "color":"grey",
    "label":"Random"
}
prediction_methods.loc["accessibility"] = {
    "color":"#333",
    "label":"Accessibility",
    "type":"baseline",
}
for method_name in ["v30", "v30/positive", "v30/negative", "v30/effect", "v30/interaction", "v30/interaction_abs", "v30/deltacor2", "v30/interaction_effect", "v30/pure_effect", "v30/normalized", "v33", "v34"]:
    prediction_methods.loc[method_name] = {
        "color":"#0074D9",
        "label":"ChromatinHD " + (method_name.split("/")[1] if "/" in method_name else ""),
        "type":"ours",
    }
for peakcaller in peakcallers:
    prediction_methods.loc[peakcaller + "/linear"] = prediction_methods.loc[peakcaller + "/lasso"].copy()
    prediction_methods.loc[peakcaller + "/linear_absolute"] = prediction_methods.loc[peakcaller + "/linear"].copy()
    prediction_methods.loc[peakcaller + "/linear_absolute", "label"] += " abs"
    prediction_methods.loc[peakcaller + "/linear_positive"] = prediction_methods.loc[peakcaller + "/linear"].copy()
    prediction_methods.loc[peakcaller + "/linear_positive", "label"] += " pos"
    prediction_methods.loc[peakcaller + "/linear_test"] = prediction_methods.loc[peakcaller + "/linear"].copy()
    prediction_methods.loc[peakcaller + "/linear_test", "label"] += " test"

# %%
w_oi = 10

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_width = 0.2))

plotdata = methodscores.loc[(methodscores["w"] == w_oi) | (methodscores["method"] == "all")].copy()
methods = prediction_methods.loc[methodscores["method"].unique()]

panel, ax = fig.main.add_under(polyptich.grid.Panel((2, len(methods)*0.3)))
color = prediction_methods.reindex(plotdata["method"])["color"]
ax.barh(plotdata["method"], plotdata["aupr"], color = color)
ax.set_xlim(0., 1)
ax.set_xlabel("AUPRC")
ax.set_yticks(np.arange(len(plotdata["method"])))
ax.set_yticklabels(prediction_methods.reindex(plotdata["method"])["label"])

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["auroc"], color = color)
ax.set_xlim(0.5, 1)
ax.set_yticks([])
ax.set_xlabel("AUROC")
ax.set_title("Fulco et al. 2019")

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["cor"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("cor")

panel, ax = fig.main.add_right(polyptich.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["r2"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("r2")

fig.plot()

# %%
fig, (ax_aupr, ax_auroc) = plt.subplots(1, 2, figsize = (5, 2))

eps = 1e-5

allslicescores_oi = allslicescores_stacked.loc[(allslicescores_stacked["w"] == w_oi) | (allslicescores_stacked["method"] == "all")]

for (method_name, w), slicescores_oi in allslicescores_oi.groupby(["method", "w"]):
    slicescores_oi["significant_expression"] = slicescores_oi["HS_LS_logratio"].abs() > lfc_cutoff

    curve = sklearn.metrics.precision_recall_curve(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    ax_aupr.plot(curve[1], curve[0], label = method_name, color = prediction_methods.loc[method_name, "color"])

    curve = sklearn.metrics.roc_curve(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    ax_auroc.plot(curve[0], curve[1], label = method_name, color = prediction_methods.loc[method_name, "color"])
    ax_auroc.plot([0, 1], [0, 1], color = "black", linestyle = "--")
ax_auroc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %% [markdown]
# ### Per gene

# %%
genescores_stacked = pd.concat([pd.DataFrame(genescore) for genescore in genescores.values()], ignore_index=True)

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (10, 2))
genescores_oi = genescores_stacked
for method_name, plotdata in genescores_oi.groupby("method"):
    plotdata = plotdata.groupby("w").mean(numeric_only = True).reset_index()
    ax0.plot(plotdata["w"], plotdata["aupr"], label = method_name, color = prediction_methods.loc[method_name, "color"])
    ax1.plot(plotdata["w"], plotdata["auroc"], label = method_name, color = prediction_methods.loc[method_name, "color"])
    ax2.plot(plotdata["w"], plotdata["cor"], label = method_name, color = prediction_methods.loc[method_name, "color"])
    if len(plotdata) > 0:
        print(method_name, plotdata.loc[plotdata["spearman"].idxmax(), "w"])
ax0.set_xlabel("w")
ax1.set_xlabel("w")

# %%
fig = polyptich.grid.Figure(polyptich.grid.Wrap())

for method, plotdata in allslicescores_stacked.query("w == @w_oi").groupby("method"):
    # if method.startswith("rolling"):
    #     plotdata["score"] = np.clip(plotdata["score"], 0.05, np.inf)
    plotdata["HS_LS_logratio2"] = plotdata["HS_LS_logratio"]
    plotdata["score"] = plotdata["score"].abs()
    plotdata["HS_LS_logratio2"] = plotdata.groupby("score")["HS_LS_logratio"].transform("mean")
    panel, ax = fig.main.add(polyptich.grid.Panel((2, 2)))
    cors = []
    auprs = []
    aurocs = []
    for gene, plotdata in plotdata.groupby("gene"):
        plotdata["score"] = (plotdata["score"] -plotdata["score"].min()) / (plotdata["score"].max() - plotdata["score"].min())
        plotdata = plotdata.groupby("score").mean(numeric_only = True).reset_index()
        plotdata["HS_LS_logratio"] = plotdata["HS_LS_logratio"] / plotdata["HS_LS_logratio"].abs().max()
        plotdata["HS_LS_logratio2"] = plotdata["HS_LS_logratio2"] / plotdata["HS_LS_logratio2"].abs().max()
        ax.set_title(method)
        cors.append(np.corrcoef(-plotdata["HS_LS_logratio"], plotdata["score"] + np.random.normal(scale = 1e-5, size = len(plotdata)))[0, 1])

        ax.scatter(-plotdata["HS_LS_logratio"], plotdata["score"], s = 1)
        # spearmans.append(scipy.stats.spearmanr(-plotdata["HS_LS_logratio2"], plotdata["score"])[0])
    ax.annotate(f"cor = {(sum(cors)/len(cors)):.2f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top")
    ax.set_xlabel("CRISPRi fold enrichment")
    ax.set_ylabel("Score")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)

fig.plot()

# %%
sns.heatmap(genescores_stacked.groupby(["w", "method"])["cor"].mean().unstack().T, vmin = 0)

# %%
gene_order = transcriptome.var["dispersions_norm"][genes_oi].sort_values(ascending=False).index

w_oi = 10

plotdata = (
    genescores_stacked.loc[(genescores_stacked["w"] == w_oi) | (genescores_stacked["method"] == "all")]
    .groupby(["method", "gene"])
    .mean(numeric_only=True)["cor"]
    .unstack()[gene_order]
)
fig, ax = plt.subplots(figsize=np.array(plotdata.shape[::-1]) * 0.2)
norm = mpl.colors.Normalize(vmin=0)
cmap = mpl.colormaps["rocket_r"]

matshow = ax.matshow(plotdata, vmin=0, cmap=cmap)
ax.set_yticks(np.arange(len(plotdata.index)))
ax.set_yticklabels(prediction_methods.reindex(plotdata.index)["label"])
ax.set_xticks(np.arange(len(plotdata.columns)))
ax.set_xticklabels(transcriptome.symbol(plotdata.columns), rotation=90)
fig.colorbar(matshow, ax=ax, label="AUPRC")

# %%
plotdata = (
    genescores_stacked.loc[(genescores_stacked["w"] == w_oi)]
    .groupby(["method", "gene"])
    .mean(numeric_only=True)
).reset_index()
plotdata["cor"] = np.clip(plotdata["cor"], -0.05, 1)

plotdata_mean = plotdata.groupby("method").mean(numeric_only = True).reset_index()
plotdata_mean["gene"] = "mean"

plotdata = pd.concat([
    plotdata,
    plotdata_mean
])

# %%
plotdata_groupby = plotdata.reset_index().groupby("gene")

# %%
y_info = pd.DataFrame({"gene":list(genes_oi) + ["mean"], "label":transcriptome.symbol(genes_oi).tolist() + ["Mean"]}).set_index("gene")
y_info["ix"] = np.arange(len(y_info))

# %%
method_info = prediction_methods

# %%
import matplotlib as mpl
import seaborn as sns
import chromatinhd.plot.quasirandom

fig, ax = plt.subplots(figsize=(4, len(plotdata_groupby) * 0.25))

score = "cor"
# score = "odds"
# score = "r2"
# score = "scored"
# score = "cor"

for dataset, plotdata_dataset in plotdata_groupby:
    x = y_info.loc[dataset]["ix"]

    plotdata_dataset[score] = plotdata_dataset[score].fillna(0.)
    y = np.array(chd.plot.quasirandom.offsetr(np.array(plotdata_dataset[score].values.tolist()), adjust=0.1)) * 0.8 + x
    plotdata_dataset["y"] = y

    plotdata_dataset["color"] = method_info.loc[
        plotdata_dataset["method"], "color"
    ].values
    plotdata_dataset.loc[plotdata_dataset["method"] == "v32", "color"] = "pink"
    plotdata_dataset.loc[plotdata_dataset["method"] == "v31", "color"] = "turquoise"

    ax.axhspan(x - 0.5, x + 0.5, color="#33333308" if dataset != "mean" else "#33333311", ec="white", zorder = -2)
    ax.scatter(plotdata_dataset[score], plotdata_dataset["y"], s=5, color=plotdata_dataset["color"], lw=0)

    plotdata_dataset["type"] = method_info.loc[
        plotdata_dataset["method"], "type"
    ].values
    plotdata_top = plotdata_dataset.sort_values(score, ascending=False).groupby("type").first()
    for i, (type, plotdata_top_) in enumerate(plotdata_top.groupby("type")):
        ax.plot(
            [plotdata_top_[score]] * 2,
            [x - 0.45, x + 0.45],
            color=plotdata_top_["color"].values[0],
            lw=2,
            zorder=0,
            alpha=0.9,
            solid_capstyle = "butt",
        )
    plotdata_ours = plotdata_dataset.loc[plotdata_dataset["method"] == "v33"].iloc[0]
    plotdata_top_others = plotdata_top.loc[plotdata_top.index != "ours"]

    try:
        plotdata_others_max = plotdata_top_others.loc[plotdata_top_others[score].idxmax()]
        rectangle = mpl.patches.Rectangle(
            (plotdata_others_max[score], x-0.45),
            plotdata_ours[score] - plotdata_others_max[score],
            0.9,
            fc=plotdata_ours["color"],
            ec="none",
            zorder=-1,
            alpha=1/3,
        )
        ax.add_patch(rectangle)
    except ValueError:
        pass
ax.set_yticks(y_info["ix"])
ax.set_yticklabels(y_info["label"])
for ticklabel in ax.get_yticklabels():
    if ticklabel.get_text() == "Mean":
        ticklabel.set_fontweight("bold")
    else:
        ticklabel.set_fontsize(9)
ax.set_xlabel("Pearson $r$ between smooth region importance \nand CRISPRi fold enrichment")
ax.set_xlim(-0.07)
sns.despine(ax=ax, left=True, bottom=True)
ax.set_ylim(-0.5, y_info["ix"].max() + 0.5)

manuscript.save_figure(fig, "2", "aggregate_crispri")

# %%
y_info["color"] = sns.color_palette("Dark2", n_colors=len(y_info)).as_hex()

# %%
fig = polyptich.grid.Figure(polyptich.grid.Wrap(ncol = 2, padding_height = 0.15))

methods_oi = ["cellranger/linear", "encode_screen/linear", "rolling_500/linear", "v33"]

for method in methods_oi:
    plotdata = allslicescores_stacked.query("w == @w_oi").query("method == @method")
    plotdata["HS_LS_logratio2"] = plotdata["HS_LS_logratio"]
    plotdata["score"] = plotdata["score"].abs()
    plotdata["HS_LS_logratio2"] = plotdata.groupby("score")["HS_LS_logratio"].transform("mean")
    panel, ax = fig.main.add(polyptich.grid.Panel((1.3, 1.3)))
    cors = []
    auprs = []
    aurocs = []
    for gene, plotdata in plotdata.groupby("gene"):
        plotdata["score"] = (plotdata["score"] -plotdata["score"].min()) / (plotdata["score"].max() - plotdata["score"].min())
        cors.append(np.corrcoef(-plotdata["HS_LS_logratio"], plotdata["score"] + np.random.normal(scale = 1e-5, size = len(plotdata)))[0, 1])
        plotdata = plotdata.groupby("score").mean(numeric_only = True).reset_index()
        plotdata["HS_LS_logratio"] = plotdata["HS_LS_logratio"] / plotdata["HS_LS_logratio"].abs().max()
        plotdata["HS_LS_logratio2"] = plotdata["HS_LS_logratio2"] / plotdata["HS_LS_logratio2"].abs().max()

        ax.scatter(-plotdata["HS_LS_logratio"], plotdata["score"], s = 1, color = y_info.loc[gene, "color"])
        # spearmans.append(scipy.stats.spearmanr(-plotdata["HS_LS_logratio2"], plotdata["score"])[0])
    ax.set_title(methods.loc[method, "label"])
    ax.annotate(f"$r$ = {(sum(cors)/len(cors)):.2f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top", bbox = {"facecolor":"white", "alpha":0.5})
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "max"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "max"])

panel, ax = fig.main[2]
ax.set_xlabel("CRISPRi fold enrichment\n(0-max scaled per gene)")
ax.set_ylabel("Locus importance\n(0-max scaled per gene)")

fig.plot()

manuscript.save_figure(fig, "2", "aggregate_crispri_per_method")

# %%
w_oi = genescores_stacked.query("method == 'v33'").groupby(["w"])["cor"].mean().idxmax()
w_oi

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_width = 0.2))

plotdata = genescores_stacked.loc[(genescores_stacked["w"] == w_oi) | (genescores_stacked["method"] == "all")].groupby("method").mean(numeric_only = True).reset_index()
methods = prediction_methods.loc[plotdata["method"].unique()]

plotdata["color"] = prediction_methods.reindex(plotdata["method"])["color"].values

panel, ax = fig.main.add_right(polyptich.grid.Panel((1.2, len(methods)*0.2)))
ax.barh(plotdata["method"], plotdata["cor"]**2, color = plotdata["color"], height = 0.9, lw = 0)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("R2 between\nprediction and CRISPRi")
ax.set_yticks(np.arange(len(plotdata["method"])))
ax.set_yticklabels(prediction_methods.reindex(plotdata["method"])["label"])

panel, ax = fig.main.add_right(polyptich.grid.Panel((1.2, len(methods)*0.2)))
ax.barh(plotdata["method"], plotdata["cor"], color = plotdata["color"], height = 0.9, lw = 0)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("Correlation between\nprediction and CRISPRi")
ax.set_yticks([])

# panel, ax = fig.main.add_right(polyptich.grid.Panel((2, len(methods)*0.3)))
# ax.barh(plotdata["method"], plotdata["aupr"], color = plotdata["color"])
# ax.set_xlim(0., 1)
# ax.set_xlabel("AUPRC")
# ax.set_yticks([])

# panel, ax = fig.main.add_right(polyptich.grid.Panel((2, len(methods)*0.3)))
# ax.barh(plotdata["method"], plotdata["auroc"], color = plotdata["color"])
# ax.set_xlim(0.5, 1)
# ax.set_yticks([])
# ax.set_xlabel("AUROC")
# ax.set_title("Fulco et al. 2019")

fig.plot()

# %%
plotdata_mean = genescores_stacked.loc[(genescores_stacked["w"] == w_oi) | (genescores_stacked["method"] == "all")].groupby("method").mean(numeric_only = True).reset_index()
plotdata_mean["peakcaller"] = [method.split("/")[0] if method != "v33" else "chd" for method in plotdata_mean["method"]]
plotdata_mean.to_csv(chd.get_output() / "aggregate_crispr.csv")

# %% [markdown]
# ## Find examples

# %%
w_oi = 10.

# %%
# m1 = "v34"
m1 = "v33"
m2 = "cellranger/linear"

position_scores = allslicescores_stacked.query("(w == @w_oi)").set_index(["gene", "start", "method"])["score"].unstack()
position_scores["lfc"] = -allslicescores_stacked.query("(w == @w_oi)").set_index(["gene", "start", "method"])["HS_LS_logratio"].xs(m1, level = "method")
position_scores["significant_expression"] = allslicescores_stacked.query("(w == 10)").set_index(["gene", "start", "method"])["significant_expression"].xs(m1, level = "method")

# position_scores = position_scores.loc[position_scores["rolling_500/linear"] > 0.01]

cmap = mpl.cm.Blues
norm = mpl.colors.Normalize(vmin=-1, vmax=2)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize = (10, 2.5))
ax0.scatter(
    position_scores[m1],
    position_scores[m2],
    c = cmap(norm(position_scores["significant_expression"]))
)
ax0.set_xlabel(m1)
ax0.set_ylabel(m2)
ax1.scatter(
    position_scores[m1],
    position_scores["lfc"],
    c = cmap(norm(position_scores["significant_expression"]))
)
ax1.set_xlabel(m1)
ax1.set_ylabel("lfc")
ax2.scatter(
    position_scores[m2],
    position_scores["lfc"],
    c = cmap(norm(position_scores["significant_expression"]))
)
ax2.set_xlabel(m2)
ax2.set_ylabel("lfc")

# select false-positives
idx_oi = (position_scores.loc[~position_scores["significant_expression"], m1]/position_scores[m1].std() - position_scores.loc[~position_scores["significant_expression"], m2]/position_scores[m2].std()).sort_values()
position_scores["oi"] = idx_oi < -1.

# select false-negatives
idx_oi = (position_scores.loc[position_scores["significant_expression"], m1]/position_scores[m1].std() - position_scores.loc[position_scores["significant_expression"], m2]/position_scores[m2].std()).sort_values()
position_scores["oi"] = idx_oi > 1.

position_scores.loc[pd.isnull(position_scores["oi"]), "oi"] = False
ax3.scatter(
    position_scores[m1],
    position_scores[m2],
    c = cmap(norm(position_scores["oi"].astype(int)))
)
# ax0.set_xscale("log")

print(np.corrcoef(position_scores["lfc"].abs(), position_scores[m1])[0, 1], np.corrcoef(position_scores["lfc"].abs(), position_scores[m2])[0, 1])

# %%
idx_oi.loc[transcriptome.gene_id("CALR")].sort_values().tail(30)

# %% [markdown]
# ## Plot example

# %%
# idx_oi = (position_scores.loc[~position_scores["significant_expression"], m1]/position_scores[m1].std() - position_scores.loc[~position_scores["significant_expression"], m2]/position_scores[m2].std()).sort_values()
# position_scores["oi"] = idx_oi < -1.

# idx_oi = (position_scores.loc[position_scores["significant_expression"], m1]/position_scores[m1].std() - position_scores.loc[position_scores["significant_expression"], m2]/position_scores[m2].std()).sort_values(ascending = False)
# position_scores["oi"] = idx_oi > 1.

# gene_oi, locus = idx_oi.index[10]
# gene_oi, locus

# Any locus
# window = [locus - 2000, locus + 2000]
# arrows = [{"position": locus, "orientation": "right", "y": 0.5}]

# GATA1 locus
gene_oi, locus, arrows, window = transcriptome.gene_id("GATA1"), 13916, [{"position": 14000, "orientation": "right", "y": 0.5}], [13916-2000, 13916+2000]

# KLF1 promoter locus
# gene_oi, locus, arrows, window = transcriptome.gene_id("KLF1"), 800, [
#     {"position": 1000, "orientation": "left", "y": 0.5},
#     {"position": -650, "orientation": "right", "y": 0.5},
# ], [800-2000, 800+2000]

# %%
arrows = []

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

joined = joined_all.query("gene == @gene_oi").copy()
joined["score"] = joined["HS_LS_logratio"] * joined["deltacor"]

# %%
joined = joined_genes[gene_oi]
peak_gene_links_oi, x = peakcounts.get_peak_counts(gene_oi)

def calculate_peak_difference(joined, peak_gene_links, x):
    peak_gene_links_oi = chd.data.peakcounts.peakcounts.center_peaks(peak_gene_links, fragments.regions.coordinates.loc[gene_oi])
    y = transcriptome.layers["magic"][:, transcriptome.var.index.get_loc(gene_oi)]

    x = x.T
    y = y[:, None].T

    cors = np.corrcoef(x, y)[:-1, -1]
    cors[np.isnan(cors)] = 0
    peakscores = cors

    peak_gene_links_oi["cor"] = cors

    cor = np.zeros(len(joined))
    joined["cor"] = 0.
    for peak_id, peak_gene_link in peak_gene_links_oi.iterrows():
        cor[np.arange(np.searchsorted(design["window_start"], peak_gene_link.start), np.searchsorted(design["window_start"], peak_gene_link.end))] = peak_gene_link["cor"]
    joined["cor"] = cor

    joined["chd_score"] = -joined["deltacor_positive"]
    joined["peak_score"] = np.abs(joined["cor"])# * joined["lost"]

    joined["peak_score_norm"] = joined["peak_score"] / joined["peak_score"].mean()
    joined["chd_score_norm"] = joined["chd_score"] / joined["chd_score"].mean()

    joined["peak_score_norm"] = (joined["peak_score"] - joined["peak_score"].mean()) / joined["peak_score"].std()
    joined["chd_score_norm"] = (joined["chd_score"] - joined["chd_score"].mean()) / joined["chd_score"].std()

    return joined

    # joined["peak_score_norm"] = scipy.stats.rankdata(joined["peak_score"]) / joined.shape[0]
    # joined["chd_score_norm"] = scipy.stats.rankdata(joined["chd_score"]) / joined.shape[0]

def fill_between_gradient(x, y1, y2, y, ax, cmap, norm, **kwargs):
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y1, pd.Series):
        y = y.values
    (background,) = ax.plot(
        x,
        y1,
        color="black",
        lw=1,
        zorder=1,
        linestyle="dashed",
    )
    (differential,) = ax.plot(
        x,
        y2,
        color="black",
        lw=1,
        zorder=1,
    )
    polygon = ax.fill_between(
        x,
        y1,
        y2,
        color="black",
        zorder=0,
    )

    # up/down gradient
    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    c = y
    c[c == np.inf] = 0.0
    c[c == -np.inf] = -10.0
    gradient = ax.imshow(
        c.reshape(1, -1),
        cmap=cmap,
        aspect="auto",
        extent=[
            verts[:, 0].min(),
            verts[:, 0].max(),
            verts[:, 1].min(),
            verts[:, 1].max(),
        ],
        zorder=25,
        norm=norm,
    )
    gradient.set_clip_path(polygon.get_paths()[0], transform=ax.transData)
    polygon.set_alpha(0)


# %%
resolution = 2500

# %%
symbol = transcriptome.symbol(gene_oi)

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.1))

# binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]
binwidth = 50

width = (window[1] - window[0])/resolution

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width=width, window=window, label_genome = True, symbol = symbol))
ax.set_xlim(*window)

# Pileup
panel, ax = fig.main.add_under(
    chd.models.pred.plot.Pileup(regionmultiwindow.get_plotdata(gene_oi), window=window, width=width)
)
sns.despine(ax=ax)

# CRISPRI
panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 0.55)))
ax.bar(
    joined["window_mid"],
    joined["HS_LS_logratio"],
    lw=0,
    width=binwidth,
    color="#333",
)
ax.step(
    joined["window_mid"],
    joined["HS_LS_logratio"].fillna(0.0),
    lw=0.5,
    color="#AAA",
    where="mid",
    zorder=0,
    linestyle="--",
)
ax.set_xlim(*window)
ax.set_xticks([])
ax.set_ylabel(
    f"CRISPRi\nfold enrichment\nhigh vs low {transcriptome.symbol(gene_oi)}", rotation=0, ha="right", va="center"
)
ax.set_yticks(np.log([0.125, 0.25, 0.5, 1]))
ax.set_yticklabels(["⅛", "¼", "½", 1])
ax.set_ylim(np.log([1.5, 1/32]))
sns.despine(ax=ax)

panel, ax = fig.main.add_under(
    chd.models.pred.plot.Predictivity(regionmultiwindow.get_plotdata(gene_oi), window=window, width=width, limit=-0.1, label_y = "ChromatinHD\npredictivity", height = 0.4)
)
sns.despine(ax=ax)
ax.set_ylim(0, -0.1)
ax.set_yticks([0, -0.1])
for arrow in arrows:
    panel.add_arrow(**arrow)

# difference
for i, peakcaller in enumerate([
    "cellranger",
    "macs2_leiden_0.1_merged",
    "macs2_summits",
    "encode_screen",
    # "rolling_500",
]):
    panel, ax = fig.main.add_under(
        chdm.plotting.Peaks(
            region,
            chd.get_output() / "peaks" / dataset_name,
            window = window,
            width=width,
            peakcallers=[
                peakcaller,
            ],
            label_rows = chdm.peakcallers.loc[peakcaller, "label"],
            label_methods = False,
            lw = 0,
        ),
        padding = 0.1,
    )
    sns.despine(ax=ax, bottom = True, left = True)

    peakcounts = chd.flow.Flow.from_path(
        path=chromatinhd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / "100k100k"
    )
    joined = joined_genes[gene_oi].copy()
    peak_gene_links_oi, x = peakcounts.get_peak_counts(gene_oi)

    joined = calculate_peak_difference(joined, peak_gene_links_oi, x)

    panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 0.4)), padding = 0)

    norm = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
    cmap = mpl.cm.PiYG
    fill_between_gradient(
        joined["window_mid"],
        joined["peak_score_norm"],
        joined["chd_score_norm"],
        joined["peak_score_norm"] - joined["chd_score_norm"],
        ax,
        cmap,
        norm, 
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(*window)
    ax.set_ylabel("$\Delta$ z-score", rotation = 0, ha = "right", va = "center")
    sns.despine(ax=ax)

fig.plot()
manuscript.save_figure(fig, "2", f"crispri_example_{symbol}")

# %% [markdown]
# -----

# %%
import chromatinhd as chd
import pandas as pd

# %%
peaks_bed = chd.get_output() / "peaks" / "hspc" / "encode_screen" / "peaks.bed"
# peaks_bed = chd.get_output() / "peaks" / "hspc" / "macs2_summits" / "peaks.bed"

# %%
peaks_bed.exists()

# %%
peaks = pd.read_csv(peaks_bed, sep = "\t", header = None, names = ["chrom", "start", "end", "name", "score", "strand"])

# %%
region = ["chr11", 5234141.0, 5234981.0]

# %%
peaks.query("chrom == @region[0]").query("start > @region[1]").query("end < @region[2]")

# %%
peaks.loc[41241]

# %%
# peaks.loc[755726, "start"] = 48800794
# peaks.loc[344214, "start"] = 12887525
# peaks.loc[853235, "start"], peaks.loc[853235, "end"] = 181447051, 181447401
peaks.loc[41241, "start"], peaks.loc[41241, "end"] = 5234450+150, 5234650+150

# %%
peaks.to_csv(peaks_bed, sep = "\t", header = False, index = False)

# %%