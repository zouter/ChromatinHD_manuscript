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

# %% [markdown]
# ### Download CRISPRi data

# %%
file = chd.get_output() / "data" / "crispri" / "nasser_2021" / "41586_2021_3446_MOESM5_ESM.txt"
file.parent.mkdir(parents=True, exist_ok=True)

import requests
if not file.exists():
    response = requests.get("https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03446-x/MediaObjects/41586_2021_3446_MOESM5_ESM.txt")
    with open(file, "wb") as f:
        f.write(response.content)

# %%
import liftover
converter = liftover.get_lifter("hg19", "hg38")

# %%
data_raw = pd.read_table(file, comment = "#")
data_raw = data_raw.loc[~pd.isnull(data_raw["chr"])]

stacked = data_raw.groupby(["chr", "start", "end", "GuideSequenceMinusG", "Gene", "Bin"])["count"].mean().unstack().T
# stacked = stacked.loc[:, (stacked.sum(0) > 50)] # only keep bins with at least 50 reads

# %%
x = np.arange(len(stacked.index))
y = np.log1p(stacked / stacked.values.sum(1, keepdims = True) * 1e6)

# a simple linear regression between the bins
slope = ((x-x.mean())[:, None] * (y - y.mean(0))).sum(0) / ((x - x.mean())**2).sum()
intercept = y.mean(0) - slope * x.mean()
lfc = slope * x[-1]
lfc[np.isnan(lfc)] = 0

# %%
data_orig = pd.DataFrame({"HS_LS_logratio":lfc}, index = stacked.columns).reset_index()
data_orig = data_orig.loc[~data_orig["Gene"].isin(["CD83"])] # !!! these genes contain only a few guides

# %%
data = data_orig.copy()
data["start_orig"] = data["start"]
data["end_orig"] = data["end"]
data["chrom"] = data["chr"]

data["start"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["start_orig"])]
data["end"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["end_orig"])]
data["Significant"] = np.abs(data["HS_LS_logratio"]) > np.log(2.0)

# %%
data["Significant"].mean()

# %%
data_orig["Gene"].value_counts()

# %% [markdown]
# ### Load interpretation

# %%
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")
splitter = "5x5"
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / splitter)

symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)

data = data.loc[data["Gene"].isin(symbols_oi)].copy()
data["gene"] = transcriptome.gene_id(data["Gene"]).values

# %%
print(genes_oi.tolist())

# %%
import chromatinhd.models.pred.model.better
# model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / splitter / "magic" / "v30"
model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / splitter / "magic" / "v31"
models = chd.models.pred.model.better.Models(model_folder)

regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
    models.path / "scoring" / "crispri" / "fulco_2019" / "regionmultiwindow",
)

# %% [markdown]
# ### Check

# %%
joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 100)
# joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 100, regionpairwindow=regionpairwindow)
joined_observed = joined_all.loc[~pd.isnull(joined_all["HS_LS_logratio"])].copy()

# %%
deltacor_cutoff = -0.005
lfc_cutoff = np.log(2.0)

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
# ## Example

# %%
symbols_oi

# %%
# gene_oi = transcriptome.gene_id("CD40")
gene_oi = transcriptome.gene_id("ICOSLG")
# gene_oi = transcriptome.gene_id("PFKFB3")
# gene_oi = transcriptome.gene_id("IFNGR2")
# gene_oi = transcriptome.gene_id("BLK")
# gene_oi = transcriptome.gene_id("IL2RA")
# gene_oi = transcriptome.gene_id("ITGAL")
# gene_oi = transcriptome.gene_id("PPIF")
# gene_oi = transcriptome.gene_id("BLK")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

joined = joined_all.query("gene == @gene_oi").copy()
joined["score"] = joined["HS_LS_logratio"] * joined["deltacor"]

# %%
# regionmultiwindow2 = regionmultiwindow

# censorer = chd.models.pred.interpret.MultiWindowCensorer(
#     fragments.regions.window,
#     (100, ),
# )
# regionmultiwindow2 = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset = True)
# regionmultiwindow2.score(models, censorer, regions = [gene_oi])
# regionmultiwindow2.interpolate()

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.1))

# binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]
binwidth = 100

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

panel, ax = fig.main.add_under(chd.grid.Panel((10, 1)))
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


fig.plot()

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata, color = gene_oi)

# %% [markdown]
# ## Slices

# %%
import sklearn.metrics

# %%
# window_size = 50
window_size = 100
# window_size = 500

joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = window_size)
windows_oi = regionmultiwindow.design.query("window_size == @window_size").index
design = regionmultiwindow.design.loc[windows_oi]

# remove promoters
# joined_all.loc[(joined_all["window_mid"] > -1000) & (joined_all["window_mid"] < 1000), "HS_LS_logratio"] = np.nan

# lfc_cutoff = np.log(1.2)
lfc_cutoff = np.log(1.1)
# lfc_cutoff = np.log(2.0)
joined_all["significant_expression"] = joined_all["HS_LS_logratio"].abs() > lfc_cutoff

joined_genes = {k:v for k, v in joined_all.groupby("gene")}

# %%
import sklearn.metrics
import chromatinhd.data.peakcounts
import scipy.stats

# %%
# genes_oi = joined_all.groupby("gene")["significant_expression"].any() & ~joined_all.dropna().groupby("gene")["significant_expression"].all()
# genes_oi = genes_oi[genes_oi].index
transcriptome.symbol(genes_oi)

# %%
genescores = {}
slicescores = {}
allslicescores = {}

# %%
ws = [0, 1, 2, 3, 5, 7, 10 ,20,25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

# %%
deltacor_cutoff = -0.001

method_names = [
    # "v30/positive",
    # "v30/negative",
    # "v30/effect",
    "accessibility",
    "v30",
    # "v30/normalized",
    # "v30/interaction",
    # "v30/interaction_effect",
    # "v30/pure_effect",
    # "v30/deltacor2",
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
            if method_name == "v30":
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

            joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
            
            joined_oi.loc[joined_oi.index[0], "significant_expression"] = False
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
peakcallers = [
    "encode_screen",
    "macs2_leiden_0.1_merged",
    # "macs2_leiden_0.1",
    "cellranger",
    "rolling_500",
    "rolling_100",
    "rolling_50",
    # "genrich",
]

# %%
for peakcaller in peakcallers:
    peakcounts = chd.flow.Flow.from_path(
        path=chromatinhd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / "100k100k"
        # path=chromatinhd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / "500k500k"
    )

    transcriptome.var["ix"] = np.arange(transcriptome.var.shape[0])

    # for method_name in [peakcaller + "/linear_absolute"]:
    for method_name in [peakcaller + "/linear"]:
    # for method_name in [peakcaller + "/linear", peakcaller + "/linear_absolute", peakcaller + "/linear_positive"]:
        genescores[method_name] = []
        slicescores[method_name] = []
        allslicescores[method_name] = []

        for gene_oi in tqdm.tqdm(genes_oi):
            gene = transcriptome.var.loc[gene_oi]

            peak_gene_links_oi, x = peakcounts.get_peak_counts(gene_oi)
            peak_gene_links_oi = chd.data.peakcounts.peakcounts.center_peaks(peak_gene_links_oi, fragments.regions.coordinates.loc[gene_oi])
            y = transcriptome.layers["magic"][:, transcriptome.var.index.get_loc(gene_oi)]

            x = x.T
            y = y[:, None].T

            cors = np.corrcoef(x, y)[:-1, -1]
            cors[np.isnan(cors)] = 0
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

                if method_name.endswith("linear"):
                    joined_oi["score"] = crispri.rolling_max(joined_oi["cor"].values, w)
                elif method_name.endswith("linear_absolute"):
                    joined_oi["score"] = crispri.rolling_max(np.abs(joined_oi["cor"].abs().values), w)
                elif method_name.endswith("linear_positive"):
                    joined_oi["score"] = crispri.rolling_max(np.where(joined_oi["cor"] > 0, joined_oi["cor"].values, 0), w)

                joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
                joined_oi.loc[joined_oi.index[0], "significant_expression"] = False

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
    joined_oi.loc[joined_oi.index[0], "significant_expression"] = False
    
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
# ### All genes together

# %%
allslicescores_stacked = pd.concat([pd.concat(allslicescores[method_name]) for method_name in allslicescores.keys()], ignore_index=True)
slicescores_stacked = pd.concat([pd.concat(slicescores[method_name]) for method_name in slicescores.keys()], ignore_index=True)

# %%
w_oi = 10

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

for method, plotdata in allslicescores_stacked.query("w == @w_oi").groupby("method"):
    panel, ax = fig.main.add(chd.grid.Panel((2, 2)))
    cors = []
    for gene, plotdata in plotdata.groupby("gene"):
        ax.set_title(method)
        ax.scatter(-plotdata["HS_LS_logratio"], plotdata["score"], s = 1)
        cors.append(np.corrcoef(-plotdata["HS_LS_logratio"], plotdata["score"] + np.random.normal(scale = 1e-5, size = len(plotdata)))[0, 1])
    ax.annotate(f"cor = {sum(cors)/len(cors):.2f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top")

fig.plot()

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

w_oi = 10.
for method_name, slicescores_oi in allslicescores_stacked.query("w == @w_oi").groupby("method"):
    panel, ax = fig.main.add(chd.grid.Panel((2, 2)))

    x = np.clip(slicescores_oi["score"].abs(), 0, np.inf)
    # x = -slicescores_oi["score_rank"]
    # x = scipy.stats.rankdata(slicescores_oi["score"] + np.random.normal(scale = 1e-5, size = len(slicescores_oi)))
    y = -slicescores_oi["HS_LS_logratio"]
    # y = scipy.stats.rankdata(slicescores_oi["HS_LS_logratio"])

    cor = np.corrcoef(x, y)[0, 1]
    r2 = cor**2
    ax.annotate(f"r2 = {r2:.2f}", (0.05, 0.95), xycoords = "axes fraction", ha = "left", va = "top")
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
sns.heatmap(methodscores.groupby(["w", "method"])["cor"].mean().unstack().T, vmin = 0)

# %%
methodscores.loc[(methodscores["method"] == "all") | (methodscores["w"] == w_oi)].style.bar()

# %%
prediction_methods = chdm.methods.prediction_methods
prediction_methods.loc["all"] = {
    "color":"grey",
    "label":"Random"
}
prediction_methods.loc["accessibility"] = {
    "color":"grey",
    "label":"Accessibility"
}
for method_name in ["v30", "v30/positive", "v30/negative", "v30/effect", "v30/interaction", "v30/interaction_abs", "v30/deltacor2", "v30/interaction_effect", "v30/pure_effect", "v30/normalized"]:
    prediction_methods.loc[method_name] = {
        "color":"blue",
        "label":"ChromatinHD " + (method_name.split("/")[1] if "/" in method_name else "")
    }
for peakcaller in peakcallers:
    prediction_methods.loc[peakcaller + "/linear"] = prediction_methods.loc[peakcaller + "/linear"]
    prediction_methods.loc[peakcaller + "/linear_absolute"] = prediction_methods.loc[peakcaller + "/linear"].copy()
    prediction_methods.loc[peakcaller + "/linear_absolute", "label"] += " abs"
    prediction_methods.loc[peakcaller + "/linear_positive"] = prediction_methods.loc[peakcaller + "/linear"].copy()
    prediction_methods.loc[peakcaller + "/linear_positive", "label"] += " pos"

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width = 0.2))

plotdata = methodscores.loc[(methodscores["w"] == w_oi) | (methodscores["method"] == "all")].copy()
methods = prediction_methods.loc[methodscores["method"].unique()]

panel, ax = fig.main.add_under(chd.grid.Panel((2, len(methods)*0.3)))
color = prediction_methods.reindex(plotdata["method"])["color"]
ax.barh(plotdata["method"], plotdata["aupr"], color = color)
ax.set_xlim(0., 1)
ax.set_xlabel("AUPRC")
ax.set_yticks(np.arange(len(plotdata["method"])))
ax.set_yticklabels(prediction_methods.reindex(plotdata["method"])["label"])

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["auroc"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("AUROC")
ax.set_title("Fulco et al. 2019")

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["cor"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("cor")

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["r2"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("r2")

fig.plot()

# %%
fig, (ax_aupr, ax_auroc) = plt.subplots(1, 2, figsize = (5, 2))

eps = 1e-5

allslicescores_oi = allslicescores_stacked.loc[(allslicescores_stacked["w"] == 20) | (allslicescores_stacked["method"] == "all")]

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
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (5, 2))
genescores_oi = genescores_stacked.query("method == 'v30'")
for gene_name, plotdata in genescores_oi.groupby("gene"):
    ax0.plot(plotdata["w"], plotdata["aupr"], label = gene_name)
    ax1.plot(plotdata["w"], plotdata["auroc"], label = gene_name)
ax0.set_xlabel("w")
ax1.set_xlabel("w")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %%
sns.heatmap(genescores_stacked.groupby(["w", "method"])["auroc"].mean().unstack())

# %%
plotdata = genescores_stacked.loc[(genescores_stacked["w"] == 30) | (genescores_stacked["method"] == "all")].groupby(["method", "gene"]).mean(numeric_only = True)["aupr"].unstack()
fig, ax = plt.subplots(figsize = np.array(plotdata.shape) * 0.5)
norm = mpl.colors.Normalize(vmin = 0)
cmap = mpl.colormaps["rocket_r"]

matshow = ax.matshow(plotdata, vmin = 0, cmap = cmap)
ax.set_yticks(np.arange(len(plotdata.index)))
ax.set_yticklabels(prediction_methods.reindex(plotdata.index)["label"])
ax.set_xticks(np.arange(len(plotdata.columns)))
ax.set_xticklabels(transcriptome.symbol(plotdata.columns), rotation = 90)
fig.colorbar(matshow, ax = ax, label = "AUPRC")

# %%
w_oi = 0

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width = 0.2))

plotdata = genescores_stacked.loc[(genescores_stacked["w"] == w_oi) | (genescores_stacked["method"] == "all")].groupby("method").mean(numeric_only = True).reset_index()
methods = prediction_methods.loc[plotdata["method"].unique()]

panel, ax = fig.main.add_under(chd.grid.Panel((2, len(methods)*0.3)))
color = prediction_methods.reindex(plotdata["method"])["color"]
ax.barh(plotdata["method"], plotdata["aupr"], color = color)
ax.set_xlim(0., 1)
ax.set_xlabel("AUPRC")
ax.set_yticks(np.arange(len(plotdata["method"])))
ax.set_yticklabels(prediction_methods.reindex(plotdata["method"])["label"])

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["auroc"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("AUROC")
ax.set_title("Fulco et al. 2019")
# ax.set_xlim(0.5, 1)

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["cor"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("cor")

fig.plot()

# %%
# explore the results of one gene, where is the increase in lfc coming from?
gene_oi = transcriptome.gene_id("ETS1")
allslicescores_stacked.query("method == 'v30'").query("w == 2").query("gene == @gene_oi").sort_values("HS_LS_logratio", ascending = False)

# %%
fig, ax = plt.subplots()
plotdata = allslicescores_stacked.query("method == 'v30'").query("w == 30").query("gene == @gene_oi")
ax.scatter(plotdata["mid"], plotdata["score"], color = "orange")
ax2 = ax.twinx()
ax2.scatter(plotdata["mid"], -plotdata["HS_LS_logratio"], color = "red")
plotdata = allslicescores_stacked.query("method == 'cellranger/linear_absolute'").query("w == 10").query("gene == @gene_oi")
ax3 = ax.twinx()
ax3.scatter(plotdata["mid"], plotdata["score"], color = "blue")
