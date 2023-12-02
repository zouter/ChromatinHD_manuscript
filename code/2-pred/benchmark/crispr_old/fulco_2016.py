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
chd.set_default_device("cuda:0")

# %% [markdown]
# ### Download CRISPRi data

# %%
file = chd.get_output() / "data" / "crispri" / "fulco_2016" / "aag2445_table_s2.xlsx"
file.parent.mkdir(parents=True, exist_ok=True)
import pathlib
file = pathlib.Path("aag2445_table_s2_2.xlsx")

import requests
if not file.exists():
    response = requests.get("https://www.science.org/doi/suppl/10.1126/science.aag2445/suppl_file/aag2445_table_s2.xlsx")
    with open(file, "wb") as f:
        f.write(response.content)

# %%
data_orig = pd.read_excel(file, skiprows=1, sheet_name="Table S2")
data_orig = data_orig.loc[data_orig["Set"].isin(["MYC Tiling", "GATA1 Tiling"])].copy()
data_orig["chrom"] = data_orig["chr"]
data_orig["HS_LS_logratio"] = data_orig["CRISPRi Score"]

data = data_orig.copy()
data["start_orig"] = data["start"]
data["end_orig"] = data["end"]

import liftover
converter = liftover.get_lifter("hg19", "hg38")
data["start"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["start_orig"])]
data["end"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["end_orig"])]

# %%
data["Gene"] = data["Set"].str.split(" ").str[0]

# %%
data["Significant"] = np.abs(data["CRISPRi Score"]) > 0.5

# %% [markdown]
# ### Load interpretation

# %%
dataset_name = "hspc"

# %%
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x5")

# %%
import chromatinhd.models.pred.model.better
model_folder = chd.get_output() / "pred" / dataset_name / "100k100k" / "5x5" / "magic" / "v30"
models = chd.models.pred.model.better.Models.create(fragments = fragments, transcriptome = transcriptome, folds = folds, path = path)

# %%
# model_folder = chd.get_output() / "pred" / "pbmc10k" / "100k100k" / "5x5" / "normalized" / "v20"

# models = [chd.models.pred.model.additive.Model(model_folder / str(model_ix)) for model_ix in range(25)]
# folds = chd.data.folds.Folds(chd.get_output() / "datasets" / "pbmc10k" / "folds" / "5x5")

# %%
symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)

data = data.loc[data["Gene"].isin(symbols_oi)].copy()
data["gene"] = transcriptome.gene_id(data["Gene"]).values

# %%
models.train_models(regions_oi = genes_oi.tolist(), device = "cuda")

# %%
model_folder = chd.get_output() / "test"

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, (100, 500, ), relative_stride = 1)

scoring_folder = model_folder / "scoring" / "crispri" / "fulco_2016"

regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(scoring_folder / "regionmultiwindow")
regionmultiwindow.score(models, censorer, regions = genes_oi)
regionmultiwindow.interpolate()

# %% [markdown]
# ## Check enrichment

# %%
# joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 500)
joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = 100)
joined_observed = joined_all.loc[~pd.isnull(joined_all["HS_LS_logratio"])].copy()

# %%
deltacor_cutoff = -0.001
lfc_cutoff = np.log(1.5)

# %%
joined_observed["significant_expression"] = joined_observed["HS_LS_logratio"].abs() > lfc_cutoff
joined_observed["significant_chd"] = joined_observed["deltacor"] < deltacor_cutoff

# %%
for gene_oi, joined_oi in joined_observed.groupby("gene"):
    confirmed_n  = (joined_oi["significant_expression"] & joined_oi["significant_chd"]).sum()
    total_n = joined_oi["significant_chd"].sum()
    confirmed = confirmed_n / total_n

    randoms = []
    for i in range(1000):
        randoms.append((joined_oi.iloc[np.random.permutation(np.arange(len(joined_oi)))]["significant_expression"].values & joined_oi["significant_chd"]).sum() / joined_oi["significant_chd"].sum())
    randoms = np.array(randoms)

    p = (randoms > confirmed).mean()

    print(f"{transcriptome.symbol(gene_oi)} Observed {confirmed:.2%} Random {randoms.mean():.2%} p-value {p:.2f}")

# %%
confirmed_n  = (joined_observed["significant_expression"] & joined_observed["significant_chd"]).sum()
total_n = joined_observed["significant_chd"].sum()
confirmed = confirmed_n / total_n

randoms = []
for i in range(10000):
    randoms.append((joined_observed.iloc[np.random.permutation(np.arange(len(joined_observed)))]["significant_expression"].values & joined_observed["significant_chd"]).sum() / joined_observed["significant_chd"].sum())
randoms = np.array(randoms)

fig, ax =plt.subplots(figsize = (2, 1))
ax.hist(randoms, bins = np.linspace(0, 1, 20), density = True, label = "Random", lw = 0)
ax.axvline(confirmed, color = "red", label = "Observed")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.set_xlabel("Ratio high CRISPRi score\nvs high+low CRISPRi score")
ax.set_ylabel("Density\n(n=10000)")
(randoms >= confirmed).mean(), f"{confirmed:.2%}", f"{randoms.mean():.2%}"

# %%
fig, ax =plt.subplots(figsize = (5, 2))
ax.scatter(joined_observed["deltacor"], joined_observed["HS_LS_logratio"], c = joined_observed["significant_expression"], cmap = "viridis", s = 10, vmin = 0, vmax = 1)

fig, ax =plt.subplots(figsize = (5, 2))
colors = pd.Series(sns.color_palette("tab20", len(joined_observed["gene"].unique())), index = joined_observed["gene"].unique())
ax.scatter(joined_observed["deltacor"], joined_observed["HS_LS_logratio"], c = colors[joined_observed["gene"]], s = 2)
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
models["ENSG00000136997_0"]

# %%
fig, ax =plt.subplots(figsize = (5, 2))
ax.scatter(joined_observed["deltacor"], joined_observed["HS_LS_logratio"], c = joined_observed["significant_expression"], cmap = "viridis", s = 10, vmin = 0, vmax = 1)

fig, ax =plt.subplots(figsize = (5, 2))
colors = pd.Series(sns.color_palette("tab20", len(joined_observed["gene"].unique())), index = joined_observed["gene"].unique())
ax.scatter(joined_observed["deltacor"], joined_observed["HS_LS_logratio"], c = colors[joined_observed["gene"]], s = 2)
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

# %%
data["Gene"].value_counts()[symbols_oi].sort_values()

# %% [markdown]
# ## Focus on one gene

# %%
gene_oi = transcriptome.gene_id("GATA1")

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
joined = joined_all.query("gene == @gene_oi")

# %%
# regionmultiwindow2 = regionmultiwindow

censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)
regionmultiwindow2 = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset = True)
regionmultiwindow2.score(fragments, transcriptome, models, folds, censorer, regions = [gene_oi])
regionmultiwindow2.interpolate()

# %%
regionmultiwindow2.interpolated[gene_oi]["deltacor"].values[114000:115000] = (regionmultiwindow2.interpolated[gene_oi]["deltacor"].values[114000:115000] * 3)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.1))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = fragments.regions.window
# window = [-10000, 10000]

window = [-10000, 20000]  # GATA1 region oi
arrows = [{"position": 14000, "orientation": "right", "y": 0.5}]

# window = [-10000, 10000] # KLF1 TSS
# window = [-65000, -45000] # CALR upstream

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width=10, window=window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(
    chd.models.pred.plot.Pileup(regionmultiwindow2.get_plotdata(gene_oi), window=window, width=10)
)  # (regionmultiwindow2.get_plotdata(gene_oi), window = window, width = 10))

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
ax.set_ylabel(f"CRISPRi\nfold enrichment\ngrowth selection", rotation=0, ha="right", va="center")
# ax.set_ylabel(f"CRISPRi\nfold enrichment\nhigh vs low {transcriptome.symbol(gene_oi)}", rotation = 0, ha = "right", va = "center")
ax.set_xticks([])
ax.set_yticks(np.log([0.125, 0.25, 0.5, 1, 2]))
ax.set_yticklabels(["⅛", "¼", "½", 1, 2])

panel, ax = fig.main.add_under(
    chd.models.pred.plot.Predictivity(regionmultiwindow2.get_plotdata(gene_oi), window=window, width=10)
)
for arrow in arrows:
    panel.add_arrow(**arrow)

fig.plot()

# %%
arrows = [
    {"position":14000, "angle":45, "y":0.5}
]

arrow = arrows[0]

panel, ax = fig.main[2, 0]
trans = mpl.transforms.blended_transform_factory(x_transform=ax.transData, y_transform=ax.transAxes)
ax.annotate(
    text = "",
    xy = (arrow["position"], arrow["y"]),
    xytext = (-15, 15),
    textcoords = "offset points",
    xycoords = trans,
    arrowprops = dict(arrowstyle = "-|>", color = "black", lw = 1, connectionstyle = "arc3"),
)

fig.plot()
fig

# %%
import scanpy as sc
symbols_oi2 = ["CALR"]
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(symbols_oi2), title = symbols_oi2)

# %% [markdown]
# ## Slices

# %%
import sklearn.metrics

# %%
genes_oi = [transcriptome.gene_id("GATA1")]

# %%
window_size = 100
# window_size = 500

joined_all = crispri.calculate_joined(regionmultiwindow, data, genes_oi, fragments.regions, window_size = window_size)
windows_oi = regionmultiwindow.design.query("window_size == @window_size").index
design = regionmultiwindow.design.loc[windows_oi]

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
ws = [0, 1, 2, 3, 5, 7, 10 ,20,25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

# %%
deltacor_cutoff = -0.001

method_name = "v20"
print(method_name)
genescores[method_name] = []
slicescores[method_name] = []
allslicescores[method_name] = []
for gene_oi in tqdm.tqdm(genes_oi):
    for w in ws:
        joined_oi = joined_genes[gene_oi]
        joined_oi["significant_cor"] = joined_oi["deltacor"] < deltacor_cutoff

        joined_oi["score"] = -joined_oi["deltacor"]
        joined_oi["score"] = crispri.rolling_max(-joined_oi["deltacor"].values, w)

        joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()
        
        aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
        auroc = sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], joined_oi["score"])

        confirmed = (joined_oi["significant_expression"] & joined_oi["significant_cor"]).sum() / joined_oi["significant_cor"].sum()

        allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, w = w))

        joined_chd = joined_oi.loc[joined_oi["significant_cor"]]

        genescores[method_name].append({
            "gene": gene_oi,
            "method": method_name,
            "lfc_mean": joined_chd["HS_LS_logratio"].abs().mean(),
            "n": joined_chd.shape[0],
            "aupr":aupr,
            "auroc":auroc,
            "cor":np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score"])[0, 1],
            "confirmed":confirmed,
            "w":w,
        })

        slicescores[method_name].append(pd.DataFrame({
            "gene": gene_oi,
            "method": method_name,
            "lfc": joined_chd["HS_LS_logratio"].abs(),
            "score": -joined_chd["deltacor"],
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

ax.set_xlim([-10000, 20000])
# ax.set_xlim([-10000, 10000])

# %%
aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score_original"])
aupr

# %%
aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
aupr

# %%
aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], np.random.permutation(joined_oi["score"]))
aupr

# %%
peakcallers = [
    "macs2_leiden_0.1_merged",
    # "macs2_leiden_0.1",
    "cellranger",
    "rolling_500",
    # "genrich",
]

# %%
for peakcaller in peakcallers:
    peakcounts = chromatinhd.data.peakcounts.PeakCounts(
        path=chromatinhd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / "100k100k"
    )
    X_peaks = peakcounts.counts
    X_transcriptome = transcriptome.layers["normalized"]
    def extract_data(gene_oi, peaks_oi):
        x = np.array(X_peaks[:, peaks_oi["ix"]].todense())
        y = np.array(X_transcriptome[:, gene_oi["ix"]])
        return x, y

    transcriptome.var["ix"] = np.arange(transcriptome.var.shape[0])

    var_peaks = peakcounts.var
    var_peaks["ix"] = np.arange(var_peaks.shape[0])

    method_name = peakcaller + "/" + "linear"
    genescores[method_name] = []
    slicescores[method_name] = []
    allslicescores[method_name] = []

    for gene_oi in tqdm.tqdm(genes_oi):
        # get correlation between peak and expression
        peak_gene_links_oi = peakcounts.peaks.loc[peakcounts.peaks["gene"] == gene_oi].copy()

        peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
        gene = transcriptome.var.loc[gene_oi]

        x, y = extract_data(gene, peaks_oi)

        x = x.T
        y = y[:, None].T

        cors = np.corrcoef(x, y)[:-1, -1]
        cors[np.isnan(cors)] = 0
        peakscores = cors
        
        peak_gene_links_oi["cor"] = cors

        joined_oi = joined_genes[gene_oi]

        cor = np.zeros(len(joined_oi))
        joined_oi["cor"] = 0.
        for peak_id, peak_gene_link in peak_gene_links_oi.iterrows():
            cor[np.arange(np.searchsorted(design["window_start"], peak_gene_link.relative_start), np.searchsorted(design["window_start"], peak_gene_link.relative_end))] = peak_gene_link["cor"]
        joined_oi["cor"] = cor
        joined_oi = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].copy()

        for w in ws:
            joined_oi = joined_oi.copy()
            joined_oi["score"] = crispri.rolling_max(joined_oi["cor"].abs().values, w)

            aupr = sklearn.metrics.average_precision_score(joined_oi["significant_expression"], joined_oi["score"])
            auroc = sklearn.metrics.roc_auc_score(joined_oi["significant_expression"], joined_oi["score"])
            
            joined_oi["significant_cor"] = joined_oi["cor"].abs() >= 0.05

            allslicescores[method_name].append(joined_oi.assign(method = method_name, gene = gene_oi, w = w))

            joined_cre = joined_oi.loc[joined_oi["significant_cor"]]
            
            genescores[method_name].append({
                "gene": gene_oi,
                "method": method_name,
                "lfc_mean": joined_cre["HS_LS_logratio"].abs().mean(),
                "n": joined_cre.shape[0],
                "aupr":aupr,
                "auroc":auroc,
                "cor":np.corrcoef(joined_oi["HS_LS_logratio"].abs(), joined_oi["score"] + np.random.normal(0, 1e-5, size = len(joined_oi)))[0, 1],
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

eps = 1e-5

methodscores = []
for (method_name, w), slicescores_oi in allslicescores_stacked.groupby(["method", "w"]):
    aupr = sklearn.metrics.average_precision_score(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    auroc = sklearn.metrics.roc_auc_score(slicescores_oi["significant_expression"], slicescores_oi["score"])

    methodscores.append({
        "method": method_name,
        "aupr": aupr,
        "auroc": auroc,
        "cor": np.corrcoef(slicescores_oi["HS_LS_logratio"].abs(), slicescores_oi["score"])[0, 1],
        "w":w
    })
methodscores = pd.DataFrame(methodscores)
methodscores["r2"] = methodscores["cor"]**2

# %%
methodscores.loc[(methodscores["method"] == "all") | (methodscores["w"] == 30)].style.bar()

# %%
fig, (ax_aupr, ax_auroc) = plt.subplots(1, 2, figsize = (5, 2))

eps = 1e-5

allslicescores_oi = allslicescores_stacked.loc[(allslicescores_stacked["w"] == 30) | (allslicescores_stacked["method"] == "all")]

for (method_name, w), slicescores_oi in allslicescores_oi.groupby(["method", "w"]):
    slicescores_oi["significant_expression"] = slicescores_oi["HS_LS_logratio"].abs() > lfc_cutoff

    curve = sklearn.metrics.precision_recall_curve(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    ax_aupr.plot(curve[1], curve[0], label = method_name)

    curve = sklearn.metrics.roc_curve(slicescores_oi["significant_expression"], slicescores_oi["score"] + np.random.normal(0, eps, slicescores_oi.shape[0]))
    ax_auroc.plot(curve[0], curve[1], label = method_name)
    ax_auroc.plot([0, 1], [0, 1], color = "black", linestyle = "--")
ax_auroc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %% [markdown]
# ### Per gene

# %%
genescores_stacked = pd.concat([pd.DataFrame(genescore) for genescore in genescores.values()], ignore_index=True)

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (5, 2))
genescores_oi = genescores_stacked
for method_name, plotdata in genescores_oi.groupby("method"):
    plotdata = plotdata.groupby("w").mean(numeric_only = True).reset_index()
    ax0.plot(plotdata["w"], plotdata["aupr"], label = method_name)
    ax1.plot(plotdata["w"], plotdata["auroc"], label = method_name)
ax0.set_xlabel("w")
ax1.set_xlabel("w")

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (5, 2))
genescores_oi = genescores_stacked.loc[genescores_stacked["gene"] == transcriptome.gene_id("GATA1")]
for method_name, plotdata in genescores_oi.groupby("method"):
    plotdata = plotdata.groupby("w").mean(numeric_only = True).reset_index()
    ax0.plot(plotdata["w"], plotdata["aupr"], label = method_name)
    ax1.plot(plotdata["w"], plotdata["auroc"], label = method_name)
ax0.set_xlabel("w")
ax1.set_xlabel("w")
ax0.set_title("GATA1")

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (5, 2))
genescores_oi = genescores_stacked.query("method == 'v20'")
for gene_name, plotdata in genescores_oi.groupby("gene"):
    ax0.plot(plotdata["w"], plotdata["aupr"], label = gene_name)
    ax1.plot(plotdata["w"], plotdata["auroc"], label = gene_name)
ax0.set_xlabel("w")
ax1.set_xlabel("w")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %%
prediction_methods = chdm.methods.prediction_methods
prediction_methods.loc["all"] = {
    "color":"grey",
    "label":"Random"
}

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_width = 0.2))

plotdata = genescores_stacked.loc[(genescores_stacked["w"] == 30) | (genescores_stacked["method"] == "all")].groupby("method").mean(numeric_only = True).reset_index()
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

panel, ax = fig.main.add_right(chd.grid.Panel((2, len(methods)*0.3)))
ax.barh(plotdata["method"], plotdata["cor"], color = color)
ax.set_xlim(0., 1)
ax.set_yticks([])
ax.set_xlabel("cor")

fig.plot()


# %% [markdown]
# ### Look per slice

# %%
slicescores_stacked = pd.concat([pd.concat(x) for x in list(slicescores.values())])

# %%
cumscores = []
fig, ax = plt.subplots()
for method_name, methoddata in slicescores_stacked.groupby("method"):
    ax.plot(np.arange(len(methoddata)), (methoddata.sort_values("score", ascending = False)["lfc"].cumsum() / (np.arange(len(methoddata))+1)), label = method_name)
plt.legend()
ax.set_xlim(0, 200)

# %%
fig, ax = plt.subplots()
methods = pd.DataFrame({
    "method": list(genescores.keys()),
    "ix": np.arange(len(genescores)),
}).set_index("method")
for method, plotdata in slicescores_stacked.groupby("method"):
    sns.ecdfplot(x = "lfc", data = plotdata.dropna(), ax = ax, label = f"{method} (n={plotdata.shape[0]})")
plt.legend()

# %%
slicescores_mean = slicescores_stacked.groupby("method").mean(numeric_only = True)
slicescores_mean["n"] = slicescores_stacked.groupby("method").size()
slicescores_mean.style.bar()

plotdata = slicescores_mean
fig, ax = plt.subplots(figsize = (2, 1.5))
ax.scatter(np.exp(plotdata["lfc"]), plotdata.index, color = "#333")
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

# %%
plotdata = slicescores_stacked.groupby(["gene", "method"]).mean(numeric_only = True).unstack()["lfc"].T

fig = chd.grid.Figure(chd.grid.Grid(padding_width = 0.1))

cmap = mpl.colormaps['YlGnBu']
cmap.set_bad('#EEEEEE')

norm = mpl.colors.Normalize(vmin = 0)

panel, ax = fig.main.add_under(chd.grid.Panel(np.array(plotdata.shape)[::-1] * 0.25))
ax.matshow(plotdata.values, cmap = cmap, aspect = "auto", norm = norm)
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
ax.set_xticks(np.arange(len(plotdata.columns)))
ax.set_xticklabels(transcriptome.symbol(plotdata.columns), rotation = 45, ha = "left")
ax.set_yticks(np.arange(len(plotdata.index)))
ax.set_yticklabels(plotdata.index)

panel, ax = fig.main.add_right(chd.grid.Panel([0.25, plotdata.shape[0] * 0.25]))
ax.matshow(slicescores_mean["lfc"].values[:, None], cmap = cmap, aspect = "auto", norm = norm)
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
ax.set_xticks([0])
ax.set_xticklabels(["Mean"], rotation = 45, ha = "left")
ax.set_yticks([])

fig.plot()

# %%
plotdata = slicescores_stacked.groupby(["gene", "method"]).mean(numeric_only = True).unstack()["lfc"].T
plotdata = genescores_stacked.groupby(["gene", "method"]).mean(numeric_only = True).unstack()["aupr"].T

fig = chd.grid.Figure(chd.grid.Grid(padding_width = 0.1))

cmap = mpl.colormaps['YlGnBu']
cmap.set_bad('#EEEEEE')

norm = mpl.colors.Normalize(vmin = 0)

panel, ax = fig.main.add_under(chd.grid.Panel(np.array(plotdata.shape)[::-1] * 0.25))
ax.matshow(plotdata.values, cmap = cmap, aspect = "auto", norm = norm)
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
ax.set_xticks(np.arange(len(plotdata.columns)))
ax.set_xticklabels(transcriptome.symbol(plotdata.columns), rotation = 45, ha = "left")
ax.set_yticks(np.arange(len(plotdata.index)))
ax.set_yticklabels(plotdata.index)

panel, ax = fig.main.add_right(chd.grid.Panel([0.25, plotdata.shape[0] * 0.25]))
ax.matshow(slicescores_mean["lfc"].values[:, None], cmap = cmap, aspect = "auto", norm = norm)
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
ax.set_xticks([0])
ax.set_xticklabels(["Mean"], rotation = 45, ha = "left")
ax.set_yticks([])

fig.plot()

# %%
# explore the results of one gene, where is the increase in lfc coming from?
gene_oi = transcriptome.gene_id("H1FX")
slicescores_stacked.query("method == 'v20'").query("w == 30").query("gene == @gene_oi").sort_values("lfc", ascending = False)
