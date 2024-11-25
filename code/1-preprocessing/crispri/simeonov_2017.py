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
import polyptich as pp
pp.setup_ipython()

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
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
chd.set_default_device("cuda:1")

# %% [markdown]
# ### Download CRISPRa data

# %%
folder = chd.get_output() / "data" / "crispri" / "simeonov_2017"

# %%
file = folder / "41586_2017_BFnature23875_MOESM3_ESM.xlsx"
file.parent.mkdir(parents=True, exist_ok=True)

import requests
if not file.exists():
    response = requests.get("https://static-content.springer.com/esm/art%3A10.1038%2Fnature23875/MediaObjects/41586_2017_BFnature23875_MOESM3_ESM.xlsx")
    with open(file, "wb") as f:
        f.write(response.content)

# %%
data_orig1 = pd.read_excel(file, sheet_name="cd69_screen_full_results")
data_orig1["Gene"] = "CD69"
data_orig1 = data_orig1.loc[~pd.isnull(data_orig1["chr"])]
data_orig1["chrom"] = data_orig1["chr"]

data_orig2 = pd.read_excel(file, sheet_name="il2ra_full_screen_results")
data_orig2["Gene"] = "IL2RA"
data_orig2 = data_orig2.loc[~pd.isnull(data_orig2["chr"])]
data_orig2["chrom"] = data_orig2["chr"]

##
stacked = (data_orig1.iloc[:, 1:11]).T
stacked = stacked.values / stacked.mean(1).values[:, None]

x = np.array(([0] * 2) + ([1] * 2) + ([2] * 2) + ([3] * 2) + ([4] * 2))
y = np.log1p(stacked)

# a simple linear regression between the bins
slope = ((x-x.mean())[:, None] * (y - y.mean(0))).sum(0) / ((x - x.mean())**2).sum()
intercept = y.mean(0) - slope * x.mean()
lfc = slope
lfc[np.isnan(lfc)] = 0

data_orig1["HS_LS_logratio"] = lfc

stacked = (data_orig2.iloc[:, 1:11]).T
stacked = stacked.values / stacked.mean(1).values[:, None]

x = np.array(([0] * 2) + ([1] * 2) + ([2] * 2) + ([3] * 2) + ([4] * 2))
y = np.log1p(stacked)

# a simple linear regression between the bins
slope = ((x-x.mean())[:, None] * (y - y.mean(0))).sum(0) / ((x - x.mean())**2).sum()
intercept = y.mean(0) - slope * x.mean()
lfc = slope
lfc[np.isnan(lfc)] = 0

data_orig2["HS_LS_logratio"] = lfc
##

data_orig = pd.concat([data_orig1, data_orig2], axis=0)
data_orig.index = np.arange(stop=len(data_orig))

data = data_orig.copy()
data["start_orig"] = data["PAM_3primeEnd_coord"]
data["end_orig"] = data["PAM_3primeEnd_coord"]+1

import liftover
converter = liftover.get_lifter("hg19", "hg38")
data["start"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["start_orig"])]
data["end"] = [converter[chrom][pos][0][1] for chrom, pos in zip(data["chrom"], data["end_orig"])]

# %%
data.loc[data["HS_LS_logratio"] < 0, "HS_LS_logratio"] = 0

# %%
data["HS_LS_logratio"].plot()

# %%
data["Significant"] = np.abs(data["HS_LS_logratio"]) > 0.5

# %% [markdown]
# ### Store

# %%
data[["start", "end", "HS_LS_logratio", "chrom", "Gene"]].to_csv(folder / "data.tsv", sep = "\t", index=False)

# %% [markdown]
# ### Load interpretation

# %%
dataset_name = "pbmc10k"
# dataset_name = "hspc"
# dataset_name = "lymphoma"

# %%
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / "100k100k")

# %%
models = [chd.models.pred.model.additive.Model(chd.get_output() / "pred" /dataset_name / "100k100k" / "5x5" / "normalized" / "v20" / str(model_ix)) for model_ix in range(5)]
# models = [chd.models.pred.model.additive.Model(chd.get_output() / "pred" /dataset_name / "100k100k" / "5x5" / "magic" / "v20" / str(model_ix)) for model_ix in range(3)]
folds = chd.data.folds.Folds(chd.get_output() / "datasets" /dataset_name / "folds" / "5x5")

# %%
symbols_oi = transcriptome.var["symbol"][transcriptome.var["symbol"].isin(data["Gene"])].tolist()
genes_oi = transcriptome.gene_id(symbols_oi)
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window, (500, ), relative_stride=1)
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset = True)
regionmultiwindow.score(fragments, transcriptome, models, folds, censorer, regions = genes_oi)
regionmultiwindow.interpolate()

# %% [markdown]
# ### Check which symbols overlap

# %%
joined_all = []
for gene_oi in tqdm.tqdm(genes_oi):
# for gene_oi in tqdm.tqdm(transcriptome.gene_id(["JUNB"])):
    region = fragments.regions.coordinates.loc[gene_oi]
    symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]
    
    data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
    data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
    data_oi["start"] = data_oi["start"].astype(int)
    data_oi["end"] = data_oi["end"].astype(int)

    data_oi["z"] = (data["HS_LS_logratio"]) / data["HS_LS_logratio"].std()

    data_oi = data_oi.loc[data_oi["start"] > region["start"]]
    data_oi = data_oi.loc[data_oi["end"] < region["end"]]

    if data_oi.shape[0] > 0:
        data_centered = chd.plot.genome.genes.center(data_oi, region)
        data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

        data_centered["bin"] = regionmultiwindow.design.index[np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"])-1]
        data_binned = data_centered.groupby("bin").mean(numeric_only = True)

        joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
        joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]
        # joined = joined.loc[joined["lost"] > 0.01]

        joined_shared = joined.loc[joined["HS_LS_logratio"].notnull()].copy()
        joined_shared["gene"] = gene_oi
        joined_all.append(joined_shared)
joined_all = pd.concat(joined_all)

# %%
fig, ax =plt.subplots(figsize = (5, 2))
ax.scatter(joined_all["deltacor"], joined_all["HS_LS_logratio"], c = joined_all["Significant"], cmap = "viridis", s = 10, vmin = 0, vmax = 1)

fig, ax =plt.subplots(figsize = (5, 2))
colors = pd.Series(sns.color_palette("tab20", len(joined_all["gene"].unique())), index = joined_all["gene"].unique())
ax.scatter(joined_all["deltacor"], joined_all["HS_LS_logratio"], c = colors[joined_all["gene"]])
for gene, color in colors.items():
    ax.scatter([], [], c = color, label = transcriptome.symbol(gene))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axvline(-0.00)

# %%
joined_all["significant_expression"] = joined_all["Significant"] > 0
joined_all["significant_chd"] = joined_all["deltacor"] < -0.001

# %%
confirmed = (joined_all["significant_expression"] & joined_all["significant_chd"]).sum() / joined_all["significant_chd"].sum()

randoms = []
for i in range(1000):
    randoms.append((joined_all.iloc[np.random.permutation(np.arange(len(joined_all)))]["significant_expression"].values & joined_all["significant_chd"]).sum() / joined_all["significant_chd"].sum())
randoms = np.array(randoms)

fig, ax =plt.subplots(figsize = (5, 2))
ax.hist(randoms, bins = np.linspace(0, 1, 20), density = True)
ax.axvline(confirmed, color = "red")
(randoms >= confirmed).mean()

# %%
import fisher
contingency = pd.crosstab(joined_all["significant_expression"], joined_all["significant_chd"])
fisher.pvalue(*contingency.values.flatten())
odds = (contingency.iloc[1, 1] * contingency.iloc[0, 0]) / (contingency.iloc[1, 0] * contingency.iloc[0, 1])
odds

# %%
gene_oi = transcriptome.gene_id("CD69")
# gene_oi = transcriptome.gene_id("IL2RA")

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata, color = [gene_oi])

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
data_oi["start"] = data_oi["start"].astype(int)
data_oi["end"] = data_oi["end"].astype(int)

data_oi = data_oi.loc[data_oi["start"] > region["start"]]
data_oi = data_oi.loc[data_oi["end"] < region["end"]]

data_centered = chd.plot.genome.genes.center(data_oi, region)
data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

data_centered["bin"] = regionmultiwindow.design.index[np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"])-1]
data_binned = data_centered.groupby("bin").mean(numeric_only = True)

# %%
joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height = 0))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = fragments.regions.window
# window = [-10000, 20000]
# window = [-20000, 0] # TSS
# window = [-10000, 10000] # TSS
window = [-40000, -30000]

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width = 10, window = window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 1)))
ax.bar(joined["window_mid"], joined["HS_LS_logratio"], lw = 0, width = binwidth)
ax.set_ylim(*ax.get_ylim()[::-1])
ax.set_xlim(*window)
ax.set_ylabel("CRISPRi score", rotation = 0, ha = "right", va = "center")
ax.set_xticks([])

panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 0.5)))
ax.bar(joined["window_mid"], joined["deltacor"], lw = 0, width = binwidth)
ax.set_xlim(*window)
ax.set_ylabel("$\Delta cor$", rotation = 0, ha = "right", va = "center")
ax.set_ylim(0, joined["deltacor"].min())
ax.set_xticks([])

panel, ax = fig.main.add_under(polyptich.grid.Panel((10, 0.5)))
ax.bar(joined["window_mid"], joined["lost"], lw = 0, width = binwidth)
ax.set_xlim(*window)
ax.set_ylabel("# fragments", rotation = 0, ha = "right", va = "center")
ax.set_xticks([])

panel, ax = fig.main.add_under(chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = fragments.regions.window, width = 10))
ax.set_xlim(*window)
ax.set_xticks([])

fig.plot()

# %%
fig, ax = plt.subplots()
ax.scatter(joined["window_mid"], joined["HS_LS_logratio"])
ax2 = ax.twinx()
joined_shared = joined.loc[joined["HS_LS_logratio"].notnull()]
ax2.scatter(joined_shared["window_mid"], joined_shared["deltacor"], color = "orange")
# ax.set_xlim(-1000, 1000)

# %%
fig, ax = plt.subplots()
ax.plot(joined["window_mid"], joined["lost"], color = "green")
joined_shared = joined.loc[joined["HS_LS_logratio"].notnull()]
ax.scatter(joined_shared["window_mid"], joined_shared["lost"], c = joined_shared["HS_LS_logratio"], cmap = "viridis")

# %%
fig, ax = plt.subplots()
ax.plot(joined["window_mid"], joined["deltacor"], color = "orange")
joined_shared = joined.loc[joined["HS_LS_logratio"].notnull()]
ax.scatter(joined_shared["window_mid"], joined_shared["deltacor"], c = joined_shared["HS_LS_logratio"], cmap = "viridis")

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata, color = gene_oi)

# %%
fig, ax = plt.subplots()
joined_oi = joined.loc[joined["lost"] > 0.1]
ax.scatter(joined["deltacor"], joined["HS_LS_logratio"])
ax.scatter(joined_oi["deltacor"], joined_oi["HS_LS_logratio"], color = "orange")
ax.scatter(joined["deltacor"], [0] * len(joined), color = "grey")
ax.set_xlabel("deltacor")
ax.set_ylabel("HS_LS_logratio")

# %%
fig, ax = plt.subplots()
joined_oi = joined.loc[joined["lost"] > 0.1]
ax.scatter(joined["lost"], joined["HS_LS_logratio"])
ax.scatter(joined_oi["lost"], joined_oi["HS_LS_logratio"], color = "orange")
ax.scatter(joined["lost"], [0] * len(joined), color = "grey")
ax.set_xlabel("lost")
ax.set_ylabel("HS_LS_logratio")

# %%
fig, ax = plt.subplots()
ax.scatter(data_centered["start"], data_centered["HS_LS_logratio"], s=30)
(regionmultiwindow.interpolated[gene_oi]["deltacor"]*10).plot(ax = ax, color = "green")
ax.set_xlim(-100000, 100000)

# %%
regionmultiwindow.interpolated[gene_oi]["deltacor"].plot()

# %% [markdown]
# ## Slices

# %%
joined_genes = {}
for gene_oi in tqdm.tqdm(genes_oi):
# for gene_oi in tqdm.tqdm(transcriptome.gene_id(["JUNB"])):
    region = fragments.regions.coordinates.loc[gene_oi]
    symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]
    
    data_oi = data.loc[data["chrom"] == region["chrom"]].copy()
    data_oi = data_oi.loc[(data_oi["Gene"] == symbol_oi)]
    data_oi["start"] = data_oi["start"].astype(int)
    data_oi["end"] = data_oi["end"].astype(int)

    data_oi = data_oi.loc[data_oi["start"] > region["start"]]
    data_oi = data_oi.loc[data_oi["end"] < region["end"]]

    if data_oi.shape[0] > 0:
        data_centered = chd.plot.genome.genes.center(data_oi, region)
        data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

        data_centered["bin"] = regionmultiwindow.design.index[np.digitize(data_centered["mid"], regionmultiwindow.design["window_mid"])-1]
        data_binned = data_centered.groupby("bin").mean(numeric_only = True)
        data_binned["n_guides"] = data_centered.groupby("bin").size()
        
        data_binned = data_binned.reindex(regionmultiwindow.design.index)

        joined = regionmultiwindow.scores[gene_oi].mean("model").to_pandas().join(data_binned, how="left")
        joined["window_mid"] = regionmultiwindow.design.loc[joined.index, "window_mid"]

        joined_genes[gene_oi] = joined

# %%
genescores = {}
slicescores = {}

# %%
import chromatinhd.data.peakcounts

# %%
for peakcaller in [
    "macs2_leiden_0.1_merged",
    # "macs2_leiden_0.1",
    "cellranger",
    "rolling_500",
    "genrich",
]:
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

    method_name = peakcaller + "/" + "cor"
    genescores[method_name] = []
    slicescores[method_name] = []

    for gene_oi in tqdm.tqdm(genes_oi):
        # get correlation between peak and expression
        peak_gene_links_oi = peakcounts.peaks.loc[peakcounts.peaks["gene"] == gene_oi].copy()

        peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
        gene = transcriptome.var.loc[gene_oi]

        x, y = extract_data(gene, peaks_oi)

        cors = np.corrcoef(x.T, y[:, None].T)[:-1, -1]
        peakscores = cors
        
        peak_gene_links_oi["cor"] = cors

        joined_oi = joined_genes[gene_oi]

        cor = np.zeros(len(joined_oi))
        joined_oi["cor"] = 0.
        for peak_id, peak_gene_link in peak_gene_links_oi.iterrows():
            cor[np.arange(np.searchsorted(regionmultiwindow.design["window_start"], peak_gene_link.relative_start), np.searchsorted(regionmultiwindow.design["window_start"], peak_gene_link.relative_end))] = peak_gene_link["cor"]
        joined_oi["cor"] = cor

        # joined_cre = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].sort_values("cor", ascending = False).iloc[:top]
        joined_cre = joined_oi.loc[(joined_oi["cor"].abs() >= 0.05) & ~pd.isnull(joined_oi["HS_LS_logratio"])]

        genescores[method_name].append({
            "gene": gene_oi,
            "method": method_name,
            "lfc_mean": joined_cre["HS_LS_logratio"].abs().mean(),
            "n": joined_cre.shape[0],
        })

        slicescores[method_name].append(pd.DataFrame({
            "gene": gene_oi,
            "method": method_name,
            "lfc": joined_cre["HS_LS_logratio"].abs(),
            "slice":joined_cre.index,
        }))

# %%
fig, ax = plt.subplots()
for gene_oi in regionmultiwindow.scores.keys():
    regionmultiwindow.scores[gene_oi].mean("model")["deltacor"].to_pandas().plot()
ax.axhline(-0.001)

# %%
# for deltacor_cutoff in [-0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001]:
for deltacor_cutoff in [-0.001]:
# for deltacor_cutoff in [-0.05]:
    method_name = "v20" + "/" + "{:.0e}".format(deltacor_cutoff)
    print(method_name)
    genescores[method_name] = []
    slicescores[method_name] = []
    for gene_oi in tqdm.tqdm(genes_oi):
        joined_oi = joined_genes[gene_oi]
        joined_oi["significant_chd"] = joined_oi["deltacor"] < deltacor_cutoff

        # joined_chd = joined_oi.loc[~pd.isnull(joined_oi["HS_LS_logratio"])].sort_values("deltacor", ascending = True).iloc[:top]
        joined_chd = joined_oi.loc[joined_oi["significant_chd"] & ~pd.isnull(joined_oi["HS_LS_logratio"])]

        genescores[method_name].append({
            "gene": gene_oi,
            "method": method_name,
            "lfc_mean": joined_chd["HS_LS_logratio"].abs().mean(),
            "n": joined_chd.shape[0],
        })

        slicescores[method_name].append(pd.DataFrame({
            "gene": gene_oi,
            "method": method_name,
            "lfc": joined_chd["HS_LS_logratio"].abs(),
            "slice":joined_chd.index,
        }))

# %%
method_name = "all"
genescores[method_name] = []
slicescores[method_name] = []
for gene_oi in tqdm.tqdm(genes_oi):
    joined_oi = joined_genes[gene_oi]
    genescores[method_name].append({
        "gene": gene_oi,
        "method": method_name,
        "lfc_mean": joined_oi["HS_LS_logratio"].abs().mean(),
        "n": joined_chd.shape[0],
    })
    slicescores[method_name].append(pd.DataFrame({
        "gene": gene_oi,
        "method": method_name,
        "lfc": joined_oi["HS_LS_logratio"].abs(),
    }))

# %%
slicescores_stacked = pd.concat([pd.concat(x) for x in list(slicescores.values())])

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
plotdata = slicescores_stacked.groupby(["gene", "method"]).mean().unstack()["lfc"].T

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_width = 0.1))

cmap = plt.cm.get_cmap('magma')
cmap.set_bad('#EEEEEE')

norm = mpl.colors.Normalize(vmin = 0)

panel, ax = fig.main.add_under(polyptich.grid.Panel(np.array(plotdata.shape)[::-1] * 0.25))
ax.matshow(plotdata.values, cmap = cmap, aspect = "auto", norm = norm)
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
ax.set_xticks(np.arange(len(plotdata.columns)))
ax.set_xticklabels(transcriptome.symbol(plotdata.columns), rotation = 45, ha = "left")
ax.set_yticks(np.arange(len(plotdata.index)))
ax.set_yticklabels(plotdata.index)

panel, ax = fig.main.add_right(polyptich.grid.Panel([0.25, plotdata.shape[0] * 0.25]))
ax.matshow(slicescores_mean["lfc"].values[:, None], cmap = cmap, aspect = "auto", norm = norm)
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
ax.set_xticks([0])
ax.set_xticklabels(["Mean"], rotation = 45, ha = "left")
ax.set_yticks([])

fig.plot()

# %%
# explore the results of one gene, where is the increase in lfc coming from?
gene_oi = transcriptome.gene_id("CD69")
slicescores_stacked.query("method == 'v20/-1e-03'").query("gene == @gene_oi")
