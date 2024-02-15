# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

from IPython import get_ipython
import chromatinhd as chd

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

sns.set_style("ticks")

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, promoter_window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"
outcome_source = "counts"

# splitter = "permutations_5fold5repeat"
# promoter_name, promoter_window = "10k10k", np.array([-10000, 10000])
# outcome_source = "magic"
# prediction_name = "v20"
# prediction_name = "v21"

splitter = "permutations_5fold5repeat"
promoter_name, promoter_window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"
outcome_source = "magic"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = promoter_window[1] - promoter_window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# %%
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# %%
genes_oi = transcriptome.var.index
# genes_oi = transcriptome.gene_id(["CD74"])

# %% [markdown]
# ## Gather scores

# %%
def center_peaks(peaks, promoter):
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns=["start", "end", "method"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] - promoter["tss"]) * promoter["strand"],
                (peak["end"] - promoter["tss"]) * promoter["strand"],
            ][:: promoter["strand"]]
            for _, peak in peaks.iterrows()
        ]
    return peaks

# %% [markdown]
# ### Gather ChromatinHD scores

# %%
# chromatinhd scores
chdscores_genes = {}
for gene in tqdm.tqdm(genes_oi):
    try:
        scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
        windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

        scores_folder = prediction.path / "scoring" / "window_gene" / gene
        window_scoring = chd.scoring.prediction.Scoring.load(scores_folder) 
    except FileNotFoundError:
        continue

    promoter = promoters.loc[gene]

    score = (
        windowsize_scoring.genescores.mean("model")
        .sel(phase=["validation", "test"], gene=gene)
        .mean("phase")
        .to_pandas()
    ).reset_index()
    score.loc[:, ["window", "size"]] = windowsize_scoring.design[
        ["window", "size"]
    ].values

    deltacor_by_window = (
        score.set_index(["window", "size"])["deltacor"]
        .unstack()
    )

    lost_by_window = (
        score.set_index(["window", "size"])["lost"]
        .unstack()
    )

    windowscores = window_scoring.genescores.sel(gene=gene).sel(phase = ["test", "validation"]).mean("phase").mean("model").to_pandas()

    # ratio and number of reads per size bin
    ratio_12 = (lost_by_window[100] + 1) / (lost_by_window[30] + 1)
    n = lost_by_window.rename(columns = lambda x: "n_" + str(int(x)))
    deltacor = deltacor_by_window.rename(columns = lambda x: "deltacor_" + str(int(x)))
    windowscores["ratio"] = ratio_12.reindex(windowscores.index)
    windowscores["logratio"] = np.log(windowscores["ratio"])
    windowscores[n.columns] = n.reindex(windowscores.index)
    windowscores[deltacor.columns] = deltacor.reindex(windowscores.index)
    windowscores["absdeltacor"] = np.abs(windowscores["deltacor"])

    chdscores_genes[gene] = windowscores

# %% [markdown]
# ### Gather footprints

# %%
footprints_file = chd.get_git_root() / "tmp" / "rgt2" / "footprints.bed"
footprints_name = "HINT"

footprints_file.parent.mkdir(exist_ok = True, parents = True)
if not footprints_file.exists():
    # !rsync -a --progress wsaelens@updeplasrv6.epfl.ch:{footprints_file} {footprints_file.parent} -v
import pybedtools
bed_footprints = pybedtools.BedTool(str(footprints_file))

# it's unclear to me what the score means, it is not really described in the manuscript
# in any case, we put a threshold on it, otherwise there are footprints everywhere
footprints = bed_footprints.to_dataframe().query("score > exp(5)")

# %%
# gather footprints from the Vierstra 2020 paper
# this code works but is only used for supplementary figure 7
# overview: https://www.vierstra.org/resources/dgf

# footprint_samples = {
#     "h.CD4+-DS17212":{"label":"CD4+ T-cells"},
#     "h.CD8+-DS17885":{"label":"CD8+ T-cells"},
#     "h.CD14+-DS17215":{"label":"CD14+ Monocytes"},
#     "CD20+-DS18208":{"label":"CD20+ B-cells"},
# }
# footprints_name = "h.CD4+-DS17212"
# footprints_name = "h.CD8+-DS17885"
# footprints_name = "CD20+-DS18208"
# footprints_name = "h.CD14+-DS17215"
# footprints_file = chd.get_git_root() / "tmp" / footprints_name / "interval.all.fps.0.01.bed.gz"
# footprints_file.parent.mkdir(exist_ok = True, parents = True)
# if not footprints_file.exists():
    # !wget https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/{footprints_name}/interval.all.fps.0.01.bed.gz -O {footprints_file}
# import pybedtools
# bed_footprints = pybedtools.BedTool(str(footprints_file))
# footprints = bed_footprints.to_dataframe()

# %%
# footprint scores
footprintscores_genes = {}
for gene in tqdm.tqdm(genes_oi):
    footprints_gene = footprints.loc[(footprints["chrom"] == promoter["chr"]) & (footprints["start"] < promoter["end"]) & (footprints["end"] > promoter["start"])].copy()
    footprints_gene = center_peaks(footprints_gene, promoter)

    footprints_gene["mid"] = footprints_gene["end"] - (footprints_gene["end"] - footprints_gene["start"]) / 2
    footprints_gene["window"] = window_scoring.design.index[np.searchsorted(window_scoring.design["window_start"], footprints_gene["mid"]) - 1]

    windowscores = footprints_gene.groupby("window").size().to_frame("n_footprints")
    footprintscores_genes[gene] = windowscores

# %% [markdown]
# ### Gather ChIP-seq sites
# %%
# get information on each bed file
features_file = chd.get_output() / "bed/gm1282_tf_chipseq" / "files.csv"
features_file.parent.mkdir(exist_ok = True, parents = True)
if not features_file.exists():
    # !rsync -a --progress wsaelens@updeplasrv6.epfl.ch:{features_file} {features_file.parent} -v

# get the processes sites (filtered for sites within the 100k100k window around a TSS)
sites_file = chd.get_output() / "bed/gm1282_tf_chipseq_filtered" / "sites.csv"
sites_file.parent.mkdir(exist_ok = True, parents = True)
if not sites_file.exists():
    # !rsync -a --progress wsaelens@updeplasrv6.epfl.ch:{sites_file} {sites_file.parent} -v

# %%
sites = pd.read_csv(sites_file, index_col = 0)
sites["gene"] = pd.Categorical(sites["gene"], categories = transcriptome.var.index)

files = pd.read_csv(features_file, index_col = 0)
files_oi = files.copy()
sites = sites.loc[sites["file_accession"].isin(files_oi.accession)]

sites_genes = dict(list(sites.groupby("gene")))

# %% [markdown]
# ### Determine which binders are most associated with ChromatinHD scores

# %%
# First gather the ChromatinHD predictive scores for each position and gene
chdscores = pd.concat(chdscores_genes.values()).reset_index()
chdscores["gene"] = pd.Categorical(chdscores["gene"], categories = transcriptome.var.index)
chdscores["window"] = pd.Categorical(chdscores["window"], categories = window_scoring.design.index)
chdscores_deltacor = chdscores.set_index(["gene", "window"])["deltacor"].unstack()

# %%
# For each ChIP-seq file, determine the correlation between the number of sites and the ChromatinHD scores
file_scores = []
for accession, sites_oi in tqdm.tqdm(sites.groupby("file_accession")):
    sites_oi["mid"] = sites_oi["end"] - (sites_oi["end"] - sites_oi["start"]) / 2
    sites_oi["window"] = pd.Categorical(window_scoring.design.index[np.searchsorted(window_scoring.design["window_start"], sites_oi["mid"]) - 1], categories = window_scoring.design.index)

    chipscores_sites = sites_oi.groupby(["gene", "window"]).size().unstack().reindex(chdscores_deltacor.index)
    file_scores.append({
        "accession": accession,
        "cor":np.corrcoef(chdscores_deltacor.values.flatten(), chipscores_sites.values.flatten())[0, 1],
    })
file_scores = pd.DataFrame(file_scores).set_index("accession")
file_scores["experiment_target"] = files.set_index("accession").loc[file_scores.index, "experiment_target"]

# %%
# choose which filter to apply
filterer = "top30"
# filterer = "all"
if filterer == "top30":
    accessions_oi = file_scores.sort_values("cor", ascending = True).head(30).index
else:
    accessions_oi = file_scores.index

# %%
file_scores.sort_values("cor", ascending = True).head(10)

# %%
# get chip-seq scores of selected TFs
# basically, for each window the number of TFs that are bound
chipscores_genes = {}
for gene in tqdm.tqdm(genes_oi):
    sites_gene = sites_genes[gene].copy()
    sites_gene = sites_gene.loc[sites_gene["file_accession"].isin(accessions_oi)]

    sites_gene["mid"] = sites_gene["end"] - (sites_gene["end"] - sites_gene["start"]) / 2
    sites_gene["window"] = window_scoring.design.index[np.searchsorted(window_scoring.design["window_start"], sites_gene["mid"]) - 1]

    windowscores = sites_gene.groupby("window").size().to_frame("n_sites")
    chipscores_genes[gene] = windowscores

# %% [markdown]
# ## Combine scores
# windowscores_genes = {}
# for gene in tqdm.tqdm(genes_oi):
#     if gene in chdscores_genes:
#         windowscores = chdscores_genes[gene].copy()
#         windowscores["n_footprints"] = footprintscores_genes[gene].reindex(windowscores.index).fillna(0)
#         windowscores["n_sites"] = chipscores_genes[gene].reindex(windowscores.index).fillna(0)
#         windowscores_genes[gene] = windowscores

# %% [markdown]
# ### Individual gene
# %%
windowscores_oi = windowscores_genes[transcriptome.gene_id("BCL2")]
# %%
fig, ax = plt.subplots()
ax.set_xlim(*promoter_window)
ax.plot(windowscores_oi.index, windowscores_oi["n_sites"])
ax.plot(windowscores_oi.index, windowscores_oi["n_footprints"])
ax2 = ax.twinx()
ax2.plot(windowscores_oi.index, -windowscores_oi["deltacor"], color = "green", alpha = 0.5)

# %% [markdown]
# ### Determine differentiall expressed genes

# %%
sns.heatmap(windowscores_oi[["n_footprints", "n_sites", "absdeltacor"]].corr(), vmin = -1, vmax = 1, cmap = "coolwarm",  annot = True, fmt = ".2f")

# %%
import scanpy as sc

transcriptome.adata.obs["oi"] = pd.Categorical(
    np.array(["noi", "oi"])[
        transcriptome.adata.obs["celltype"]
        # .isin(["CD14+ Monocytes"]"])
        .isin(["naive B", "memory B", "Plasma"])
        .values.astype(int)
    ]
)
sc.tl.rank_genes_groups(transcriptome.adata, groupby="oi")
diffexp = (
    sc.get.rank_genes_groups_df(
        transcriptome.adata,
        group="oi",
    )
    .rename(columns={"names": "gene"})
    .assign(symbol=lambda x: transcriptome.var.loc[x["gene"], "symbol"].values)
    .set_index("gene")
)

genes_diffexp = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 0.1").index


# %% [markdown]
# ## Global smoothing

# %%
windowscores = pd.concat(windowscores_genes)

# %%
def smooth_spline_fit(x, y, x_smooth):
    import rpy2.robjects as robjects
    r_y = robjects.FloatVector(y)
    r_x = robjects.FloatVector(x)

    r_smooth_spline = robjects.r['smooth.spline'] #extract R function# run smoothing function
    spline1 = r_smooth_spline(x=r_x, y=r_y)
    ySpline=np.array(robjects.r['predict'](spline1,robjects.FloatVector(x_smooth)).rx2('y'))

    return ySpline

# %%
def smooth_spline_fit_se(x, y, x_smooth):
    import rpy2.robjects as ro
    ro.globalenv["x"] = ro.FloatVector(x)
    ro.globalenv["y"] = ro.FloatVector(y)
    ro.globalenv["x_pred"] = ro.FloatVector(x_smooth)
    script = """
    # Install and load the mgcv package if not yet done
    if (!require(mgcv)) {
    install.packages('mgcv')
    library(mgcv)
    }

    # Fit a GAM with a smoothing spline, just like smooth.spline
    gam_model <- gam(y ~ s(x, sp = 1), method = 'REML')

    # Make predictions
    y_pred <- predict(gam_model, newdata = data.frame(x = x_pred), type = 'response', se.fit = TRUE)

    # Extract predicted values and standard errors
    fit <- y_pred$fit
    se <- y_pred$se.fit

    list(fit = fit, se = se)
    """
    out = ro.r(script)
    fit = np.array(out[0])
    se = np.array(out[1])
    return np.stack([fit, se]).T

# %%
# Define a custom sampling function
def sample_rows(group, n = 1000):
    if len(group) < n:
        return group  # Return all rows if the group has fewer than n rows
    else:
        return group.sample(n)  # Randomly sample n rows from the group
import functools
windowscores_oi = windowscores.loc[~pd.isnull(windowscores["ratio"])].copy().groupby("n_sites").apply(functools.partial(sample_rows, n = 5000))
windowscores_diffexp = windowscores.loc[~pd.isnull(windowscores["ratio"]) & windowscores["gene"].isin(genes_diffexp)].copy().groupby("n_sites").apply(functools.partial(sample_rows, n = 5000))

plotdata_smooth = pd.DataFrame({
    "n_sites": np.linspace(0, min(len(accessions_oi)*2/3, 50), 100),
})

# %%
def add_smooth(plotdata_smooth, variable, windowscores_oi, suffix = ""):
    plotdata_smooth[[variable + "_smooth" + suffix, variable + "_se" + suffix]] = smooth_spline_fit_se(windowscores_oi["n_sites"] + np.random.uniform(0, 1, len(windowscores_oi)), windowscores_oi[variable], plotdata_smooth["n_sites"])
    return plotdata_smooth

# %%
plotdata_smooth = add_smooth(plotdata_smooth, "n_footprints", windowscores_oi)
plotdata_smooth = add_smooth(plotdata_smooth, "absdeltacor", windowscores_oi)
plotdata_smooth = add_smooth(plotdata_smooth, "ratio", windowscores_oi)
plotdata_smooth = add_smooth(plotdata_smooth, "lost", windowscores_oi)

# plotdata_smooth = add_smooth(plotdata_smooth, "n_footprints", windowscores_diffexp, suffix = "_diffexp")
# plotdata_smooth = add_smooth(plotdata_smooth, "absdeltacor", windowscores_diffexp, suffix = "_diffexp")
# plotdata_smooth = add_smooth(plotdata_smooth, "ratio", windowscores_diffexp, suffix = "_diffexp")

# %%
def minmax_norm(x):
    return (x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0))

idxmax = plotdata_smooth["n_footprints_smooth"].idxmax()

ratio = minmax_norm(plotdata_smooth["n_footprints_smooth"])[idxmax] / minmax_norm(plotdata_smooth[["ratio_smooth", "absdeltacor_smooth"]]).loc[idxmax].max()

# %%
fig, ax = plt.subplots(figsize = (2, 2))

ax.set_xlabel("# bound TFs (GM12878)")

smooth_suffix = ""
smooth_suffix = "_smooth"

color = "#0074D9"
ax.plot(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints" + smooth_suffix], color = color)
ax.fill_between(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints" + smooth_suffix] + plotdata_smooth["n_footprints_se"] * 1.96, plotdata_smooth["n_footprints"  + smooth_suffix] - plotdata_smooth["n_footprints_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
ax.tick_params(axis='y', labelcolor=color, color = color)
ax.spines["left"].set_edgecolor(color)
ax.spines["top"].set_visible(False)
ax.set_ylim(0, ax.get_ylim()[1] * ratio)
ax.yaxis.set_label_coords(-0.0,1.02)
ax.set_ylabel("# footprints", color = color, rotation = 0, ha = "center", va = "bottom")

color = "tomato"
variable = "absdeltacor"
ax2 = ax.twinx()
ax2.plot(plotdata_smooth["n_sites"], plotdata_smooth[variable + ""  + smooth_suffix], color = color)
ax2.fill_between(plotdata_smooth["n_sites"], plotdata_smooth[variable + "" + smooth_suffix] + plotdata_smooth[variable + "_se"] * 1.96, plotdata_smooth[variable + ""  + smooth_suffix] - plotdata_smooth[variable + "_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
ax2.yaxis.set_label_coords(1.0,1.02)
ax2.set_ylabel("$\Delta$ cor", color = color, rotation = 0, ha = "center", va = "bottom")
ax2.tick_params(axis='y', labelcolor=color, color = color)
ax2.spines["right"].set_edgecolor(color)
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_ylim(0)

color = "#2ECC40"
variable = "ratio"
ax3 = ax.twinx()
ax3.plot(plotdata_smooth["n_sites"], plotdata_smooth[variable + ""  + smooth_suffix], color = color)
ax3.fill_between(plotdata_smooth["n_sites"], plotdata_smooth[variable + "" + smooth_suffix] + plotdata_smooth[variable + "_se"] * 1.96, plotdata_smooth[variable + ""  + smooth_suffix] - plotdata_smooth[variable + "_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
ax3.set_yscale("log")
yaxis_x = 1.4
ax3.yaxis.set_label_coords(yaxis_x,1.02)
ax3.set_ylabel(r"$\frac{\mathit{Mono-}}{\mathit{TF\;footprint}}$", color = color, rotation = 0, ha = "center", va = "bottom")
ax3.spines["right"].set_position(("axes", yaxis_x))
ax3.spines["right"].set_edgecolor(color)
ax3.spines["left"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.tick_params(axis='y', labelcolor=color, color = color)
ax3.tick_params(axis='y', which='minor', labelcolor=color, color = color)
ax3.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax3.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax3.set_ylim(1)

color = "#FF851B"
variable = "lost"
ax3 = ax.twinx()
ax3.plot(plotdata_smooth["n_sites"], plotdata_smooth[variable + ""  + smooth_suffix], color = color)
ax3.fill_between(plotdata_smooth["n_sites"], plotdata_smooth[variable + "" + smooth_suffix] + plotdata_smooth[variable + "_se"] * 1.96, plotdata_smooth[variable + ""  + smooth_suffix] - plotdata_smooth[variable + "_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
yaxis_x = 1.7
ax3.yaxis.set_label_coords(yaxis_x,1.02)
ax3.set_ylabel("# fragments", color = color, rotation = 0, ha = "left", va = "bottom")
ax3.spines["right"].set_position(("axes", yaxis_x))
ax3.spines["right"].set_edgecolor(color)
ax3.spines["left"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.tick_params(axis='y', labelcolor=color, color = color)
ax3.tick_params(axis='y', which='minor', labelcolor=color, color = color)
ax3.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax3.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax3.set_ylim(0, ax3.get_ylim()[1] * ratio)

if footprints_name == "HINT":
    if filterer == "top30":
        manuscript.save_figure(fig, "7", "windowsize_footprints_top30", dpi = 300)
    else:
        manuscript.save_figure(fig, "7", "windowsize_footprints", dpi = 300)

# %%
fig, ax = plt.subplots(figsize = (2, 2))

ax.set_xlabel("# binding sites (GM12878)")

color = "#0074D9"
ax.plot(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints" + smooth_suffix], color = color)
ax.fill_between(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints" + smooth_suffix] + plotdata_smooth["n_footprints_se"] * 1.96, plotdata_smooth["n_footprints"  + smooth_suffix] - plotdata_smooth["n_footprints_se"] * 1.96, color = color, alpha = 0.2)
ax.set_ylabel(f"# footprints", color = color, rotation = 0, ha = "right", va = "center")
ax.tick_params(axis='y', labelcolor=color, color = color)
ax.spines["left"].set_edgecolor(color)
ax.spines["top"].set_visible(False)
ax.set_title(footprints_name)
ax.set_ylim(0)

manuscript.save_figure(fig, "7", f"footprints_sites_{''.join(x for x in footprints_name if x.isalnum())}", dpi = 300)


# %% [markdown]
# ### Gene-wise scoring

# %%
genescores = []
for gene in tqdm.tqdm(genes_oi):
    if gene in windowscores_genes:
        windowscores = windowscores_genes[gene].copy()
        windowscores["log1p_n_footprints"] = np.log1p(windowscores["n_footprints"])
        windowscores["log1p_n_sites"] = np.log1p(windowscores["n_sites"])
        windowscores["absdeltacor_30"] = np.abs(windowscores["deltacor_30"])
        windowscores["absdeltacor_100"] = np.abs(windowscores["deltacor_100"])
        windowscores["ratio"] = np.exp(windowscores["logratio"])
        # windowscores["ratio"] = (windowscores["n_100"]) / (windowscores["n_30"])

        windowscores_oi = windowscores.loc[~pd.isnull(windowscores["ratio"])].copy()

        score = {
            "gene":gene,
        }

        # correlation
        metrics =  ["log1p_n_footprints", "n_30", "n_100", "absdeltacor_30", "absdeltacor_100", "absdeltacor", "ratio", "logratio", "log1p_n_sites"]
        data_for_cor = windowscores[metrics].copy()

        score.update({
            "cor": data_for_cor.corr(method = "spearman"),
        })

        # r2 to predict log1p_n_sites
        # r2 = {}
        # for featureset in [("ratio", "absdeltacor"), ("ratio", ), ("absdeltacor", ), ("ratio", "absdeltacor", "n_footprints"), ("n_footprints", ), ("absdeltacor", "n_footprints", )]:
        #     import statsmodels.api as sm
        #     X = windowscores_oi[list(featureset)]
        #     X = sm.add_constant(X)
        #     y = windowscores_oi["log1p_n_sites"]
        #     model = sm.OLS(y, X)
        #     results = model.fit()
        #     r2.update({
        #         featureset: results.rsquared,
        #     })
        # score.update({
        #     "r2": r2,
        # })

        # odds
        metrics =  ["n_footprints", "n_30", "n_100", "absdeltacor_30", "absdeltacor_100", "ratio", "logratio", "n_sites"]
        bools = np.stack([windowscores_oi[k] > windowscores_oi[k].mean() for k in metrics])
        
        contingencies = np.stack(
            [
                (bools[:, None] & bools[None]).sum(-1),
                (~bools[:, None] & bools[None]).sum(-1),
                (bools[:, None] & ~bools[None]).sum(-1),
                (~bools[:, None] & ~bools[None]).sum(-1),
            ]
        )

        score.update({
            "contingencies": contingencies,
        })

        genescores.append(score)
genescores = pd.DataFrame(genescores).set_index("gene")

# %%
fig = chd.grid.Figure(chd.grid.Grid())
panel, ax = fig.main.add_under(chd.grid.Panel((3, 3)))
plotdata = pd.DataFrame(
    np.nan_to_num(np.stack(genescores.loc[genescores.index.intersection(genes_diffexp)]["cor"]), 0).mean(0),
    index = genescores["cor"].iloc[0].index,
    columns = genescores["cor"].iloc[0].index,
)
ax.set_title("Diffexp B-cell genes")
sns.heatmap(plotdata, vmin = -1, vmax = 1, cmap = "coolwarm",  annot = True, fmt = ".2f", ax = ax, cbar = False)
panel, ax = fig.main.add_right(chd.grid.Panel((3, 3)))
plotdata = pd.DataFrame(
    np.nan_to_num(np.stack(genescores["cor"]), 0).mean(0),
    index = genescores["cor"].iloc[0].index,
    columns = genescores["cor"].iloc[0].index,
)
ax.set_title("All genes")
sns.heatmap(plotdata, vmin = -1, vmax = 1, cmap = "coolwarm",  annot = True, fmt = ".2f", ax = ax, cbar = False)
ax.set_yticks([])
fig.plot()

# %%
contingencies = np.stack(genescores["contingencies"]).sum(0)
odds = pd.DataFrame((contingencies[0] * contingencies[3]) / (contingencies[1] * contingencies[2]), index = metrics, columns = metrics)
norm = mpl.colors.LogNorm(vmin = 1/8, vmax = 8)

sns.heatmap(odds, norm = norm, cmap = "coolwarm", annot = True, fmt = ".2f")

# %%
