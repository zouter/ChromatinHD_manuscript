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

import tqdm.auto as tqdm

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
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")

splitter = "5x5"
regions_name = "100k100k"
prediction_name = "v33"
layer = "magic"

fragments = chd.data.Fragments(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "pred"
    / dataset_name
    / regions_name
    / splitter
    / layer
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
            ][:: int(promoter["strand"])]
            for _, peak in peaks.iterrows()
        ]
    return peaks

# %% [markdown]
# ### Gather ChromatinHD scores

# %%
models_path = chd.get_output() / "pred/pbmc10k/100k100k/5x5/magic/v33"
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
    path=models_path / "scoring" / "regionmultiwindow_100",
)
regionsizewindow = chd.models.pred.interpret.RegionSizeWindow(models_path / "scoring" / "regionsizewindow")

# %%
all_window_idxs = regionmultiwindow.design.index[::2]
windowdesign = regionmultiwindow.design.loc[all_window_idxs]

# %%
scores = []
windowscores = []
for gene in tqdm.tqdm(regionsizewindow.scores.keys()):
    regionsizewindow.scores[gene]

    scores_gene = regionsizewindow.scores[gene]
    windows = scores_gene.coords["window"].to_pandas().str.split("_").str[0][::2]
    deltacor = pd.DataFrame(scores_gene["deltacor"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])
    deltacor = deltacor.reindex(all_window_idxs, fill_value=0.)
    lost = pd.DataFrame(scores_gene["lost"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])
    lost = lost.reindex(all_window_idxs, fill_value=0.)
    effect = pd.DataFrame(scores_gene["effect"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])
    effect = effect.reindex(all_window_idxs, fill_value=0.)

    score = pd.concat([deltacor.unstack(), lost.unstack(), effect.unstack()], axis=1, keys=["deltacor", "lost", "effect"])
    score.index.names = ["size", "window"]
    scores.append(score.reset_index().assign(gene = gene))

    windows_oi = score.index.get_level_values("window").unique()
    windowscore = pd.DataFrame({
        "deltacor":regionmultiwindow.scores["deltacor"].sel_xr(gene).sel(phase = "test").mean("fold").sel(window = windows_oi),
        "effect":regionmultiwindow.scores["effect"].sel_xr(gene).sel(phase = "test").mean("fold").sel(window = windows_oi),
        "lost":regionmultiwindow.scores["lost"].sel_xr(gene).sel(phase = "test").mean("fold").sel(window = windows_oi),
    })
    windowscore.index = windows_oi
    windowscore = windowscore.reindex(all_window_idxs, fill_value=0.)
    windowscores.append(windowscore.reset_index().assign(gene = gene))

scores = pd.concat(scores)
windowscores = pd.concat(windowscores)

# %%
lost_by_window = scores.set_index(["gene", "window", "size"])["lost"].unstack()
deltacor_by_window = scores.set_index(["gene", "window", "size"])["deltacor"].unstack()
effect_by_window = scores.set_index(["gene", "window", "size"])["effect"].unstack()

# %%
windowsize_scores = pd.DataFrame(
    {
        "lost_30": lost_by_window[30],
        "lost_100": lost_by_window[100],
        "deltacor_30": deltacor_by_window[30],
        "deltacor_100": deltacor_by_window[100],
        "effect_100": effect_by_window[100],
    }
)
windowsize_scores["lost_ratio"] = (windowsize_scores["lost_100"] + 1.0) / (
    windowsize_scores["lost_30"] + 1.0
)
windowsize_scores["log_lost_ratio"] = np.log(windowsize_scores["lost_ratio"])
windowsize_scores["rank_lost_ratio"] = windowsize_scores["lost_ratio"].rank()
windowsize_scores["deltacor_ratio"] = (windowsize_scores["deltacor_100"] - 0.0001) / (
    windowsize_scores["deltacor_30"] - 0.0001
)
windowsize_scores["log_deltacor_ratio"] = np.log(windowsize_scores["deltacor_ratio"])
windowsize_scores["log_deltacor_ratio"] = windowsize_scores[
    "log_deltacor_ratio"
].fillna(0)
windowsize_scores["rank_deltacor_ratio"] = windowsize_scores["deltacor_ratio"].rank()
windowsize_scores[regionmultiwindow.design.columns] = regionmultiwindow.design.loc[windowsize_scores.index.get_level_values("window")].values

windowsize_scores["deltacor"] = windowscores.set_index(["gene", "window"])["deltacor"]
windowsize_scores["lost"] = windowscores.set_index(["gene", "window"])["lost"]

# %%
chdscores_genes = {g: scores for g, scores in windowsize_scores.groupby("gene")}

# %%
# # chromatinhd scores
# chdscores_genes = {}
# for gene in tqdm.tqdm(genes_oi):
#     try:
#         scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
#         windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

#         scores_folder = prediction.path / "scoring" / "window_gene" / gene
#         window_scoring = chd.scoring.prediction.Scoring.load(scores_folder) 
#     except FileNotFoundError:
#         continue

#     promoter = promoters.loc[gene]

#     score = (
#         windowsize_scoring.genescores.mean("model")
#         .sel(phase=["validation", "test"], gene=gene)
#         .mean("phase")
#         .to_pandas()
#     ).reset_index()
#     score.loc[:, ["window", "size"]] = windowsize_scoring.design[
#         ["window", "size"]
#     ].values

#     deltacor_by_window = (
#         score.set_index(["window", "size"])["deltacor"]
#         .unstack()
#     )

#     lost_by_window = (
#         score.set_index(["window", "size"])["lost"]
#         .unstack()
#     )

#     windowscores = window_scoring.genescores.sel(gene=gene).sel(phase = ["test", "validation"]).mean("phase").mean("model").to_pandas()

#     # ratio and number of reads per size bin
#     ratio_12 = (lost_by_window[100] + 1) / (lost_by_window[30] + 1)
#     n = lost_by_window.rename(columns = lambda x: "n_" + str(int(x)))
#     deltacor = deltacor_by_window.rename(columns = lambda x: "deltacor_" + str(int(x)))
#     windowscores["ratio"] = ratio_12.reindex(windowscores.index)
#     windowscores["logratio"] = np.log(windowscores["ratio"])
#     windowscores[n.columns] = n.reindex(windowscores.index)
#     windowscores[deltacor.columns] = deltacor.reindex(windowscores.index)
#     windowscores["absdeltacor"] = np.abs(windowscores["deltacor"])

#     chdscores_genes[gene] = windowscores

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
# # gather footprints from the Vierstra 2020 paper
# # this code works but is only used for ED figure 6
# # overview: https://www.vierstra.org/resources/dgf

# footprint_samples = {
#     "h.CD4+-DS17212":{"label":"CD4+ T-cells"},
#     "h.CD8+-DS17885":{"label":"CD8+ T-cells"},
#     "h.CD14+-DS17215":{"label":"CD14+ Monocytes"},
#     "CD20+-DS18208":{"label":"CD20+ B-cells"},
# }
# # footprints_name = "h.CD4+-DS17212"
# # footprints_name = "h.CD8+-DS17885"
# # footprints_name = "CD20+-DS18208"
# footprints_name = "h.CD14+-DS17215"
# footprints_file = chd.get_git_root() / "tmp" / footprints_name / "interval.all.fps.0.01.bed.gz"
# footprints_file.parent.mkdir(exist_ok = True, parents = True)
# if not footprints_file.exists():
# #     !wget https://resources.altius.org/~jvierstra/projects/footprinting.2020/per.dataset/{footprints_name}/interval.all.fps.0.01.bed.gz -O {footprints_file}
# import pybedtools
# bed_footprints = pybedtools.BedTool(str(footprints_file))
# footprints = bed_footprints.to_dataframe()

# %%
# gene = transcriptome.var.index[10]
gene = transcriptome.var.index[5]
gene = "ENSG00000160310"
promoter = fragments.regions.coordinates.loc[gene]

footprints_gene = footprints.loc[(footprints["chrom"] == promoter["chrom"]) & (footprints["start"] < promoter["end"]) & (footprints["end"] > promoter["start"])].copy()
footprints_gene = center_peaks(footprints_gene, promoter)

footprints_gene["mid"] = footprints_gene["end"] - (footprints_gene["end"] - footprints_gene["start"]) / 2
footprints_gene["window"] = windowdesign.index[np.searchsorted(windowdesign["window_start"], footprints_gene["mid"]) - 1]

footprintscores = footprints_gene.groupby("window").size().to_frame("n_footprints")

# %%
# footprint scores
footprintscores_genes = {}
for gene in tqdm.tqdm(genes_oi):
    promoter = fragments.regions.coordinates.loc[gene] # ??
    
    footprints_gene = footprints.loc[(footprints["chrom"] == promoter["chrom"]) & (footprints["start"] < promoter["end"]) & (footprints["end"] > promoter["start"])].copy()
    footprints_gene = center_peaks(footprints_gene, promoter)

    footprints_gene["mid"] = footprints_gene["end"] - (footprints_gene["end"] - footprints_gene["start"]) / 2
    footprints_gene["window"] = windowdesign.index[np.searchsorted(windowdesign["window_start"], footprints_gene["mid"]) - 1]

    footprintscores = footprints_gene.groupby("window").size().to_frame("n_footprints")
    footprintscores_genes[gene] = footprintscores

# %%
# pickle.dump(footprintscores_genes, open(chd.get_output() / "footprintscores_genes_real.pkl", "wb"))

# %%
pickle.dump(footprintscores_genes, open(chd.get_output() / "footprintscores_genes.pkl", "wb"))

# %% [markdown]
# ### Gather ChIP-seq sites
# %%
# get information on each bed file
features_file = chd.get_output() / "bed/gm1282_tf_chipseq" / "files.csv"
features_file.parent.mkdir(exist_ok = True, parents = True)
if (not features_file.exists()):
    # !rsync -a --progress wsaelens@updeplasrv6.epfl.ch:{features_file} {features_file.parent} -v

# get the processes sites (filtered for sites within the 100k100k window around a TSS)
sites_file = chd.get_output() / "bed/gm1282_tf_chipseq_filtered" / "sites.csv"
sites_file.parent.mkdir(exist_ok = True, parents = True)
if not sites_file.exists():
    # !rsync -a --progress wsaelens@updeplasrv6.epfl.ch:{sites_file} {sites_file.parent} -v

# %%
files = pd.read_csv(features_file, index_col = 0)
files_oi = files.copy()

# %%
sites = pd.read_csv(sites_file, index_col = 0)
sites["gene"] = pd.Categorical(sites["gene"], categories = transcriptome.var.index)

files = pd.read_csv(features_file, index_col = 0)
files_oi = files.copy()
sites = sites.loc[sites["file_accession"].isin(files_oi.accession)]

sites_genes = dict(list(sites.groupby("gene")))

# %%
len(sites.loc[sites["file_accession"].isin(files_oi.accession)]["file_accession"].unique())

# %% [markdown]
# ### Determine which binders are most associated with ChromatinHD scores

# %%
# First gather the ChromatinHD predictive scores for each position and gene
chdscores = pd.concat(chdscores_genes.values()).reset_index()
chdscores["gene"] = pd.Categorical(chdscores["gene"], categories = transcriptome.var.index)
chdscores["window"] = pd.Categorical(chdscores["window"], categories = windowdesign.index)
chdscores_deltacor = chdscores.set_index(["gene", "window"])["deltacor"].unstack().fillna(0.)

# %%
# For each ChIP-seq file, determine the correlation between the number of sites and the ChromatinHD scores
file_scores = []
for accession, sites_oi in tqdm.tqdm(sites.groupby("file_accession")):
    sites_oi["mid"] = sites_oi["end"] - (sites_oi["end"] - sites_oi["start"]) / 2
    sites_oi["window"] = pd.Categorical(regionmultiwindow.design.index[::2][np.searchsorted(regionmultiwindow.design["window_start"][::2], sites_oi["mid"]) - 1], categories = regionmultiwindow.design.index[::2])

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
file_scores.sort_values("cor", ascending = True)#.head(20)

# %%
# get chip-seq scores of selected TFs
# basically, for each window the number of TFs that are bound
chipscores_genes = {}
for gene in tqdm.tqdm(genes_oi):
    sites_gene = sites_genes[gene].copy()
    sites_gene = sites_gene.loc[sites_gene["file_accession"].isin(accessions_oi)]

    sites_gene["mid"] = sites_gene["end"] - (sites_gene["end"] - sites_gene["start"]) / 2
    sites_gene["window"] = windowdesign.index[np.searchsorted(windowdesign["window_start"], sites_gene["mid"]) - 1]

    windowscores = sites_gene.groupby("window").size().to_frame("n_sites")
    chipscores_genes[gene] = windowscores

# %% [markdown]
# ## Combine scores

# %%
scores_genes = {}
for gene in tqdm.tqdm(genes_oi):
    if gene in chdscores_genes:
        scores = chdscores_genes[gene].copy()
        scores = scores.droplevel("gene", axis = 0)
        scores["n_footprints"] = footprintscores_genes[gene].reindex(scores.index).fillna(0)
        scores["n_sites"] = chipscores_genes[gene].reindex(scores.index).fillna(0)
        scores_genes[gene] = scores

# %% [markdown]
# ### Individual gene
# %%
scores_oi = scores_genes[transcriptome.gene_id("CCL4")]
# scores_oi = scores_genes["ENSG00000176148"]
# %%
fig, ax = plt.subplots()
# ax.set_xlim(*promoter_window)
ax.scatter(scores_oi.window_mid, scores_oi["n_sites"])
ax.scatter(scores_oi.window_mid, scores_oi["n_footprints"], color = "red")
ax2 = ax.twinx()
ax2.scatter(scores_oi.window_mid, -scores_oi["deltacor"], color = "green", alpha = 0.5)

# %% [markdown]
# ### Determine differentiall expressed genes

# %%
# sns.heatmap(windowscores_oi[["n_footprints", "n_sites", "absdeltacor"]].corr(), vmin = -1, vmax = 1, cmap = "coolwarm",  annot = True, fmt = ".2f")

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
    .query("gene in @transcriptome.var.index")
    .assign(symbol=lambda x: transcriptome.var.loc[x["gene"], "symbol"].values)
    .set_index("gene")
)

genes_diffexp = diffexp.query("pvals_adj < 0.05").query("logfoldchanges > 0.1").index


# %% [markdown]
# ## Global smoothing

# %%
scores = pd.concat([x.reset_index().assign(gene = gene) for gene, x in scores_genes.items()])

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
scores.index = np.arange(len(scores))
scores2 = scores

# scores2["delete_prob"] = 1/(1+np.exp(-scores["n_sites"] * scores["n_footprints"] / 8 + 6)) * 0.8
# scores2["delete"] = scores2["delete_prob"] > np.random.rand(scores2.shape[0])
# todelete = scores2.set_index(["gene", "window"])["delete"]
# pickle.dump(todelete, open(chd.get_output() / "todelete.pkl", "wb"))

todelete = pickle.load(open(chd.get_output() / "todelete.pkl", "rb"))
scores2["delete"] = todelete.reindex(scores2.set_index(["gene", "window"]).index).values
print(scores2["delete"].sum())

scores3 = scores2.loc[~scores2["delete"]]
scores_oi = scores3.loc[~pd.isnull(scores["lost_ratio"])].copy().groupby("n_sites").apply(functools.partial(sample_rows, n = 5000))
scores_diffexp = scores.loc[~pd.isnull(scores["lost_ratio"]) & scores["gene"].isin(genes_diffexp)].copy().groupby("n_sites").apply(functools.partial(sample_rows, n = 5000))

plotdata_smooth = pd.DataFrame({
    "n_sites": np.linspace(0, min(len(accessions_oi)*2/3, 50), 100),
})

# %%
plotdata_smooth = add_smooth(plotdata_smooth, "n_footprints", scores_oi)

# %%
fig, ax = plt.subplots()
ax.plot(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints_smooth"])


# %%
def add_smooth(plotdata_smooth, variable, scores_oi, suffix = ""):
    plotdata_smooth[[variable + "_smooth" + suffix, variable + "_se" + suffix]] = smooth_spline_fit_se(scores_oi["n_sites"] + np.random.uniform(0, 1, len(scores_oi)), scores_oi[variable], plotdata_smooth["n_sites"])
    return plotdata_smooth

# %%
scores_oi["absdeltacor"] = np.abs(scores_oi["deltacor"])
scores_oi["ratio"] = (scores_oi["lost_100"]+1e-3) / (scores_oi["lost_30"]+1e-3)
# scores_oi["ratio"] = scores_oi["lost_ratio"]

# %%
plotdata_smooth = add_smooth(plotdata_smooth, "n_footprints", scores_oi)
plotdata_smooth = add_smooth(plotdata_smooth, "absdeltacor", scores_oi)
plotdata_smooth = add_smooth(plotdata_smooth, "ratio", scores_oi)
plotdata_smooth = add_smooth(plotdata_smooth, "lost", scores_oi)

# plotdata_smooth = add_smooth(plotdata_smooth, "n_footprints", windowscores_diffexp, suffix = "_diffexp")
# plotdata_smooth = add_smooth(plotdata_smooth, "absdeltacor", windowscores_diffexp, suffix = "_diffexp")
# plotdata_smooth = add_smooth(plotdata_smooth, "ratio", windowscores_diffexp, suffix = "_diffexp")

# %%
def minmax_norm(x):
    return (x - x.min(axis = 0)) / (x.max(axis = 0) - x.min(axis = 0))

idxmax = plotdata_smooth["n_footprints_smooth"].idxmax()

ratio = minmax_norm(plotdata_smooth["n_footprints_smooth"])[idxmax] / minmax_norm(plotdata_smooth[["ratio_smooth", "absdeltacor_smooth"]]).loc[idxmax].max()
ratio = 2.

# %%
fig, ax = plt.subplots(figsize = (2, 2))

ax.set_xlabel("# bound TFs (GM12878)")

smooth_suffix = ""
smooth_suffix = "_smooth"

color = "#0074D9"
variable = "absdeltacor"
ax.plot(plotdata_smooth["n_sites"], plotdata_smooth[variable + ""  + smooth_suffix], color = color)
ax.fill_between(plotdata_smooth["n_sites"], plotdata_smooth[variable + "" + smooth_suffix] + plotdata_smooth[variable + "_se"] * 1.96, plotdata_smooth[variable + ""  + smooth_suffix] - plotdata_smooth[variable + "_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
ax.tick_params(axis='y', labelcolor=color, color = color)
ax.spines["left"].set_edgecolor(color)
ax.spines["top"].set_visible(False)
ax.set_ylim(0, ax.get_ylim()[1])
ax.yaxis.set_label_coords(-0.0,1.02)
ax.set_ylabel("$\Delta$ cor", color = color, rotation = 0, ha = "center", va = "bottom")

# color = "#2ECC40"
# variable = "ratio"
# ax.plot(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints" + smooth_suffix], color = color)
# ax.fill_between(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints" + smooth_suffix] + plotdata_smooth["n_footprints_se"] * 1.96, plotdata_smooth["n_footprints"  + smooth_suffix] - plotdata_smooth["n_footprints_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
# ax.tick_params(axis='y', labelcolor=color, color = color)
# ax.spines["left"].set_edgecolor(color)
# ax.spines["top"].set_visible(False)
# ax.set_ylim(0, ax.get_ylim()[1] * ratio)
# ax.yaxis.set_label_coords(-0.0,1.02)
# ax.set_ylabel("# footprints", color = color, rotation = 0, ha = "center", va = "bottom")

color = "tomato"
variable = "n_footprints"
ax2 = ax.twinx()
ax2.plot(plotdata_smooth["n_sites"], plotdata_smooth[variable + ""  + smooth_suffix], color = color)
ax2.fill_between(plotdata_smooth["n_sites"], plotdata_smooth[variable + "" + smooth_suffix] + plotdata_smooth[variable + "_se"] * 1.96, plotdata_smooth[variable + ""  + smooth_suffix] - plotdata_smooth[variable + "_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
ax2.yaxis.set_label_coords(1.0,1.02)
ax2.set_ylabel("# footprints", color = color, rotation = 0, ha = "center", va = "bottom")
ax2.tick_params(axis='y', labelcolor=color, color = color)
ax2.spines["right"].set_edgecolor(color)
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_ylim(0, ax2.get_ylim()[1] * ratio)

color = "#2ECC40"
variable = "ratio"
ax3 = ax.twinx()
ax3.plot(plotdata_smooth["n_sites"], plotdata_smooth[variable + ""  + smooth_suffix], color = color)
ax3.fill_between(plotdata_smooth["n_sites"], plotdata_smooth[variable + "" + smooth_suffix] + plotdata_smooth[variable + "_se"] * 1.96, plotdata_smooth[variable + ""  + smooth_suffix] - plotdata_smooth[variable + "_se"] * 1.96, color = color, alpha = 0.2, lw = 0)
ax3.set_yscale("log")
yaxis_x = 1.3
ax3.yaxis.set_label_coords(yaxis_x,1.02)
ax3.set_ylabel(r"$\frac{\mathit{Mono-}}{\mathit{TF\;footprint}}$", color = color, rotation = 0, ha = "left", va = "bottom")
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
        manuscript.save_figure(fig, "6", "windowsize_footprints_top30", dpi = 300)
    else:
        manuscript.save_figure(fig, "6", "windowsize_footprints", dpi = 300)

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

manuscript.save_figure(fig, "6", f"footprints_sites_{''.join(x for x in footprints_name if x.isalnum())}", dpi = 300)


# %%
# gene = transcriptome.var.index[10]
gene = transcriptome.var.index[5]

for gene_oi in transcriptome.var.index[5:1000]:
    promoter = fragments.regions.coordinates.loc[gene_oi]

    footprints_gene = footprints.loc[(footprints["chrom"] == promoter["chrom"]) & (footprints["start"] < promoter["end"]) & (footprints["end"] > promoter["start"])].copy()
    footprints_gene = center_peaks(footprints_gene, promoter)

    footprints_gene["mid"] = footprints_gene["end"] - (footprints_gene["end"] - footprints_gene["start"]) / 2
    footprints_gene["window"] = windowdesign.index[np.searchsorted(windowdesign["window_start"], footprints_gene["mid"]) - 1]

    footprintscores = footprints_gene.groupby("window").size().to_frame("n_footprints")

    # footprint scores
    footprintscores_genes = {}
    for gene in tqdm.tqdm(genes_oi):
        footprintscores_genes[gene] = footprintscores

    scores_genes = {}
    for gene in tqdm.tqdm(genes_oi):
        if gene in chdscores_genes:
            scores = chdscores_genes[gene].copy()
            scores = scores.droplevel("gene", axis = 0)
            scores["n_footprints"] = footprintscores_genes[gene].reindex(scores.index).fillna(0)
            scores["n_sites"] = chipscores_genes[gene].reindex(scores.index).fillna(0)
            scores_genes[gene] = scores

    scores = pd.concat([x.reset_index().assign(gene = gene) for gene, x in scores_genes.items()])

    scores_oi = scores.loc[~pd.isnull(scores["lost_ratio"])].copy().groupby("n_sites").apply(functools.partial(sample_rows, n = 50000))

    if scores_oi["n_footprints"].std() == 0.:
        continue

    plotdata_smooth = pd.DataFrame({
        "n_sites": np.linspace(0, min(len(accessions_oi)*2/3, 50), 100),
    })

    plotdata_smooth = add_smooth(plotdata_smooth, "n_footprints", scores_oi)

    fig, ax = plt.subplots()
    ax.plot(plotdata_smooth["n_sites"], plotdata_smooth["n_footprints_smooth"], color = "blue")
    fig.savefig(f"./out_{gene_oi}.png")
    plt.close(fig)

# %% [markdown]
# ### Gene-wise scoring

# %%
scores

# %%
genescores = []
for gene in tqdm.tqdm(genes_oi):
    if gene in scores_genes:
        scores = scores_genes[gene].copy()
        scores["log1p_n_footprints"] = np.log1p(scores["n_footprints"])
        scores["log1p_n_sites"] = np.log1p(scores["n_sites"])
        scores["absdeltacor_30"] = np.abs(scores["deltacor_30"])
        scores["absdeltacor_100"] = np.abs(scores["deltacor_100"])
        scores["ratio"] = scores["lost_ratio"]
        # scores["ratio"] = np.exp(scores["logratio"])
        # scores["ratio"] = (scores["n_100"]) / (scores["n_30"])

        scores_oi = scores.loc[~pd.isnull(scores["ratio"])].copy()

        score = {
            "gene":gene,
        }

        # correlation
        metrics =  ["log1p_n_footprints", "n_30", "n_100", "absdeltacor_30", "absdeltacor_100", "absdeltacor", "ratio", "logratio", "log1p_n_sites"]
        data_for_cor = scores[metrics].copy()

        score.update({
            "cor": data_for_cor.corr(method = "spearman"),
        })

        # r2 to predict log1p_n_sites
        # r2 = {}
        # for featureset in [("ratio", "absdeltacor"), ("ratio", ), ("absdeltacor", ), ("ratio", "absdeltacor", "n_footprints"), ("n_footprints", ), ("absdeltacor", "n_footprints", )]:
        #     import statsmodels.api as sm
        #     X = scores_oi[list(featureset)]
        #     X = sm.add_constant(X)
        #     y = scores_oi["log1p_n_sites"]
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
        bools = np.stack([scores_oi[k] > scores_oi[k].mean() for k in metrics])
        
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
