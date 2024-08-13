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

# %%
models_path = chd.get_output() / "pred/pbmc10k/100k100k/5x5/magic/v33"
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow(
    path=models_path / "scoring" / "regionmultiwindow_100",
)
regionsizewindow = chd.models.pred.interpret.RegionSizeWindow(models_path / "scoring" / "regionsizewindow")

# %%
len(regionsizewindow.scores.keys())

    # %%
    # score = pd.DataFrame({
    #     "deltacor":regionmultiwindow.scores["deltacor"].sel_xr(gene).sel(phase = "test").mean("fold").sel(window = score.index.get_level_values("window").unique()),
    #     "effect":regionmultiwindow.scores["effect"].sel_xr(gene).sel(phase = "test").mean("fold").sel(window = score.index.get_level_values("window").unique()),
    #     "lost":regionmultiwindow.scores["lost"].sel_xr(gene).sel(phase = "test").mean("fold").sel(window = score.index.get_level_values("window").unique()),
    # })

# %%
scores = []
windowscores = []
for gene in tqdm.tqdm(regionsizewindow.scores.keys()):
    regionsizewindow.scores[gene]

    scores_gene = regionsizewindow.scores[gene]
    windows = scores_gene.coords["window"].to_pandas().str.split("_").str[0][::2]
    deltacor = pd.DataFrame(scores_gene["deltacor"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])
    lost = pd.DataFrame(scores_gene["lost"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])
    effect = pd.DataFrame(scores_gene["effect"].mean("fold").values.reshape(-1, 2), index = windows, columns = [30, 100])

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
    windowscores.append(windowscore.reset_index().assign(gene = gene))

scores = pd.concat(scores)
windowscores = pd.concat(windowscores)

# %%
len(windowscores["gene"].unique())


# %% [markdown]
# ## Get GC content
# %%
def extract_gc(fasta, chrom, start, end, strand):
    """
    Extract GC content in a region
    """

    actual_start = start
    if actual_start < 0:
        actual_start = 0
    seq = fasta.fetch(chrom, actual_start, end)
    if strand == -1:
        seq = seq[::-1]

    gc = np.isin(np.array(list(seq)), ["c", "g", "C", "G"])

    return gc


# %%
import pysam
genome = "GRCh38"
fasta_file = f"/data/genome/{genome}/{genome}.fa"

fasta = pysam.FastaFile(fasta_file)

# %%
window_contents = pd.DataFrame(index = scores.groupby(["gene", "window"]).first().index).join(regionmultiwindow.design)

# %%
window_contents["region"] = window_contents.index.get_level_values("gene")
window_contents["start"] = window_contents["window_start"]
window_contents["end"] = window_contents["window_end"]
window_contents = chd.data.peakcounts.plot.uncenter_multiple_peaks(window_contents, fragments.regions.coordinates)

# %%
gcs = []
for chrom, start, end in tqdm.tqdm(
    zip(
        window_contents.chrom,
        window_contents.start_genome,
        window_contents.end_genome,
    ),
    total=len(window_contents),
):
    gcs.append(extract_gc(fasta, chrom, start, end, 1).mean())
window_contents["gc"] = gcs

# %%
window_contents.groupby("window_start").mean()["gc"].plot()

# %% [markdown]
# ## Get windowsize scores

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
windowsize_scores["lost_ratio"] = (windowsize_scores["lost_100"] + 0.1) / (
    windowsize_scores["lost_30"] + 0.1
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

# %%
np.corrcoef(
    np.log(
        windowsize_scores.query("lost_30>0").query("lost_100>0")["lost_ratio"].abs()
    ),
    np.log(
        windowsize_scores.query("lost_30>0").query("lost_100>0")["deltacor_ratio"].abs()
    ),
)

# %%
windowsize_scores["deltacor"] = windowscores.set_index(["gene", "window"])["deltacor"]
windowsize_scores["lost"] = windowscores.set_index(["gene", "window"])["lost"]

# %%
plt.scatter(
    windowsize_scores["log_deltacor_ratio"],
    windowsize_scores["lost"],
)

# %% [markdown]
# ## GC and deltacor ratio

# %%
windowsize_scores["start"] = window_contents.loc[windowsize_scores.index, "start"]
windowsize_scores["end"] = window_contents.loc[windowsize_scores.index, "end"]

# %%
# windowsize_scores_oi = windowsize_scores
windowsize_scores_oi = windowsize_scores.query("(start < -10000) | (start > 10000)")
# windowsize_scores_oi = windowsize_scores.query("(start > -10000) | (start > 10000)")

# %%
plotdata = pd.DataFrame(
    {
        "rank_lost": windowsize_scores_oi["lost"].rank(),
        "rank_lost_ratio": windowsize_scores_oi["rank_lost_ratio"],
        "rank_deltacor_ratio": windowsize_scores_oi["rank_deltacor_ratio"],
        "gc": window_contents.loc[windowsize_scores_oi.index]["gc"],
    }
)
plotdata["bin_deltacor_ratio"] = pd.cut(
    plotdata["rank_lost"] / len(plotdata), bins=np.linspace(0, 1, 100)
)
plotdata = plotdata.groupby("bin_deltacor_ratio").agg({"gc": "mean"}).reset_index()
fig, ax = plt.subplots()
ax.scatter(plotdata["bin_deltacor_ratio"].astype(str), plotdata["gc"])

# %% [markdown]
# ## Different motifs in footprint vs submono
# %%
windowsize_scores = windowsize_scores.drop(columns = ["start", "end"]).join(window_contents)

# %%
# windowsize_scores["chr"] = promoters.loc[
#     windowsize_scores.index.get_level_values("gene")
# ]["chr"].values
# windowsize_scores["strand"] = promoters.loc[
#     windowsize_scores.index.get_level_values("gene")
# ]["strand"].values
# windowsize_scores["start"] = window_scoring.design.loc[
#     windowsize_scores.index.get_level_values("window")
# ]["window_start"].values
# windowsize_scores["end"] = window_scoring.design.loc[
#     windowsize_scores.index.get_level_values("window")
# ]["window_end"].values
# windowsize_scores["tss"] = promoters.loc[
#     windowsize_scores.index.get_level_values("gene")
# ]["tss"].values
# windowsize_scores["gstart"] = (
#     windowsize_scores["tss"]
#     + (windowsize_scores["start"] * (windowsize_scores["strand"] == 1))
#     - (windowsize_scores["end"] * (windowsize_scores["strand"] == -1))
# ).values
# windowsize_scores["gend"] = (
#     (windowsize_scores["tss"])
#     + (windowsize_scores["end"] * (windowsize_scores["strand"] == 1))
#     - (windowsize_scores["start"] * (windowsize_scores["strand"] == -1))
# ).values
# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
motifscan_counts = motifscan.count_slices(windowsize_scores)

# %%
# indptr = motifscan.indptr
# indices = motifscan.indices
# motifscan_counts = []
# for (gene, window), y in tqdm.tqdm(
#     windowsize_scores.iterrows(), total=len(windowsize_scores)
# ):
#     gene_ix = transcriptome.var.index.get_loc(gene)

#     indptr_start = (
#         (gene_ix * (promoter_window[1] - promoter_window[0]))
#         + y["start"]
#         - promoter_window[0]
#     )
#     indptr_end = (
#         (gene_ix * (promoter_window[1] - promoter_window[0]))
#         + y["end"]
#         - promoter_window[0]
#     )

#     motifindices = motifscan.indices[
#         motifscan.indptr[indptr_start] : motifscan.indptr[indptr_end + 1]
#     ]

#     motifcounts = np.bincount(motifindices, minlength=(motifscan.n_motifs))

#     motifscan_counts.append(motifcounts)
# motifscan_counts = np.stack(motifscan_counts)

# %%
deltacor_cor = np.corrcoef(
    motifscan_counts, windowsize_scores["rank_deltacor_ratio"], rowvar=False
)
lost_cor = np.corrcoef(
    motifscan_counts, windowsize_scores["rank_lost_ratio"], rowvar=False
)

# %%
motifscores = pd.DataFrame(
    {
        "deltacor_cor": deltacor_cor[-1, :-1],
        "lost_cor": lost_cor[-1, :-1],
        "motif": motifscan.motifs.index,
    }
).set_index("motif")

# %%
motifscores.sort_values("deltacor_cor", ascending=False).head(20)

# %%
motifscores.sort_values("deltacor_cor", ascending=True).head(20)

# %%
motif_oi = motifscan.motifs.index[motifscan.motifs.index.str.contains("CTCF")][0]
motif_oi_ix = motifscan.motifs.index.get_loc(motif_oi)

# %%
plotdata = pd.DataFrame(
    {
        "rank_lost_ratio": windowsize_scores["rank_lost_ratio"],
        "rank_deltacor_ratio": windowsize_scores["rank_deltacor_ratio"],
        "motifscan_counts": motifscan_counts.values[:, motif_oi_ix],
    }
)
plotdata["bin_deltacor_ratio"] = pd.cut(
    plotdata["rank_deltacor_ratio"] / len(plotdata), bins=np.linspace(0, 1, 100)
)
plotdata = (
    plotdata.groupby("bin_deltacor_ratio")
    .agg({"motifscan_counts": "mean"})
    .reset_index()
)
fig, ax = plt.subplots()
ax.scatter(plotdata["bin_deltacor_ratio"].astype(str), plotdata["motifscan_counts"])

# %%
def smooth_spline_fit(x, y, x_smooth):
    import rpy2.robjects as robjects

    r_y = robjects.FloatVector(y)
    r_x = robjects.FloatVector(x)

    r_smooth_spline = robjects.r["smooth.spline"]
    spline1 = r_smooth_spline(x=r_x, y=r_y, nknots=10)
    ySpline = np.array(
        robjects.r["predict"](spline1, robjects.FloatVector(x_smooth)).rx2("y")
    )

    return ySpline


# %%
x = window_contents.loc[windowsize_scores.index]["gc"].values
y = motifscan_counts.values[:, motif_oi_ix]

y_smooth = smooth_spline_fit(x, y, x)
residuals = y - y_smooth

# %%
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x[np.argsort(x)], y_smooth[np.argsort(x)], color="red")

# %%
fig, ax = plt.subplots()
ax.scatter(x, residuals, color="red")

# %%
import scipy.stats

x = windowsize_scores["rank_lost_ratio"]
y = residuals

lm2 = scipy.stats.linregress(x, y)
lm2

# %%
fig, ax = plt.subplots()
ax.scatter(x, y)

# %%
gc = window_contents.loc[windowsize_scores.index]["gc"].values
count = motifscan_counts.values[:, motif_oi_ix]
# outcome = windowsize_scores["effect_100"]
outcome = windowsize_scores["rank_deltacor_ratio"]
# outcome = windowsize_scores["rank_lost_ratio"]

gc_binned = pd.cut(gc, bins=np.quantile(gc, np.linspace(0, 1, 10)))

data = pd.DataFrame(
    {"gc": gc, "count": count, "outcome": outcome, "gc_binned": gc_binned}
)

contingencies = []
for bin, data_bin in data.groupby("gc_binned"):
    a = data_bin["outcome"] > data_bin["outcome"].median()
    b = data_bin["count"] > 0

    # contingency
    contingency = np.array([[sum(a & b), sum(~a & b)], [sum(a & ~b), sum(~a & ~b)]])
    contingencies.append(contingency)
    odds = np.log(scipy.stats.fisher_exact(contingency)[0])
    print(bin, odds)
contingency = np.stack(contingencies).sum(axis=0)
odds, pvalue = scipy.stats.fisher_exact(contingency)
print("total", odds, pvalue)


# %% [markdown]
# ## Compare across TFs with GC correction

# %%
def smooth_spline_fit(x, y, x_smooth):
    import rpy2.robjects as robjects

    r_y = robjects.FloatVector(y)
    r_x = robjects.FloatVector(x)

    r_smooth_spline = robjects.r["smooth.spline"]
    spline1 = r_smooth_spline(x=r_x, y=r_y, nknots=20)
    ySpline = np.array(
        robjects.r["predict"](spline1, robjects.FloatVector(x_smooth)).rx2("y")
    )

    return ySpline


def score_gc_corrected(gc, count, outcome):
    count_smooth = smooth_spline_fit(gc, count, gc)
    residuals = count - count_smooth

    lm2 = scipy.stats.linregress(outcome, residuals)

    return {
        "slope": lm2.slope,
        "r": lm2.rvalue,
        "p": lm2.pvalue,
    }


# %%
motifscores_corrected = []
gc = window_contents.loc[windowsize_scores.index]["gc"].values
outcome = windowsize_scores["rank_lost_ratio"]
# outcome = windowsize_scores["rank_deltacor_ratio"]
# outcome = windowsize_scores["effect_100"].rank()
# outcome = windowsize_scores["deltacor_100"].rank()
for motif_oi_ix in tqdm.tqdm(range(motifscan_counts.shape[1])):
    count = motifscan_counts.values[:, motif_oi_ix]
    score = score_gc_corrected(gc, count, outcome)
    score["motif"] = motifscan.motifs.index[motif_oi_ix]
    motifscores_corrected.append(score)
motifscores_corrected = pd.DataFrame(motifscores_corrected).set_index("motif")

# %%
import statsmodels.stats.multitest

motifscores_corrected["qval"] = statsmodels.stats.multitest.multipletests(
    motifscores_corrected["p"], method="fdr_bh"
)[1]
# %%
motifscores_corrected.sort_values("qval", ascending=True).head(20)
# %%
motifscores_corrected.query("qval < 0.05").sort_values("r", ascending=True).head(100)

# %%
motifscores_corrected["in_diff"] = motifscan.motifs["HUMAN_gene_symbol"].isin(
    transcriptome.var["symbol"]
)

# %%
motifscores_corrected.query("in_diff").query("qval < 0.1").sort_values(
    "r", ascending=True
).head(100)

# %%
motifscores_corrected.query("in_diff").query("qval < 0.1").sort_values(
    "r", ascending=True
)["r"].plot()

# %%
transcriptome.gene_id(motifscan.motifs["HUMAN_gene_symbol"], optional = True)

# %%
motifscan.motifs["gene"] = transcriptome.gene_id(motifscan.motifs["HUMAN_gene_symbol"], found = True).values

# %%
# motifscores_corrected["dispersions_norm"] = (
#     transcriptome.var["dispersions_norm"]
#     .reindex(motifscan.motifs["gene"])[
#         motifscan.motifs.loc[motifscores_corrected.index, "gene"]
#     ]
#     .values
# )

# %%
fig, ax = plt.subplots(figsize=(2, 4))
plotdata = (
    motifscores_corrected.query("in_diff")
    .query("qval < 0.1")
    .sort_values("r", ascending=True)
    .reset_index()
)
plotdata["label"] = transcriptome.symbol(
    motifscan.motifs.loc[plotdata["motif"], "gene"]
).values
plotdata["y"] = np.arange(len(plotdata))

for ix, row in plotdata.iterrows():
    ax.plot(
        [row["r"], row["slope"]],
        [row["y"], row["y"]],
        color="black",
        alpha=0.2,
        linewidth=0.5,
    )
    ax.scatter(row["r"], row["y"], color="#333", s=5)
    # ax.scatter(row["r"], row["y"], color=cmap(norm(row["dispersions_norm"])), s=5)
    # ax.scatter(row["slope"], row["y"], color="blue", alpha=0.5)
    ax.text(
        row["r"],
        row["y"],
        "  " + row["label"] + "  ",
        fontsize=8,
        va="center",
        ha="right" if row["r"] < 0 else "left",
    )
    # ax.text(row["slope"], row["y"], row["label"], fontsize=6, va="center", ha="left")
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([])
# %%
# sc.pl.umap(transcriptome.adata, color=transcriptome.gene_id(["ZEB1"]))

# %% [markdown]
# ## Compare to external footprinting comparison
# %%
import pathlib
if not pathlib.Path("41592_2016_BFnmeth3772_MOESM205_ESM.xlsx").exists():
    # !wget https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.3772/MediaObjects/41592_2016_BFnmeth3772_MOESM205_ESM.xlsx
# %%
# !pip install openpyxl
external_data = pd.read_excel("41592_2016_BFnmeth3772_MOESM205_ESM.xlsx", sheet_name="Supplementary Dataset 1a", skiprows=2, engine="openpyxl")

# %%
external_data.loc[external_data["FACTOR"] == "PU1", "FACTOR"] = "SPI1"

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col=0)

# %%
external_data["gene"] = transcriptome.var.reset_index().set_index("symbol")["gene"].reindex(external_data["FACTOR"].unique())[external_data["FACTOR"]].values
external_data["gene"] = genes.reset_index().set_index("symbol").groupby("symbol").first()["gene"].reindex(external_data["FACTOR"].unique())[external_data["FACTOR"]].values
1-pd.isnull(external_data["gene"]).mean()

# %%
external_data["motif"] = motifscan.motifs.reset_index().set_index("gene")["motif"].reindex(external_data["gene"]).values

# %%
external_motif_scores = external_data.groupby("motif").mean()
external_motif_scores["gene"] = motifscan.motifs.reset_index().set_index("motif")["gene"].reindex(external_motif_scores.index).values

# %%
# filter on variable genes in our dataset
# doesn't make sense to consider non-variable TFs
external_motif_scores = external_motif_scores[external_motif_scores["gene"].isin(transcriptome.var.index)].copy()

# %%
motifscores_corrected["protection"] = external_motif_scores["HINT_AUC_10"].reindex(motifscores_corrected.index).values
# %%
plt.scatter(motifscores_corrected["protection"], motifscores_corrected["r"])
# %%
dat = external_motif_scores
# %%
external_motif_scores["r"] = motifscores_corrected["r"].reindex(external_motif_scores.index).values

# %%
external_motif_scores.corr()["r"].sort_values().head(10)
# external_motif_scores.corr()["r"].sort_values().tail(10)

# %%
external_columns_oi = [
    col for col in external_motif_scores.columns if (col.endswith("AUPR_ALL")) and not col.startswith("PWM")
]
external_columns_oi = [
    "Centipede_AUPR_ALL",
]

# %%
fig, ax = plt.subplots(figsize=(2, 2))
plotdata = external_motif_scores
ax.scatter(
    plotdata["r"],
    external_motif_scores[external_columns_oi].mean(1)
)

# %%
np.corrcoef(
    external_motif_scores["r"],
    external_motif_scores[external_columns_oi].mean(1)
)

# %% [markdown]
# ## Compare to overall enrichment in predictive regions

# %%
motifscores_deltacor = []
gc = window_contents.loc[windowsize_scores.index]["gc"].values
windowsize_scores["effect"] = windowscores["effect"].values
# outcome = windowsize_scores["effect"].rank()
outcome = windowsize_scores["deltacor"].rank()
# outcome = windowsize_scores["deltacor_100"].rank()
# outcome = windowsize_scores["effect_100"].rank()
# outcome = windowsize_scores["rank_deltacor_ratio"]
for motif_oi_ix in tqdm.tqdm(range(motifscan_counts.shape[1])):
    count = motifscan_counts.values[:, motif_oi_ix]
    score = score_gc_corrected(gc, count, outcome)
    score["motif"] = motifscan.motifs.index[motif_oi_ix]
    motifscores_deltacor.append(score)
motifscores_deltacor = pd.DataFrame(motifscores_deltacor).set_index("motif")

# %%
motifscores_deltacor.sort_values("r").head(20)
# %%
plotdata = pd.concat(
    [
        motifscores_deltacor.rename(columns = lambda x: x + "_deltacor"),
        motifscores_corrected.rename(columns = lambda x: x + "_corrected"),
    ],
    axis=1,
)
plotdata["gene"] = motifscan.motifs.reset_index().set_index("motif")["gene"].reindex(plotdata.index).values
plotdata = plotdata.groupby("gene").first().reset_index()
plotdata["variable"] = plotdata["gene"].isin(transcriptome.var.index)
plotdata = plotdata.loc[plotdata["variable"]].copy()
plotdata = plotdata.loc[plotdata["r_deltacor"] < (plotdata["r_deltacor"].mean() + 2 * plotdata["r_deltacor"].std())]

# %%
plotdata

# %%
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.scatter(
    -plotdata["r_deltacor"],
    plotdata["r_corrected"],
    s = 2,
    color = ["red" if row["p_corrected"] < 0.1 else "black" for ix, row in plotdata.iterrows()]
)

plotdata["oi"] = (
    False
    # | (plotdata["r_corrected"].abs() > np.quantile(plotdata["r_corrected"].abs(), 0.80)) 
    | (plotdata["r_deltacor"] < np.quantile(plotdata["r_deltacor"], 0.1)) 
    | (plotdata["gene"] == transcriptome.gene_id("CTCF"))
    | (plotdata["r_corrected"] < plotdata["r_corrected"].sort_values().iloc[4])
    | (plotdata["r_corrected"] > plotdata["r_corrected"].sort_values(ascending = False).iloc[20])
    )

r_deltacor_cutoff = np.quantile(plotdata["r_deltacor"], 0.3)

texts = []
for ix, row in plotdata.query("oi").iterrows():
    text = ax.text(-row["r_deltacor"], row["r_corrected"], transcriptome.symbol(row["gene"]), fontsize=7, va="center", ha="left")
    texts.append(text)

ax.axhline(0, color="#333", dashes = (2, 2))
ax.axvline(0, color="#333", dashes = (2, 2))

ax.set_xlabel("Enrichment in predictive regions\nslope of # motifs = f(Δcor)")
frac = r"$\log{\frac{\mathit{Mono-}}{\mathit{TF\;footprint}}}$"
ax.set_ylabel("Enrichment in Mono- vs TF footprint\nslope of # motifs = f(" + frac + ")")

# transform of data in x and axis coords in y
transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.annotate("", (1.02, 0), (1.02, ax.get_ylim()[1]), arrowprops=dict(arrowstyle = "<-", ec = "#333"), xycoords = transform, textcoords = transform)
ax.annotate("", (1.02, 0), (1.02, ax.get_ylim()[0]), arrowprops=dict(arrowstyle = "<-", ec = "#333"), xycoords = transform, textcoords = transform)
ax.text(1.05, ax.get_ylim()[1]/2, "Prefers\nMono-",rotation = 0, va = "center", ha = "left", transform = transform)
ax.text(1.05, ax.get_ylim()[0]/2, "Prefers\nTF footprint",rotation = 0, va = "center", ha = "left", transform = transform)

transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.annotate("", (0, 1.02), (ax.get_xlim()[1], 1.02), arrowprops=dict(arrowstyle = "<-", ec = "#333"), xycoords = transform, textcoords = transform)
ax.annotate("", (0, 1.02), (ax.get_xlim()[0], 1.02), arrowprops=dict(arrowstyle = "<-", ec = "#333"), xycoords = transform, textcoords = transform)
ax.text(ax.get_xlim()[1]/2, 1.05, "enriched",rotation = 0, va = "bottom", ha = "center", transform = transform)
ax.text(ax.get_xlim()[0]/2, 1.05, "depleted",rotation = 0, va = "bottom", ha = "center", transform = transform)
ax.text(0, 1.12, "In predictive regions",rotation = 0, va = "bottom", ha = "center", transform = transform)

import adjustText
annot = adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='#AAA', zorder = 0))

# make sure all annotations are drawn under the text
# and add a small border around the text
for text in texts:
    text.zorder = 10
    text.set_path_effects([mpl.patheffects.withStroke(linewidth=1.2, foreground='#FFFFFFCC')])

manuscript.save_figure(fig, "7", "motif_enrichment_ratio")

# %%
genes_oi = plotdata.query("r_deltacor < -0.015").sort_values("r_corrected")["gene"][:10]
sc.pl.heatmap(transcriptome.adata, transcriptome.symbol(genes_oi), "celltype", dendrogram=True, figsize=(4, 4))
sc.pl.stacked_violin(transcriptome.adata, transcriptome.symbol(genes_oi), groupby='celltype', dendrogram=True, gene_symbols = "symbol")

# %%
genes_oi = plotdata.query("r_deltacor < -0.015").sort_values("r_corrected", ascending = False)["gene"][:20]
sc.pl.heatmap(transcriptome.adata, transcriptome.symbol(genes_oi), "celltype", dendrogram=True, figsize=(4, 4), gene_symbols = "symbol")
sc.pl.stacked_violin(transcriptome.adata, transcriptome.symbol(genes_oi), groupby='celltype', dendrogram=True, gene_symbols = "symbol")
