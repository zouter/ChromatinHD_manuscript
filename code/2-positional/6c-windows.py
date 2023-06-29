# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

import matplotlib.pyplot as plt
import matplotlib as mpl

import xarray as xr

import seaborn as sns

sns.set_style("ticks")

import scanpy as sc

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# fragments
splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
outcome_source = "counts"
prediction_name = "v20_initdefault"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])
# outcome_source = "magic"
# prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
outcome_source = "magic"
prediction_name = "v20"

promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# create design to run
from design import get_design, get_folds_inference

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
if (prediction.path / "scoring" / "window_gene" / "genescores.pkl").exists():
    genescores = pd.read_pickle(
        prediction.path / "scoring" / "window_gene" / "genescores.pkl"
    )
    design = pd.read_pickle(prediction.path / "scoring" / "window_gene" / "design.pkl")
else:
    scores = {}
    for gene in tqdm.tqdm(promoters.index):
        scores_folder = prediction.path / "scoring" / "window_gene" / gene

        if scores_folder.exists():
            try:
                scoring = chd.scoring.prediction.Scoring.load(scores_folder)
            except FileNotFoundError:
                continue
            scores[gene] = scoring.genescores
    genescores = xr.concat(
        [scores.mean("model") for scores in scores.values()],
        dim=pd.Index(scores.keys(), name="gene"),
    ).to_dataframe()
    design = scoring.design

    genescores.to_pickle(prediction.path / "scoring" / "window_gene" / "genescores.pkl")
    design.to_pickle(prediction.path / "scoring" / "window_gene" / "design.pkl")

# %%
# fix retained being nan because of no fragments present
genescores.loc[pd.isnull(genescores["retained"]), "retained"] = 1.0
genescores.loc[np.isinf(genescores["retained"]), "retained"] = 1.0

# %%
# calcualte overall score
scores = genescores.groupby(["phase", "window"]).mean()

scores.loc["validation"][["retained"]]

# %%
scores_dir = prediction.path / "scoring" / "windows"

# %%
scores_dir_overall = prediction.path / "scoring" / "overall"

scores_overall = pd.read_pickle(scores_dir_overall / "scores.pkl")
genescores_overall = pd.read_pickle(scores_dir_overall / "genescores.pkl")
genescores_overall["label"] = transcriptome.symbol(
    genescores_overall.index.get_level_values("gene")
).values

# %%
deltacor_window_cutoff = -0.001

# scores["deltacor"] = scores["cor"] - scores_overall["cor"]
# scores["deltamse"] = -(scores["mse"] - scores_overall["mse"])

# genescores["deltacor"] = genescores["cor"] - genescores_overall["cor"]
# genescores["deltamse"] = -(genescores["mse"] - genescores_overall["mse"])
genescores["deltacor_mask"] = genescores["deltacor"] < deltacor_window_cutoff

# %%
metric = "deltacor"
# metric = "deltamse"

overallmetric = "cor_diff"
# overallmetric = "mse_diff"

# %%
genecor = genescores[metric].unstack()
generetained = genescores["retained"].unstack()
geneeffect = genescores["effect"].unstack()


# %%
def join_overall(df, columns=(overallmetric, "label")):
    todrop = [col for col in columns if col in df.columns]
    return (
        df.drop(columns=todrop)
        .join(genescores_overall.loc["validation", columns])
        .sort_values(overallmetric, ascending=False)
    )


# %%
phases = chd.plotting.phases

# %% [markdown]
# ### Global view

# %%
scores.loc["test"]["retained"].plot(color=phases.loc["validation", "color"])
scores.loc["train"]["retained"].plot(color=phases.loc["train", "color"])

# %%
mse_windows = scores["cor"].unstack().T
effect_windows = scores["effect"].unstack().T

# %%
fig, ax_mse = plt.subplots(figsize=(3, 2))
ax_mse2 = ax_mse.twinx()

patch_train = ax_mse2.plot(
    mse_windows.index,
    effect_windows["train"],
    color=phases.loc["train", "color"],
    label="train",
)

patch_validation = ax_mse.plot(
    mse_windows.index,
    effect_windows["validation"],
    color=phases.loc["validation", "color"],
    label="validation",
    zorder=10,
)

ax_mse2.set_ylabel(
    "effect train", rotation=0, ha="left", color=phases.loc["train", "color"]
)
ax_mse.set_ylabel(
    "effect validation", rotation=0, ha="right", color=phases.loc["validation", "color"]
)
ax_mse.axvline(0, color="#33333366", lw=1)

ax_mse.set_zorder(ax_mse2.get_zorder() + 1)
ax_mse.set_frame_on(False)

ax_mse.set_xlabel("window (mid)")

# plt.legend([patch_train[0], patch_validation[0]], ['train', 'validation'])

# %%
import sklearn.linear_model

lm = sklearn.linear_model.LinearRegression()
lm.fit(scores.loc["validation"][["retained"]], 1 - scores.loc["validation"][metric])
mse_residual = (
    1
    - scores.loc["validation"][metric]
    - lm.predict(scores.loc["validation"][["retained"]])
) / scores.loc["validation"][metric].std()

# %%
fig, ax = plt.subplots(figsize=(3, 2))
patch_train = ax.plot(
    mse_residual.index,
    mse_residual,
    color=phases.loc["validation", "color"],
    label="validation",
)

ax.set_xlabel("window (mid)")
ax.set_ylabel("Relative\nimportance", ha="right", rotation=0, va="center")

# %%
window_oi = (mse_residual.index > -400) & (mse_residual.index < 2000)
fig, ax = plt.subplots(figsize=(3, 2))
patch_train = ax.plot(
    mse_residual.index[window_oi],
    mse_residual[window_oi],
    color=phases.loc["validation", "color"],
    label="validation",
)

ax.set_xlabel("window (mid)")
ax.set_ylabel("Relative\nimportance", ha="right", rotation=0, va="center")

# %%
import sklearn.linear_model

lm = sklearn.linear_model.LinearRegression()
lm.fit(scores.loc["validation"][["retained"]], 1 - scores.loc["validation"]["effect"])
mse_residual = (
    1
    - scores.loc["validation"]["effect"]
    - lm.predict(scores.loc["validation"][["retained"]])
) / scores.loc["validation"]["effect"].std()

# %%
fig, ax = plt.subplots(figsize=(3, 2))
patch_train = ax.plot(
    mse_residual.index,
    mse_residual,
    color=phases.loc["validation", "color"],
    label="validation",
)

ax.set_xlabel("window (mid)")
ax.set_ylabel("Relative\neffect", ha="right", rotation=0, va="center")

# %%
window_oi = (mse_residual.index > -400) & (mse_residual.index < 2000)
fig, ax = plt.subplots(figsize=(3, 2))
patch_train = ax.plot(
    mse_residual.index[window_oi],
    mse_residual[window_oi],
    color=phases.loc["validation", "color"],
    label="validation",
)

ax.set_xlabel("window (mid)")
ax.set_ylabel("Relative\neffect", ha="right", rotation=0, va="center")


# %%
def zscore(x, dim=0):
    return (x - x.values.mean(dim, keepdims=True)) / x.values.std(dim, keepdims=True)


# %%
fig, ax = plt.subplots()
ax.set_ylabel(f"Relative {metric}", rotation=0, ha="right", va="center")
ax.plot(
    scores.loc["validation"].index,
    zscore(scores.loc["validation"][metric])
    - zscore(1 - scores.loc["validation"]["retained"]),
    color=phases.loc["validation", "color"],
)
ax.plot(
    scores.loc["train"].index,
    zscore(scores.loc["train"][metric]) - zscore(1 - scores.loc["train"]["retained"]),
    color=phases.loc["train", "color"],
)

# %% [markdown]
# ### Gene-specific view

# %%
special_genes = pd.DataFrame(index=transcriptome.var.index)

# %%
genescores["label"] = pd.Categorical(
    transcriptome.symbol(genescores.index.get_level_values("gene")).values
)

# %%
fig, ax = plt.subplots()
ax.hist(
    genecor.loc["validation"].idxmax(1),
    bins=genecor.columns,
    histtype="stepfilled",
    alpha=0.8,
    color=phases.loc["validation", "color"],
)
fig.suptitle("Most important window across genes")
None

# %%
genecor_notss = genecor.loc[:, (genecor.columns < -2000) | (genecor.columns > 2000)]

# %%
fig, ax = plt.subplots()
ax.hist(
    genecor_notss.loc["validation"].idxmax(1),
    bins=genecor.columns,
    histtype="stepfilled",
    alpha=0.8,
    color=phases.loc["validation", "color"],
)
fig.suptitle("Most important window across genes outside of TSS")

# %%
genecor_norm = (genecor - genecor.values.min(1, keepdims=True)) / (
    genecor.values.max(1, keepdims=True) - genecor.values.min(1, keepdims=True)
)

# %%
# sns.heatmap(genecor_norm.loc["validation"].loc[genescores_overall.loc["validation"].sort_values("cor", ascending = False).index])

# %% [markdown]
# #### Plot a single gene

# %%
# if you want genes with a high "negative effect" somewhere
geneeffect.loc["validation"].max(1).sort_values(ascending=False).head(8)

# if you want genes with the highest mse diff
# genescores_overall.loc["validation"].sort_values(overallmetric, ascending = False).head(8).style.bar(subset = overallmetric)

# %%
# gene_id = transcriptome.gene_id("LYN")
# gene_id = transcriptome.gene_id("GLUL")
# gene_id = transcriptome.gene_id("PTPRC")
# gene_id = transcriptome.gene_id("IL1B")
# gene_id = transcriptome.gene_id("NFKBIA")
# gene_id = transcriptome.gene_id("LAMC1")
# gene_id = transcriptome.gene_id("WARS")
# gene_id = transcriptome.gene_id("IL1B")
# gene_id = transcriptome.gene_id("GPCPD1")
gene_id = transcriptome.gene_id("LTB")
gene_id = transcriptome.gene_id("NKG7")
gene_id = transcriptome.gene_id("FOXO3")
gene_id = transcriptome.gene_id("FOSB")
gene_id = transcriptome.gene_id("BCL2")
# gene_id = transcriptome.gene_id("CTLA4")
# gene_id = "ENSG00000182901"

# gene_id = transcriptome.gene_id("Lpcat2")
# gene_id = transcriptome.gene_id("Egr1")
# gene_id = transcriptome.gene_id("Neurod2")

# %% [markdown]
# Extract promoter info of gene

# %%
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# %%
promoter = promoters.loc[gene_id]


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


# %%
peaks = []

import pybedtools

promoter_bed = pybedtools.BedTool.from_dataframe(
    pd.DataFrame(promoter).T[["chr", "start", "end"]]
)

peak_methods = []

peaks_bed = pybedtools.BedTool(folder_data_preproc / "atac_peaks.bed")
peaks_cellranger = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_cellranger["method"] = "cellranger"
peaks_cellranger = center_peaks(peaks_cellranger, promoter)
peaks.append(peaks_cellranger)
peak_methods.append({"method": "cellranger"})

peaks_bed = pybedtools.BedTool(
    chd.get_output() / "peaks" / dataset_name / "macs2_leiden_0.1_merged" / "peaks.bed"
)
peaks_macs2 = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_macs2["method"] = "macs2"
peaks_macs2 = center_peaks(peaks_macs2, promoter)
peaks.append(peaks_macs2)
peak_methods.append({"method": "macs2"})

# peaks_bed = pybedtools.BedTool(chd.get_output() / "peaks" / dataset_name / "genrich" / "peaks.bed")
# peaks_genrich = promoter_bed.intersect(peaks_bed).to_dataframe()
# peaks_genrich["method"] = "genrich"
# peaks_genrich = center_peaks(peaks_genrich, promoter)
# peaks.append(peaks_genrich)
# peak_methods.append({"method":"genrich"})

peaks = pd.concat(peaks)

peak_methods = pd.DataFrame(peak_methods).set_index("method")
peak_methods["ix"] = np.arange(peak_methods.shape[0])

# %% [markdown]
# Extract bigwig info of gene

# %%
import pyBigWig

bw = pyBigWig.open(str(folder_data_preproc / "atac_cut_sites.bigwig"))

# %%
import chromatinhd.grid

# %%
# fig, (ax_mse, ax_effect, ax_perc, ax_peak, ax_bw) = plt.subplots(5, 1, height_ratios = [1, 0.5, 0.5, 0.2, 0.2], sharex=True)

fig = chd.grid.Figure(chromatinhd.grid.Wrap(ncol=1, padding_height=0.001))

w = 5

ax_title_ = fig.main.add(
    chd.grid.Title(transcriptome.symbol(gene_id) + " promoter", dim=(w, 0.8))
)

limits = (design.index[0], design.index[-1])
# limits = (-1000, 1000)

# correlation
ax_cor_ = fig.main.add(chd.grid.Ax(dim=(5, 1)))
ax_cor = ax_cor_.ax
ax_cor2 = ax_cor.twinx()

# cor annot
ax_cor2.set_ylabel(
    "$\\Delta$ cor\ntrain",
    rotation=0,
    ha="left",
    va="center",
    color=phases.loc["train", "color"],
)
ax_cor.set_ylabel(
    "$\\Delta$ cor\nvalidation",
    rotation=0,
    ha="right",
    va="center",
    color=phases.loc["validation", "color"],
)
ax_cor.set_xlim(*limits)
ax_cor.invert_yaxis()
ax_cor2.invert_yaxis()
ax_cor.xaxis.tick_top()
ax_cor.axhline(dashes=(2, 2), color="grey")

plotdata = genescores.loc["train"].loc[gene_id][metric]
patch_train = ax_cor2.plot(
    plotdata.index,
    plotdata,
    color=phases.loc["train", "color"],
    label="train",
    alpha=0.3,
)
plotdata = genescores.loc["validation"].loc[gene_id][metric]
patch_validation = ax_cor.plot(
    plotdata.index, plotdata, color=phases.loc["validation", "color"], label="train"
)

# effect
ax_effect_ = fig.main.add(chd.grid.Ax(dim=(5, 1)))
ax_effect = ax_effect_.ax
plotdata = geneeffect.loc["train"].loc[gene_id]
patch_train = ax_effect.plot(
    plotdata.index, plotdata, color=phases.loc["train", "color"], label="train"
)
plotdata = geneeffect.loc["validation"].loc[gene_id]
patch_validation = ax_effect.plot(
    plotdata.index, plotdata, color=phases.loc["validation", "color"], label="train"
)

ax_effect.axhline(0, color="#333333", lw=0.5, zorder=0)
ax_effect.set_ylim(ax_effect.get_ylim()[::-1])
ax_effect.set_ylabel("Effect", rotation=0, ha="right", va="center")
ax_effect.set_xlim(*limits)

# fragments
ax_fragments_ = fig.main.add(chd.grid.Ax(dim=(5, 1)))
ax_fragments = ax_fragments_.ax
# plotdata = generetained.loc["train"].loc[gene_id]
# ax_perc.plot(plotdata.index, plotdata, color = phases.loc["train", "color"], label = "train", alpha = 0.3)
plotdata = 1 - generetained.loc["validation"].loc[gene_id]
ax_fragments.plot(
    plotdata.index, plotdata, color=phases.loc["validation", "color"], label="train"
)

ax_fragments.set_ylabel("% fragments", rotation=0, ha="right", va="center")
ax_fragments.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax_fragments.set_xlim(*limits)

# peaks
ax_peak_ = fig.main.add(chd.grid.Ax(dim=(5, 0.15 * peak_methods.shape[0])))
ax_peak = ax_peak_.ax
for _, peak in peaks.iterrows():
    y = peak_methods.loc[peak["method"], "ix"]
    rect = mpl.patches.Rectangle(
        (peak["start"], y), peak["end"] - peak["start"], 1, fc="#333"
    )
    ax_peak.add_patch(rect)
ax_peak.set_ylim(0, peak_methods["ix"].max() + 1)
ax_peak.set_yticks(peak_methods["ix"] + 0.5)
ax_peak.set_yticklabels(peak_methods.index)
ax_peak.set_ylabel("Peaks", rotation=0, ha="right", va="center")
ax_peak.set_xlim(ax_fragments.get_xlim())
ax_peak.set_xlim(*limits)

# bw
ax_bw_ = fig.main.add(chd.grid.Ax(dim=(5, 0.5)))
ax_bw = ax_bw_.ax
ax_bw.plot(
    np.arange(promoter["start"] - promoter["tss"], promoter["end"] - promoter["tss"])
    * promoter["strand"],
    bw.values(promoter["chr"], promoter["start"], promoter["end"]),
    color="#333",
)
ax_bw.set_ylabel("Smoothed\nfragments", rotation=0, ha="right", va="center")
ax_bw.set_ylim(0)
ax_bw.set_xlim(*limits)

# legend
# ax_cor.legend([patch_train[0], patch_validation[0]], ['train', 'validation'])
# ax_cor.set_xlim(*ax_cor.get_xlim()[::-1])
# fig.suptitle(transcriptome.symbol(gene_id) + " promoter")

for ax_ in fig.main.elements[1:]:
    pass
    # ax_.ax.axvline(0, color = "#33333366", lw = 0.5)
    # ax_.ax.axvline(6425, color = "#33333366", lw = 0.5)

fig.plot()
fig.savefig("window.png", bbox_inches="tight", pad_inches=0, dpi=100)

# %%
sc.pl.umap(transcriptome.adata, color=gene_id)

# %%
# if you want to explore this window in igv
import IPython.display

IPython.display.HTML(
    "<textarea>" + chd.utils.name_window(promoters.loc[gene_id]) + "</textarea>"
)

# %% [markdown]
# ### Does removing a window improve test performances?

# %%
sns.ecdfplot(genescores[metric].loc["test"].groupby("gene").mean().sort_values())
sns.ecdfplot(genescores[metric].loc["validation"].groupby("gene").mean().sort_values())
plt.axvline(0, color="#333333", lw=0.5)

# %% [markdown]
# ### Enhancers and repressors

# %% [markdown]
# - "up" = gene expression goes up if we remove a window, i.e. if there are fragments in this window the gene expression goes down
# - "down" = gene expression goes down if we remove a window, i.e. if there are fragments in this window the gene expression goes up

# %%
promoter_updown = pd.DataFrame(
    {
        "best_enhancer": -geneeffect.loc["test"].min(1),
        "best_repressor": +geneeffect.loc["test"].max(1),
    }
)
sns.scatterplot(x=promoter_updown["best_enhancer"], y=promoter_updown["best_repressor"])

# %%
genereleffect = geneeffect.copy()
# genereleffect[~gene_mask] = 0.
genereleffect = genereleffect / genereleffect.values.std(1, keepdims=True)

# %%
promoter_updown = pd.DataFrame(
    {
        "best_enhancer": -genereleffect.loc["validation"].min(1),
        "best_repressor": +genereleffect.loc["validation"].max(1),
    }
)
promoter_updown["label"] = transcriptome.symbol(promoter_updown.index)

cutoff_enhancer = 1  # np.quantile(promoter_updown["best_enhancer"], 0.9)
cutoff_repressor = 1  # np.quantile(promoter_updown["best_repressor"], 0.1)

promoter_updown["enhancer"] = promoter_updown["best_enhancer"] > cutoff_enhancer
promoter_updown["repressor"] = promoter_updown["best_repressor"] > cutoff_repressor
promoter_updown["type"] = (
    pd.Series(["nothing", "repressor"])[
        promoter_updown["repressor"].values.astype(int)
    ].reset_index(drop=True)
    + "_"
    + pd.Series(["nothing", "enhancer"])[
        promoter_updown["enhancer"].values.astype(int)
    ].reset_index(drop=True)
).values

promoter_updown["cor_diff"] = genescores_overall.loc["validation"]["cor_diff"]
genes_oi = genescores_overall.loc["validation"]["cor_diff"] > 0.01
print(genes_oi.sum())

fig, ax = plt.subplots()
type_info = pd.DataFrame(
    [
        ["repressor_nothing", "green"],
        ["nothing_enhancer", "blue"],
        ["repressor_enhancer", "red"],
        ["nothing_nothing", "grey"],
    ]
)

sns.scatterplot(
    x=promoter_updown.loc[genes_oi]["best_enhancer"],
    y=promoter_updown.loc[genes_oi]["best_repressor"],
    hue=promoter_updown["type"],
)

# %%
promoter_updown.groupby("type").size() / promoter_updown.shape[0]

# %%
# if you're interested in the genes with the strongest increase effect
promoter_updown.loc[genes_oi].query("repressor").sort_values(
    "cor_diff", ascending=False
).head(10)

# %%
special_genes["opening_decreases_expression"] = promoter_updown["repressor"]

# %% [markdown]
# ### # fragments => predictive ability?

# %%
sns.scatterplot(x=genescores[metric], y=genescores["retained"], s=1)

# %%
retained_cutoff = 0.99

# %%
# gene_mse_correlations = genescores.groupby(["phase", "gene"]).apply(lambda x: x["retained"].corr(x[metric]))
gene_mse_correlations = (
    genescores.query("retained < @retained_cutoff")
    .groupby(["phase", "gene"])
    .apply(lambda x: x["retained"].corr(x[metric]))
)
gene_mse_correlations = gene_mse_correlations.to_frame("retained_cor")
gene_mse_correlations = gene_mse_correlations[
    ~pd.isnull(gene_mse_correlations["retained_cor"])
]

# %%
deltacor_cutoff = 0.05
genes_oi = genescores_overall.loc["validation"].index[
    genescores_overall.loc["validation"][overallmetric] > deltacor_cutoff
]
print(f"{len(genes_oi)=}")

# %%
deltacor_cutoff = 0.1
genes_oi2 = genescores_overall.loc["validation"].index[
    genescores_overall.loc["validation"][overallmetric] > deltacor_cutoff
]
print(f"{len(genes_oi2)=}")

# %%
fig, ax = plt.subplots(figsize=(2, 2))

color = "black"
label = f"All genes"
sns.ecdfplot(
    gene_mse_correlations.loc["validation"].loc[:, "retained_cor"],
    ax=ax,
    label="all",
    color=color,
)
q = 0.4
median = np.quantile(gene_mse_correlations.loc["validation"].loc[:, "retained_cor"], q)
ax.annotate(
    label,
    (median, q),
    xytext=(10, -25),
    ha="center",
    va="center",
    arrowprops=dict(arrowstyle="-", color=color),
    textcoords="offset points",
    fontsize=8,
    bbox=dict(boxstyle="round", fc="w", ec=color),
)

color = "orange"
# label = f"$\\Delta$ cor > {deltacor_cutoff}\n({len(genes_oi)})"
label = (
    "$cor_{ChromatinHD} -$\n$cor_{Baseline}$ $\geq$ 0.1"
    + f"\n ({len(genes_oi) } genes)"
)
sns.ecdfplot(
    gene_mse_correlations.loc["validation"].loc[genes_oi, "retained_cor"],
    ax=ax,
    label=label,
    color=color,
)
q = 0.8
median = np.quantile(
    gene_mse_correlations.loc["validation"].loc[genes_oi, "retained_cor"], q
)
ax.annotate(
    label,
    (median, q),
    xytext=(-15, 15),
    ha="right",
    va="top",
    arrowprops=dict(arrowstyle="-", color=color),
    textcoords="offset points",
    fontsize=8,
    bbox=dict(boxstyle="round", fc="w", ec=color),
)
ax.set_xlabel(
    "Correlation between positional\n$\\Delta$ cor and # fragments",
    rotation=0,
    ha="center",
    va="top",
)
ax.set_xlim(-1, 1)
ax.axvline(0, dashes=(2, 2), color="#333333")
ax.axhline(0.5, dashes=(2, 2), color="#333333")
ax.set_ylabel("% genes")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

if dataset_name == "pbmc10k":
    manuscript.save_figure(fig, "4", f"correlation_nfragments_deltacor_{dataset_name}")

# %%
# if you're interested in genes with a higher correlation and a good prediction
gene_mse_correlations.loc["validation"].query("retained_cor > .8").pipe(
    join_overall
).head(10)

# if you're interested in genes with a low correlation and a good prediction
# gene_mse_correlations.loc["validation"].query("retained_cor < 0.5").pipe(join_overall).head(10)

# if you're interested in genes with a low (negative) correlation and a good prediction
gene_mse_correlations.loc["validation"].query("retained_cor < -0.").pipe(
    join_overall
).to_csv(scores_dir / "correlation_nfragments_deltacor_low.csv")
gene_mse_correlations.loc["validation"].query("retained_cor < -0.").pipe(
    join_overall
).head(10)

# %% [markdown]
# Interesting genes:
#  - *Egr1* in e18brain, probably gene body
#  - *SAT1* in both pbmc10k and lymphoma

# %% [markdown]
# ### Best predictive ability => very accessible?

# %%
special_genes["n_fragments_importance_not_correlated"] = (
    gene_mse_correlations.loc["validation"]["retained_cor"] < 0.5
)

# %%
plotdata = genescores[["retained", "deltacor"]].loc["validation"]

# %%
plotdata = pd.DataFrame(
    {
        "best_retained": genescores.groupby("gene").apply(
            lambda x: x["retained"][x["deltacor"].idxmax()]
        ),
        "max_retained": genescores.groupby("gene")["retained"].min(),
    }
)

# %%
(1 - plotdata["max_retained"].median()) / (1 - plotdata["best_retained"].median())

# %%
fig, ax = plt.subplots()
sns.ecdfplot(plotdata["best_retained"])
sns.ecdfplot(plotdata["max_retained"])
# ax.annotate(

#     plotdata["best_retained"].median(), 0.5)
# sns.ecdfplot(plotdata["max_retained"])

# %% [markdown]
# ### # fragments => effect?

# %%
sns.scatterplot(x=genescores["effect"], y=genescores["retained"], s=1)

# %% [markdown]
# We filter on $\Delta cor$ to only include windows that are predictive.

# %%
geneeffect_correlations = (
    genescores.query("deltacor_mask")
    .groupby(["phase", "gene"])
    .apply(lambda x: x["retained"].corr(x["effect"]))
)
geneeffect_correlations = geneeffect_correlations.to_frame("retained_cor")
geneeffect_correlations = geneeffect_correlations[
    ~pd.isnull(geneeffect_correlations["retained_cor"])
]

# %%
cor_diff_cutoff = 0.005

# %%
genes_oi = (
    genescores_overall.loc["validation"].query("cor_diff > @cor_diff_cutoff").index
)
genes_oi = genes_oi.intersection(
    geneeffect_correlations.loc["validation"].index.get_level_values("gene")
)
print(f"{len(genes_oi)=}")

# %%
cor_cutoff = 0.05

# %%
genes_oi2 = genescores_overall.loc["validation"].query("cor > @cor_cutoff").index
genes_oi2 = genes_oi2.intersection(
    geneeffect_correlations.loc["validation"].index.get_level_values("gene")
)
print(f"{len(genes_oi2)=}")

# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.ecdfplot(
    geneeffect_correlations.loc["validation"].loc[:, "retained_cor"], ax=ax, label="all"
)
sns.ecdfplot(
    geneeffect_correlations.loc["validation"].loc[genes_oi, "retained_cor"],
    ax=ax,
    label=f"Δcor > 0.01 ({len(genes_oi)})",
)
sns.ecdfplot(
    geneeffect_correlations.loc["validation"].loc[genes_oi2, "retained_cor"],
    ax=ax,
    label=f"cor > 0.1 ({len(genes_oi2)})",
)
ax.set_ylabel("ECDF", rotation=0, ha="right", va="center")
ax.set_xlabel(
    "Correlation between a window's effect and # fragments",
    rotation=0,
    ha="center",
    va="top",
)
ax.legend(title="genes")
ax.axvline(0, dashes=(2, 2), color="#333333")
# ax.hist(gene_mse_correlations.loc["validation"].loc[:, "retained_cor"], range = (-1, 1), density = True, histtype = "stepfilled")
# ax.hist(gene_mse_correlations.loc["validation"].loc[genes_oi, "retained_cor"], range = (-1, 1), density = True)

# %%
fig, ax = plt.subplots()
ax.hist(geneeffect_correlations.loc["validation"].loc[:, "retained_cor"], range=(-1, 1))
ax.hist(
    geneeffect_correlations.loc["validation"].loc[genes_oi, "retained_cor"],
    range=(-1, 1),
)
ax.hist(
    geneeffect_correlations.loc["validation"].loc[genes_oi2, "retained_cor"],
    range=(-1, 1),
)

# %%
# if you're interested in genes with a higher correlation and a good prediction
# geneeffect_correlations.loc["validation"].query("retained_cor > .8").join(genescores_overall[["cor_diff", "label"]]).sort_values("cor_diff", ascending = False)

# if you're interested in genes with a low correlation and a good prediction
geneeffect_correlations.loc["validation"].query("retained_cor < 0.2").join(
    genescores_overall[["cor_diff", "label"]]
).sort_values("cor_diff", ascending=False)

# if you're interested in genes with a low (negative) correlation and a good prediction
# geneeffect_correlations.loc["validation"].query("retained_cor < -0.2").sort_values("cor_diff", ascending = False).join(genescores_overall[["cor_diff", "label"]])

# %% [markdown]
# Interesting genes:
#  - **

# %%
special_genes["n_fragments_importance_not_correlated"] = (
    gene_mse_correlations.loc["validation"]["retained_cor"] < 0.5
)

# %% [markdown]
# ### Is the TSS positively, negatively or not associated with gene expression?

# %%
tss_window = design.index[np.argmin(np.abs(design.index - 0))]

# %%
genescores_tss = genescores.xs(key=design.loc[tss_window].name, level="window").copy()
# gene_scores_tss["label"] = transcriptome.symbol(gene_scores_tss.index.get_level_values("gene")).values

# %%
deltamse_cutoff = -0.01

# %%
sns.ecdfplot(
    genescores_tss.loc["validation"]
    .query(metric + " < @cutoff")
    .sort_values("effect")["effect"]
)

# %% [markdown]
# ## Comparing peaks and windows

# %% [markdown]
# ### Linking peaks to windows

# %% [markdown]
# Create a `peak_window_matches` dataframe that contains peak - window - gene in long format

# %%
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# %%
# peaks_name = "cellranger"
# peaks_name = "macs2_improved"
peaks_name = "macs2_leiden_0.1_merged"
# peaks_name = "rolling_500"

# %%
peaks_folder = folder_root / "peaks" / dataset_name / peaks_name
peaks = pd.read_table(
    peaks_folder / "peaks.bed", names=["chrom", "start", "end"], usecols=[0, 1, 2]
)

# %%
import pybedtools

promoters_bed = pybedtools.BedTool.from_dataframe(
    promoters.reset_index()[["chr", "start", "end", "gene"]]
)
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)

# %%
if peaks_name != "stack":
    intersect = promoters_bed.intersect(peaks_bed)
    intersect = intersect.to_dataframe()

    # peaks = intersect[["score", "strand", "thickStart", "name"]]
    peaks = intersect
peaks.columns = ["chrom", "start", "end", "gene"]
peaks = peaks.loc[peaks["start"] != -1]
peaks.index = pd.Index(
    peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str),
    name="peak",
)


# %%
def center_peaks(peaks, promoters):
    promoter = promoters.loc[peaks["gene"]]

    peaks2 = peaks.copy()

    peaks2["start"] = np.where(
        promoter["strand"].values == 1,
        (peaks["start"] - promoter["tss"].values) * promoter["strand"].values,
        (peaks["end"] - promoter["tss"].values) * promoter["strand"].values,
    )
    peaks2["end"] = np.where(
        promoter["strand"].values == 1,
        (peaks["end"] - promoter["tss"].values) * promoter["strand"].values,
        (peaks["start"] - promoter["tss"].values) * promoter["strand"].values,
    )
    return peaks2


# %%
localpeaks = center_peaks(peaks, promoters)

# %%
# match all localpeaks with the windows
matched_peaks, matched_windows = np.where(
    (
        (localpeaks["start"].values[:, None] < np.array(design)[:, 0][None, :])
        & (localpeaks["end"].values[:, None] > np.array(design)[:, 1][None, :])
    )
)

# %%
peak_window_matches = (
    pd.DataFrame(
        {
            "peak": localpeaks.index[matched_peaks],
            "window": design.index[matched_windows],
            "gene": localpeaks["gene"].iloc[matched_peaks],
        }
    )
    .set_index("peak")
    .reset_index()
)

# %% [markdown]
# ### Is the most predictive window inside a peak?

# %%
gene_best_windows = genescores.loc["test"].loc[
    genescores.loc["test"].groupby(["gene"])[metric].idxmin()
]
# genes_oi = (
#     genescores_overall.loc["test"]
#     .query(overallmetric + " > @overall_cutoff")
#     .index
# )
genes_oi = genescores_overall.loc["test"].index

# %%
gene_best_windows = gene_best_windows.join(
    peak_window_matches.set_index(["gene", "window"])
).reset_index(level="window")
gene_best_windows = gene_best_windows.groupby("gene").first()

# %%
gene_best_windows["matched"] = ~pd.isnull(gene_best_windows["peak"])

# %%
gene_best_windows["cor_overall"] = genescores_overall.loc["validation"]["cor"]

# %%
gene_best_windows = gene_best_windows.sort_values("cor_overall", ascending=False)
gene_best_windows["ix"] = np.arange(1, gene_best_windows.shape[0] + 1)
gene_best_windows["cum_matched"] = (
    np.cumsum(gene_best_windows["matched"]) / gene_best_windows["ix"]
)
gene_best_windows["perc"] = gene_best_windows["ix"] / gene_best_windows.shape[0]

# %% [markdown]
# Of the top 5% most predictive genes, how many are inside a peak?

# %%
top_cutoff = 1.0
perc_within_a_peak = gene_best_windows["cum_matched"].iloc[
    int(gene_best_windows.shape[0] * top_cutoff) - 1
]
print(perc_within_a_peak)
print(
    f"Perhaps the most predictive window in the promoter is not inside of a peak?\nIndeed, for {1-perc_within_a_peak:.2%} of the {top_cutoff:.0%} best predicted genes, the most predictive window does not lie within a peak."
)

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(gene_best_windows["perc"], 1 - gene_best_windows["cum_matched"])
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)
ax.plot([top_cutoff, top_cutoff], [0, 1 - perc_within_a_peak], color="red")
annot = f"{1-perc_within_a_peak:.2%}"
ax.annotate(
    annot,
    xy=(top_cutoff, 1 - perc_within_a_peak),
    xycoords=("data", "data"),
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    color="red",
    bbox=dict(fc="#FFFFFF88"),
)
# ax.annotate([top_cutoff, top_cutoff], [0, 1-perc_within_a_peak], dashes = (2, 2), color = "red")
ax.set_xlabel("Top genes (acording to cor)")
ax.set_title(
    "% genes where most predictive locus is not contained in a peak",
    rotation=0,
    loc="left",
)

# %%
scores_dir.mkdir(exist_ok=True, parents=True)
gene_best_windows.to_pickle(scores_dir / ("gene_best_windows_" + peaks_name + ".pkl"))

# %%
# if you're interested in genes where the best window is not inside a peak
gene_best_windows.query("~matched").pipe(join_overall).head(10)

# %%
special_genes["most_predictive_position_not_in_peak"] = ~gene_best_windows["matched"]

# %% [markdown]
# ### Are all predictive windows within a peak?

# %%
genescores_matched = (
    genescores.loc["test"]
    .join(peak_window_matches.set_index(["gene", "window"]))
    .groupby(["gene", "window"])
    .first()
    .reset_index(level="window")
)
genescores_matched["matched"] = ~pd.isnull(genescores_matched["peak"])
genescores_matched = genescores_matched.sort_values("deltacor", ascending=True)

# %%
genescores_matched["ix"] = np.arange(1, genescores_matched.shape[0] + 1)
genescores_matched["cum_matched"] = (
    np.cumsum(genescores_matched["matched"]) / genescores_matched["ix"]
)
genescores_matched["perc"] = genescores_matched["ix"] / genescores_matched.shape[0]

# %% [markdown]
# Of the top 5% most predictive sites, how many are inside a peak?

# %%
top_cutoff = 0.1
perc_within_a_peak = genescores_matched["cum_matched"].iloc[
    int(genescores_matched.shape[0] * top_cutoff)
]
print(perc_within_a_peak)
print(
    f"Perhaps there are many windows that are predictive, but are not contained in any peak?\nIndeed, {1-perc_within_a_peak:.2%} of the top {top_cutoff:.0%} predictive windows does not lie within a peak."
)

# %%
genescores_matched_oi = genescores_matched.iloc[
    : int(top_cutoff * genescores_matched.shape[0] * 2)
]

# %%
fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.plot(
    genescores_matched_oi["perc"], genescores_matched_oi["cum_matched"], color="#333"
)
annot = f"{perc_within_a_peak:.1%}"
ax.annotate(
    annot,
    xy=(top_cutoff, perc_within_a_peak),
    xycoords=("data", "data"),
    xytext=(0, -15),
    textcoords="offset points",
    va="top",
    ha="center",
    color="red",
    # bbox=dict(fc="#FFFFFF88"),
    # arrow with no pad
    arrowprops=dict(arrowstyle="->", color="red", shrinkA=0, shrinkB=0, lw=1),
)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)
ax.set_xlabel("Top windows\nupstream of TSS\n(ordered by $\Delta$ cor)")
ax.set_ylabel("% of windows in peak")
# ax.set_title(
#     "% of most predictive windows\nnot contained in a peak", rotation=0, loc="left"
# )

if dataset_name == "pbmc10k":
    manuscript.save_figure(fig, "4", f"predictive_windows_not_in_peak")

# %%
# if you're interested in the genes with the least matched regions
genescores_matched_summary = pd.DataFrame(
    {
        "perc_matched": genescores_matched_oi.groupby("gene")["matched"].mean(),
        "n_windows": genescores_matched_oi.groupby("gene").size(),
    }
)
genescores_matched_summary.pipe(join_overall).query("perc_matched < 0.2").query(
    "n_windows > 10"
).sort_values("cor_diff", ascending=False)

# %%
genescores_matched.to_pickle(scores_dir / ("genescores_matched_" + peaks_name + ".pkl"))

# %%
special_genes["predictive_positions_not_in_peak"] = (
    genescores_matched.iloc[: int(genescores_matched.shape[0] * top_cutoff)]
    .groupby("gene")["matched"]
    .all()
)

# %% [markdown]
# Of the top 5% most predictive sites upstream of tss, how many are inside a peak?

# %%
genescores_matched2 = genescores_matched.loc[genescores_matched["window"] < 0].copy()
genescores_matched2["ix"] = np.arange(1, genescores_matched2.shape[0] + 1)
genescores_matched2["cum_matched"] = (
    np.cumsum(genescores_matched2["matched"]) / genescores_matched2["ix"]
)
genescores_matched2["perc"] = genescores_matched2["ix"] / genescores_matched2.shape[0]

# %%
top_cutoff = 0.1
perc_within_a_peak = genescores_matched2["cum_matched"].iloc[
    int(genescores_matched2.shape[0] * top_cutoff)
]
print(perc_within_a_peak)
print(
    f"Perhaps there are many windows that are predictive, but are not contained in any peak?\nIndeed, {1-perc_within_a_peak:.2%} of the top {top_cutoff:.0%} predictive windows does not lie within a peak."
)

# %%
genescores_matched_oi = genescores_matched2.iloc[
    : int(top_cutoff * genescores_matched2.shape[0] * 2)
]

# %%
fig, ax = plt.subplots(figsize=(1.5, 1.5))
ax.plot(
    genescores_matched_oi["perc"], genescores_matched_oi["cum_matched"], color="#333"
)
annot = f"{perc_within_a_peak:.1%}"
ax.annotate(
    annot,
    xy=(top_cutoff, perc_within_a_peak),
    xycoords=("data", "data"),
    xytext=(0, -15),
    textcoords="offset points",
    va="top",
    ha="center",
    color="red",
    # bbox=dict(fc="#FFFFFF88"),
    # arrow with no pad
    arrowprops=dict(arrowstyle="->", color="red", shrinkA=0, shrinkB=0, lw=1),
)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1)
ax.set_xlabel("Top windows\n(ordered by $\Delta$ cor)")
ax.set_ylabel("% of windows in peak")
# ax.set_title(
#     "% of most predictive windows\nnot contained in a peak", rotation=0, loc="left"
# )

if dataset_name == "pbmc10k":
    manuscript.save_figure(fig, "4", f"predictive_windows_not_in_peak_upstream")

# %% [markdown]
# ### Are opposing effects put into the same peak?

# %%
gene_peak_scores = pd.DataFrame(
    {
        "effect_min": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["effect"]
        .min(),
        "effect_max": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["effect"]
        .max(),
        "deltacor_min": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["deltacor"]
        .min(),
        "deltacor_sum": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["deltacor"]
        .sum(),
        "window_mean": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["window"]
        .mean(),
        "cor_retained_deltacor": genescores_matched.query("matched")
        .groupby(["gene", "peak"])
        .apply(lambda x: np.corrcoef(x["retained"], x["deltacor"])[0, 1]),
    }
)

gene_peak_scores["label"] = transcriptome.symbol(
    gene_peak_scores.index.get_level_values("gene")
).values

# %%
gene_peak_scores["effect_highest"] = np.maximum(
    np.abs(gene_peak_scores["effect_min"]), np.abs(gene_peak_scores["effect_max"])
)
gene_peak_scores["effect_highest_cutoff"] = (
    gene_peak_scores["effect_highest"] / 8
)  # we put the cutoff at 1/8 of the highest effect

# %%
gene_peak_scores["up"] = (
    gene_peak_scores["effect_max"] > gene_peak_scores["effect_highest_cutoff"]
)
gene_peak_scores["down"] = (
    gene_peak_scores["effect_min"] < -gene_peak_scores["effect_highest_cutoff"]
)
gene_peak_scores["updown"] = gene_peak_scores["up"] & gene_peak_scores["down"]

# %%
gene_peak_scores = gene_peak_scores.sort_values("deltacor_min", ascending=True)

# %%
gene_peak_scores["ix"] = np.arange(1, gene_peak_scores.shape[0] + 1)
gene_peak_scores["cum_updown"] = (
    np.cumsum(gene_peak_scores["updown"]) / gene_peak_scores["ix"]
)
gene_peak_scores["perc"] = gene_peak_scores["ix"] / gene_peak_scores.shape[0]

# %%
gene_peak_scores["cum_updown"].plot()

# %% [markdown]
# Of the top X% most predictive peaks, how many have a single effect?

# %%
top_cutoff = 0.2
perc_updown = gene_peak_scores["cum_updown"].iloc[
    int(gene_peak_scores.shape[0] * top_cutoff)
]
print(perc_updown)
print(
    f"Perhaps within a peak there may be both windows that are positively and negatively correlated with gene expression?\nIndeed, {perc_updown:.2%} of the top {top_cutoff:.0%} predictive peaks contains both positive and negative effects."
)

# %% [markdown]
# Of the top X% most predictive peaks, are retained and $\Delta$ cor correlated?

# %%
gene_peak_scores["ix"] = np.arange(1, gene_peak_scores.shape[0] + 1)
gene_peak_scores["cum_cor_retained_deltacor"] = (
    np.cumsum(gene_peak_scores["cor_retained_deltacor"]) / gene_peak_scores["ix"]
)

# %%
gene_peak_scores["cum_cor_retained_deltacor"].plot()
# %%
top_cutoff = 0.1
perc_updown = gene_peak_scores["cor_retained_deltacor"].iloc[
    int(gene_peak_scores.shape[0] * top_cutoff)
]
print(perc_updown)
print(
    f"Perhaps within a peak there may be both windows that are positively and negatively correlated with gene expression?\nIndeed, {perc_updown:.2%} of the top {top_cutoff:.0%} predictive peaks contains both positive and negative effects."
)

# %%
sns.ecdfplot(
    gene_peak_scores["cor_retained_deltacor"].iloc[
        : int(gene_peak_scores.shape[0] * 0.05)
    ]
)
sns.ecdfplot(
    gene_peak_scores["cor_retained_deltacor"].iloc[
        : int(gene_peak_scores.shape[0] * top_cutoff)
    ]
)

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(gene_peak_scores["perc"], gene_peak_scores["cum_updown"])

ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
annot = f"{perc_updown:.2%}"
ax.annotate(
    annot,
    xy=(top_cutoff, perc_updown),
    xycoords=("data", "data"),
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    color="red",
    bbox=dict(fc="#FFFFFF88"),
)
ax.set_xlabel("Top peaks (acording to cor)")
ax.set_title("% peaks with both positive and negative effects", rotation=0, loc="left")

# %%
special_genes["up_and_down_in_peak"] = gene_peak_scores.groupby("gene")["updown"].any()

# %%
# if you're interested in the most predictive peaks with both up and down effects
gene_peak_scores.query("updown")[
    ["window_mean", "deltacor_sum", "effect_max", "effect_min"]
].pipe(join_overall).sort_values("deltacor_sum", ascending=True).head(15)

# %% [markdown]
# ### How much information do the non-peak regions contain?

# %%
genescores_matched = (
    genescores.loc["validation"]
    .join(peak_window_matches.set_index(["gene", "window"]))
    .groupby(["gene", "window"])
    .first()
    .reset_index(level="window")
)
genescores_matched["matched"] = ~pd.isnull(genescores_matched["peak"])
genescores_matched = genescores_matched.sort_values("deltacor", ascending=True)

# %%
matched_scores = (
    genescores_matched.groupby(["gene", "matched"])["deltacor"].sum().unstack()
)
matched_scores
print(
    f"Perhaps there is information outside of peaks?\nIndeed, {matched_scores.mean(0)[False] / matched_scores.mean(0).sum():.2%} of the cor is gained outside of peaks."
)

# %%
plotdata = genescores_matched.groupby(["gene", "matched"]).sum().reset_index()
sns.boxplot(x="matched", y="deltacor", data=plotdata)

# %%
genescores_matched_loci = (
    genescores_matched.groupby(["gene", "matched"])["deltacor"].sum().unstack()
)
genescores_matched_loci = genescores_matched_loci
genescores_matched_loci[genescores_matched_loci > 0] = 0
# genescores_matched_loci = genescores_matched_loci / genescores_matched_loci.values.sum(1, keepdims = True)

# %%
# gene_order = genescores_overall.loc["validation"]["cor_diff"].sort_values(ascending = False).index[:100]
gene_order = genescores_matched_loci.sum(1).sort_values(ascending=True).index[:5000]

# %%
fig = chd.grid.Figure(chd.grid.Grid())

panel, ax = fig.main.add_right(chd.grid.Panel((5, 2)))
inside_peaks = (
    genescores_matched_loci.iloc[:, 1].sum() / genescores_matched_loci.sum().sum()
)
outside_peaks = (
    genescores_matched_loci.iloc[:, 0].sum() / genescores_matched_loci.sum().sum()
)
ax.bar(
    np.arange(len(gene_order)),
    -genescores_matched_loci.loc[gene_order, True],
    width=1,
    lw=0,
    label="In peaks ({}%)".format(int(inside_peaks * 100)),
    color="#0074D9",
)
ax.bar(
    np.arange(len(gene_order)),
    -genescores_matched_loci.loc[gene_order, False],
    bottom=-genescores_matched_loci.loc[gene_order, True],
    width=1,
    lw=0,
    label="Outside peaks ({}%)".format(int(outside_peaks * 100)),
    color="#FF851B",
)
ax.set_xlim(0, len(gene_order) + 1)
ax.set_xlabel("Genes (sorted by $\\Delta$ cor)")
ax.set_ylabel("$\\Delta$ cor")
sns.despine()
ax.legend(loc="upper left", ncol=2, frameon=False)

fig.plot()

# %%
manuscript.save_figure(fig, "4", "information_beyond_peaks")


# %%
special_genes["information_beyond_peaks"] = (
    matched_scores[False] / (matched_scores[True] + matched_scores[False])
) > 0.2

# %% [markdown]
# ### Is the most informative locus in a peak also its summit?

# %%
genescores_matched = (
    genescores.loc["validation"]
    .join(peak_window_matches.set_index(["gene", "window"]))
    .groupby(["gene", "window"])
    .first()
    .reset_index(level="window")
)
genescores_matched["matched"] = ~pd.isnull(genescores_matched["peak"])


# %%
def match_deltacor_retained(df, deltacor_quantile=1.0, retained_quantile=1.0):
    return (
        (df["retained"] <= df["retained"].quantile(1 - retained_quantile))
        & (df["deltacor"] <= df["deltacor"].quantile(1 - deltacor_quantile))
    ).any()


# %%
peak_max_matches = (
    genescores_matched.query("matched")
    .groupby(["gene", "peak"])
    .apply(match_deltacor_retained, deltacor_quantile=1.0, retained_quantile=1.0)
)
# peak_max_matches = genescores_matched.query("matched").groupby(["gene", "peak"]).apply(match_deltacor_retained, deltacor_quantile = 0.8, retained_quantile = 0.9)

# %%
peak_max_scores = pd.DataFrame(
    {
        "match": peak_max_matches,
        "deltacor_sum": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["deltacor"]
        .sum(),
        "window_mean": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["window"]
        .mean(),
    }
)
peak_max_scores = peak_max_scores.sort_values("deltacor_sum", ascending=True)

# %%
peak_max_scores["ix"] = np.arange(1, peak_max_scores.shape[0] + 1)
peak_max_scores["cum_nonmatched"] = (
    np.cumsum(~peak_max_scores["match"]) / peak_max_scores["ix"]
)
peak_max_scores["perc"] = peak_max_scores["ix"] / peak_max_scores.shape[0]

# %% [markdown]
# Of the top 20% most predictive peaks, how many have a match between # of fragments and most predictive window

# %%
top_cutoff = 0.2
perc_notmatched = peak_max_scores["cum_nonmatched"].iloc[
    int(peak_max_scores.shape[0] * top_cutoff)
]
print(perc_notmatched)
print(
    f"Perhaps within a peak the peak maximum is not really the most predictive window?\nIndeed, {perc_notmatched:.2%} of the top {top_cutoff:.0%} predictive peaks does not have a match between the top predictive locus and the max of the peak."
)

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(peak_max_scores["perc"], peak_max_scores["cum_nonmatched"])

ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
annot = f"{perc_notmatched:.2%}"
ax.annotate(
    annot,
    xy=(top_cutoff, perc_notmatched),
    xycoords=("data", "data"),
    xytext=(0, 10),
    textcoords="offset points",
    va="bottom",
    ha="center",
    color="red",
    bbox=dict(fc="#FFFFFF88"),
    arrowprops={"arrowstyle": "-", "ec": "red"},
)
ax.set_xlabel("Top genes (acording to cor)")
ax.set_title(
    "% peaks where summit does not match top $\Delta$ cor", rotation=0, loc="left"
)

# %%
# if you're interested in genes where one peak's maximum does not match with the most predictive window
peak_max_scores.loc[~peak_max_scores["match"]].pipe(join_overall).sort_values(
    "deltacor_sum"
).head(10)

# %% [markdown]
# ### What is the distance between the peak maximum and the most predictive window within a peak?

# %%
peak_max_scores["distance"] = (
    genescores_matched.reset_index()
    .set_index("window")
    .groupby(["gene", "peak"])["deltacor"]
    .idxmin()
    - genescores_matched.reset_index()
    .set_index("window")
    .groupby(["gene", "peak"])["retained"]
    .idxmin()
)

# %%
fig, ax = plt.subplots()
ax.hist(
    peak_max_scores.query("perc < @top_cutoff")["distance"], range=(-500, 500), bins=11
)

# %%
# if you're interested in genes/peaks where there is a high distance between peak max and mse min
print((np.abs(peak_max_scores["distance"]) >= 500).mean())
peak_max_scores.query("abs(distance) > 500").pipe(join_overall).head(10)

# %%
top_cutoff = 0.1

# %%
fig, ax = plt.subplots(figsize=(2, 2))
sns.ecdfplot(
    np.abs(peak_max_scores.query("perc < @top_cutoff")["distance"]), color="black"
)
ax.set_ylabel("% peaks")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_xlabel("Distance between summit\nand most predictive position")
ax.set_xlim(-10, 1000)

if peaks_name == "macs2_leiden_0.1_merged":
    manuscript.save_figure(fig, "4", "peak_summit_distance")

# %% [markdown]
# ### Is the number of fragments within a peak correlated with importance?

# %%
######## COR!!!

peak_max_scores = pd.DataFrame(
    {
        "retained_cor": genescores_matched.query("matched")
        .groupby(["gene", "peak"])
        .apply(lambda x: x["retained"].corr(x[metric])),
        "window_mean": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["window"]
        .mean(),
        "deltacor_sum": genescores_matched.query("matched")
        .groupby(["gene", "peak"])["deltacor"]
        .sum(),
    }
)
peak_max_scores = peak_max_scores.sort_values("deltacor_sum", ascending=True)

# %%
peak_max_scores["ix"] = np.arange(peak_max_scores.shape[0]) + 1
peak_max_scores["perc"] = peak_max_scores["ix"] / peak_max_scores.shape[0]
peak_max_scores["cum_retained_cor"] = (
    np.cumsum(peak_max_scores["retained_cor"]) / peak_max_scores["ix"]
)

# %%
top_cutoff = 0.2
retained_cor = peak_max_scores["cum_retained_cor"].iloc[
    int(peak_max_scores.shape[0] * top_cutoff)
]
print(retained_cor)
print(
    f"Perhaps within a peak there is not really a correlation between peak height and importance?\nIndeed, {retained_cor} is the average correlation for the top {top_cutoff:.0%} peaks."
)

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(
    peak_max_scores["perc"], peak_max_scores["retained_cor"], s=0.3, color="#3335"
)
ax.plot(peak_max_scores["perc"], peak_max_scores["cum_retained_cor"])
ax.set_ylim(-1, 1)
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
annot = f"{retained_cor:.2f}"
ax.annotate(
    annot,
    xy=(top_cutoff, retained_cor),
    xycoords=("data", "data"),
    xytext=(0, -10),
    textcoords="offset points",
    va="top",
    ha="center",
    color="red",
    bbox=dict(fc="#FFFFFF88"),
    arrowprops={"arrowstyle": "-", "ec": "red"},
)
ax.set_xlabel("Top peaks (acording to $\Delta$ cor)")
ax.set_title(
    "Correlation within peaks of # fragments and $\Delta$ cor", rotation=0, loc="left"
)

# %% [markdown]
# ### Summarizing special genes

# %%
special_genes.any(1).loc[genes_oi].mean()

# %%
gene_best_windows = gene_best_windows.join(
    peak_window_matches.set_index(["gene", "window"])
).reset_index(level="window")
gene_best_windows = gene_best_windows.groupby("gene").first()

# %%
gene_best_windows["matched"] = ~pd.isnull(gene_best_windows["peak"])

# %%
gene_best_windows = gene_best_windows.sort_values("mse_diff", ascending=False)
gene_best_windows["ix"] = np.arange(1, gene_best_windows.shape[0] + 1)
gene_best_windows["cum_matched"] = (
    np.cumsum(gene_best_windows["matched"]) / gene_best_windows["ix"]
)
gene_best_windows["perc"] = gene_best_windows["ix"] / gene_best_windows.shape[0]

# %% [markdown]
# Of the top 5% most predictive genes, how many are inside a peak?

# %%
top_cutoff = 0.05
perc_within_a_peak = gene_best_windows["cum_matched"].iloc[
    int(gene_best_windows.shape[0] * top_cutoff)
]
print(perc_within_a_peak)
print(
    f"Perhaps the most predictive window in the promoter is not inside of a peak?\nIndeed, for {1-perc_within_a_peak:.2%} of the {top_cutoff:.0%} best predicted genes, the most predictive window does not lie within a peak."
)

# %%
fig, ax = plt.subplots()
ax.plot(gene_best_windows["perc"], gene_best_windows["cum_matched"])
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_xlabel("Top genes (acording to mse_loss)")
ax.set_ylabel(
    "% of genes for which\nthe top window is\ncontained in a peak",
    rotation=0,
    ha="right",
    va="center",
)

# %%
gene_best_windows["label"] = transcriptome.symbol(gene_best_windows.index)

# %%
# if you're interested in genes where the best window is not inside a peak
gene_best_windows.query("~matched").sort_values("mse_diff", ascending=False)

# %%
special_genes["most_predictive_position_not_in_peak"] = ~gene_best_windows["matched"]
