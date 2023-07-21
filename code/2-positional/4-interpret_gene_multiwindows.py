# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import IPython

if IPython.get_ipython():
    IPython.get_ipython().magic("load_ext autoreload")
    IPython.get_ipython().magic("autoreload 2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

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

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"


# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# %%
# symbol = "JCHAIN"
# symbol = "SELL"
# symbol = "TXNDC5"
# symbol = "CCR7"
symbol = "CTLA4"
# symbol = "CD79A"
# symbol = "VPREB3"
symbol = "IRS2"
symbol = "NAMPT"
# symbol = "TRAF5"
# symbol = "FOXO3"
# symbol = "IL1B"
# symbol = "SPI1"
# symbol = "TCF3"
# symbol = "CCL4"
# symbol = "CD74"
# symbol = "TCF4"
# symbol = transcriptome.symbol("ENSG00000122862")
genes_oi = transcriptome.var["symbol"] == symbol
gene = transcriptome.var.index[genes_oi][0]

gene_ix = transcriptome.gene_ix(symbol)
gene = transcriptome.var.iloc[gene_ix].name

# %%
sc.pl.umap(transcriptome.adata, color=gene, use_raw=False, show=False)

# %% [markdown]
# ## Window

# %%
# load nothing scoring
scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)

# %% [markdown]
# ### Load

# %%
scores_folder = prediction.path / "scoring" / "multiwindow_gene" / gene
multiwindow_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
import scipy.stats


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


x = (
    multiwindow_scoring.genescores["deltacor"]
    .sel(gene=gene, phase=["test", "validation"])
    .stack({"model_phase": ["model", "phase"]})
    .values.T
)
scores = []
for i in range(x.shape[1]):
    scores.append(scipy.stats.ttest_1samp(x[:, i], 0, alternative="less").pvalue)
scores = pd.DataFrame({"pvalue": scores})
scores["qval"] = fdr(scores["pvalue"])

# %%
sns.scatterplot(
    x=multiwindow_scoring.genescores["deltacor"]
    .sel(phase="test")
    .mean("model")
    .values.flatten(),
    y=multiwindow_scoring.genescores["deltacor"]
    .sel(phase="validation")
    .mean("model")
    .values.flatten(),
    # hue=np.log1p(
    #     multiwindow_scoring.genescores["lost"]
    #     .sel(phase="validation")
    #     .mean("model")
    #     .values.flatten()
    # ),
    hue=np.log1p(multiwindow_scoring.design["window_size"]),
)

# %%
plotdata = (
    multiwindow_scoring.genescores.mean("model").sel(gene=gene).stack().to_dataframe()
)
plotdata = multiwindow_scoring.design.join(plotdata)

plotdata.loc["validation", "qval"] = scores["qval"].values
plotdata.loc["test", "qval"] = scores["qval"].values

# %%
window_sizes_info = pd.DataFrame(
    {"window_size": multiwindow_scoring.design["window_size"].unique()}
).set_index("window_size")
window_sizes_info["ix"] = np.arange(len(window_sizes_info))

# %%
fig, ax = plt.subplots(figsize=(20, 3))

deltacor_norm = mpl.colors.Normalize(0, 0.001)
deltacor_cmap = mpl.cm.Reds

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = (
        plotdata.query("window_size == @window_size").query("phase == 'test'").iloc[::2]
    )
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        rect = mpl.patches.Rectangle(
            (plotdata_row["window_start"], y),
            plotdata_row["window_end"] - plotdata_row["window_start"],
            1,
            lw=0,
            fc=deltacor_cmap(deltacor_norm(-plotdata_row["deltacor"])),
        )
        ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)
ax.axvline(6000)
ax.set_yticks(window_sizes_info["ix"] + 0.5)
ax.set_yticklabels(window_sizes_info.index)

# %%
fig, ax = plt.subplots(figsize=(20, 3))

qval_norm = mpl.colors.Normalize(0, 0.1)
qval_cmap = mpl.cm.Greens_r

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size").query("phase == 'test'")
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        rect = mpl.patches.Rectangle(
            (plotdata_row["window_start"], y),
            plotdata_row["window_end"] - plotdata_row["window_start"],
            1,
            lw=0,
            fc=qval_cmap(qval_norm(plotdata_row["qval"])),
        )
        ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)
ax.axvline(6000)
ax.set_yticks(window_sizes_info["ix"] + 0.5)
ax.set_yticklabels(window_sizes_info.index)

# %%
fig, ax = plt.subplots(figsize=(20, 3))

effect_norm = mpl.colors.CenteredNorm()
effect_cmap = mpl.cm.RdBu_r

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("phase == 'validation'").query(
        "window_size == @window_size"
    )
    print(plotdata_oi.shape)
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        if plotdata_row["qval"] < 0.1:
            rect = mpl.patches.Rectangle(
                (plotdata_row["window_start"], y),
                plotdata_row["window_end"] - plotdata_row["window_start"],
                1,
                lw=0,
                fc=effect_cmap(effect_norm(-plotdata_row["effect"])),
            )
            ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)

# %% [markdown]
# ### Interpolate per position

# %%
positions_oi = np.arange(*window)

# %%
deltacor_test_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
deltacor_validation_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
retained_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
lost_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
for window_size, window_size_info in window_sizes_info.iterrows():
    # for window_size, window_size_info in window_sizes_info.query(
    #     "window_size == 200"
    # ).iterrows():
    plotdata_oi = plotdata.query("phase in ['validation']").query(
        "window_size == @window_size"
    )

    x = plotdata_oi["window_mid"].values.copy()
    y = plotdata_oi["deltacor"].values.copy()
    y[plotdata_oi["qval"] > 0.1] = 0.0
    deltacor_interpolated_ = np.clip(
        np.interp(positions_oi, x, y) / window_size * 1000,
        -np.inf,
        0,
        # np.inf,
    )
    deltacor_validation_interpolated[window_size_info["ix"], :] = deltacor_interpolated_
    plotdata_oi = plotdata.query("phase in ['test']").query(
        "window_size == @window_size"
    )
    x = plotdata_oi["window_mid"].values.copy()
    y = plotdata_oi["deltacor"].values.copy()
    y[plotdata_oi["qval"] > 0.1] = 0.0
    deltacor_interpolated_ = np.clip(
        np.interp(positions_oi, x, y) / window_size * 1000,
        -np.inf,
        0,
        # np.inf,
    )
    deltacor_test_interpolated[window_size_info["ix"], :] = deltacor_interpolated_

    retained_interpolated_ = (
        np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["retained"])
        / window_size
        * 1000
    )
    retained_interpolated[window_size_info["ix"], :] = retained_interpolated_
    lost_interpolated_ = (
        np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["lost"])
        / window_size
        * 1000
    )
    lost_interpolated[window_size_info["ix"], :] = lost_interpolated_

# %%
# save
interpolated = {
    "deltacor_validation": deltacor_validation_interpolated,
    "deltacor_test": deltacor_test_interpolated,
    "retained": retained_interpolated,
    "lost": lost_interpolated,
}
pickle.dump(interpolated, (scores_folder / "interpolated.pkl").open("wb"))

# %%
# plot
fig, ax = plt.subplots(figsize=(20, 1))
ax.plot(positions_oi, deltacor_validation_interpolated.mean(0), label="validation")
ax.plot(positions_oi, deltacor_test_interpolated.mean(0), label="test")
ax.legend()
ax.invert_yaxis()
ax2 = ax.twinx()
ax2.plot(positions_oi, lost_interpolated.mean(0), color="red", alpha=0.6)
ax2.set_ylabel("retained")

# %% [markdown]
# ## Plot

# %%
promoter = promoters.loc[gene]


# %%
plotdata_predictive = pd.DataFrame(
    {
        "deltacor": interpolated["deltacor_test"].mean(0),
        "lost": interpolated["lost"].mean(0),
        "position": pd.Series(np.arange(*window), name="position"),
    }
)

# %%
import chromatinhd.grid

main = chd.grid.Grid(3, 3, padding_width=0.1, padding_height=0.1)
fig = chd.grid.Figure(main)

padding_height = 0.001
resolution = 0.0003
panel_width = (window[1] - window[0]) * resolution

# gene annotation
genome_folder = folder_data_preproc / "genome"
genes_panel = main[0, 0] = chdm.plotting.Genes(
    promoter,
    genome_folder=genome_folder,
    window=window,
    width=panel_width,
)

peaks_folder = chd.get_output() / "peaks" / dataset_name
peaks_panel = main[2, 0] = chdm.plotting.Peaks(
    promoter,
    peaks_folder,
    window=window,
    width=panel_width,
    row_height=0.8,
)

predictive_panel = main[1, 0] = chd.models.pred.plot.Predictivity(
    plotdata_predictive,
    window,
    panel_width,
)
# %%
fig.plot()
fig

# %% [markdown]
# ## Variants/haplotypes

# %%
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
promoter = promoters.loc[gene]

# %%
motifscan_name = "gwas_immune"

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on="snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

# %%
association_oi = association.loc[
    (association["chr"] == promoter["chr"])
    & (association["pos"] >= promoter["start"])
    & (association["pos"] <= promoter["end"])
].copy()

# %%
association_oi["position"] = (association_oi["pos"] - promoter["tss"]) * promoter[
    "strand"
]

# %%
variants = pd.DataFrame(
    {
        "disease/trait": association_oi.groupby("snp")["disease/trait"].apply(list),
        "snp_main_first": association_oi.groupby("snp")["snp_main"].first(),
    }
)
variants = variants.join(snp_info)
variants["position"] = (variants["start"] - promoter["tss"]) * promoter["strand"]

haplotypes = (
    association_oi.groupby("snp_main")["snp"]
    .apply(lambda x: sorted(set(x)))
    .to_frame("snps")
)
haplotypes["color"] = sns.color_palette("hls", n_colors=len(haplotypes))

# %% [markdown]
# ### Compare to individual position ranking

# %%
fig, ax = plt.subplots(figsize=(20, 3))
ax.plot(
    positions_oi * promoter["strand"] + promoter["tss"],
    deltacor_test_interpolated.mean(0),
)
ax2 = ax.twinx()
ax2.plot(
    positions_oi * promoter["strand"] + promoter["tss"],
    retained_interpolated.mean(0),
    color="red",
    alpha=0.6,
)
ax2.set_ylabel("retained")

for _, variant in variants.iterrows():
    ax.scatter(
        variant["position"] * promoter["strand"] + promoter["tss"],
        0.9,
        color=haplotypes.loc[variant["snp_main_first"], "color"],
        s=100,
        marker="|",
        transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
    )

ax.invert_yaxis()
ax2.invert_yaxis()

# %%
import gseapy

# %%
rnk = pd.Series(0, index=pd.Series(list("abcdefghijklmnop")))
rnk.values[:] = -np.arange(len(rnk))
genesets = {"hi": ["a", "b", "c"]}

# %%
rnk = -pd.Series(deltacor_test_interpolated.mean(0), index=positions_oi.astype(str))
genesets = {"hi": np.unique(variants["position"].astype(str).values)}

# %%
# ranked = gseapy.prerank(rnk, genesets, min_size = 0)

# %%
rnk_sorted = pd.Series(np.sort(np.log(rnk)), index=rnk.index)
# rnk_sorted = pd.Series(np.sort(rnk), index = rnk.index)
fig, ax = plt.subplots()
sns.ecdfplot(rnk_sorted, ax=ax)
sns.ecdfplot(
    rnk_sorted[variants["position"].astype(int).astype(str)], ax=ax, color="orange"
)
for _, motifdatum in variants.iterrows():
    rnk_motif = rnk_sorted[str(int(motifdatum["position"]))]
    q = np.searchsorted(rnk_sorted, rnk_motif) / len(rnk_sorted)
    ax.scatter([rnk_motif], [q], color="red")
    # ax.scatter(motifdatum["position"], 0, color = "red", s = 5, marker = "|")
