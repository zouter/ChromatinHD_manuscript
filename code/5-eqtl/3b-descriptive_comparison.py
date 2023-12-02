# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdmpbmc10k_eqtl

# %%
import chromatinhd as chd
import tempfile

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    default = "pbmc10k_eqtl",
    # default = "GSE198467_H3K27ac",
    # default = "brain",
    # default = "pbmc10k"
)
parser.add_argument("--promoter_name", default = "10k10k")
# parser.add_argument("--latent_name", default = "celltype")
parser.add_argument("--latent_name", default = "leiden_0.1")
parser.add_argument("--method_name", default = 'v9_128-64-32')

try:
    get_ipython().__class__.__name__
    in_jupyter = True
except:
    in_jupyter = False
globals().update(vars(parser.parse_args("" if in_jupyter else None)))

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / dataset_name


# %% [markdown]
# ### Load data

# %%
class Prediction(chd.flow.Flow):
    pass
prediction = Prediction(chd.get_output() / "prediction_likelihood" / dataset_name / promoter_name / latent_name / method_name)
# model = chd.load((prediction.path / "model_0.pkl").open("rb"))

# %%
probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
design = pickle.load((prediction.path / "design.pkl").open("rb"))

probs_diff = probs - probs.mean(1, keepdims=True)

# %%
design["gene_ix"] = design["gene_ix"]

# %%
window = {
    "10k10k":np.array([-10000, 10000])
}[promoter_name]
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
fragments.obs["cluster"] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

# %% [markdown]
# Interpolate probs for individual positions

# %%
x = (design["coord"].values).astype(int).reshape((len(design["gene_ix"].cat.categories), len(design["active_latent"].cat.categories), len(design["coord"].cat.categories)))
desired_x = torch.arange(*window)
probs_interpolated = chd.utils.interpolate_1d(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)).numpy()

# %% [markdown]
# ## Load slices

# %%
prob_cutoff = np.log(1.)

# %%
scores_dir = prediction.path / "scoring" / "significant_up"
pureregionresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %% [markdown]
# ### Relative to peaks

# %%
import chromatinhd.peakcounts
# peakcaller = "cellranger"; peakcaller_label = "Cellranger"
# peakcaller = "macs2"
peakcaller = "macs2_improved";peakcaller_label = "MACS2"
# peakcaller = "encode_screen";peakcaller_label = "ENCODE SCREEN"
# peakcaller = "macs2_leiden_0.1_merged";peaks_label = "MACS2 clusters merged"
# peakcaller = "rolling_500";peaks_label = "Sliding window 500"
# peakcaller = "rolling_50";peaks_label = "Sliding window 50"
# peakcaller = "genrich"

# diffexp = "signac"
diffexp = "scanpy"

# %%
scores_dir = (
    chd.get_output()
    / "prediction_differential"
    / dataset_name
    / promoter_name
    / latent_name
    / diffexp
    / peakcaller
)
peakresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %%
scores_dir = prediction.path / "scoring" / peakcaller / diffexp
regionresult = pickle.load((scores_dir / "slices.pkl").open("rb"))

# %%
# peakscores = pd.read_csv(chd.get_output() / "tmp" / "results.csv", index_col = 0)

# peakscores["cluster"] = pd.Categorical(
#     peakscores["cluster"], categories=cluster_info.index
# )
# peakcounts = chd.peakcounts.FullPeak(
#     folder=chd.get_output()
#     / "peakcounts"
#     / dataset_name
#     / peakcaller
# )

# peakscores_joined = pd.merge(peakscores, peakcounts.peaks, on = "peak")
# peakscores_joined["logfoldchanges"] = peakscores_joined["avg_log2FC"]
# peakscores_joined["pvals_adj"] = peakscores_joined["p_val_adj"]

# peakresult = (
#     chromatinhd.differential.DifferentialSlices.from_peakscores(
#         peakscores_joined, window, len(transcriptome.var), logfoldchanges_cutoff = 0.1
#     )
# )

# %% [markdown]
# ## Positional overlap

# %%
method_info = pd.DataFrame([
    ["peak", "#FF4136", f"Unique to differential {peakcaller_label}"],
    ["common", "#B10DC9", f"Common"],
    ["region", "#0074D9", f"Unique to ChromatinHD"]
], columns = ["method", "color", "label"]).set_index("method")

# %%
position_chosen_region = regionresult.position_chosen
position_chosen_peak = peakresult.position_chosen

# %%
(position_chosen_region & position_chosen_peak).sum() / (position_chosen_region | position_chosen_peak).sum()

# %%
position_region = np.where(position_chosen_region)[0]
position_peak = np.where(position_chosen_peak)[0]
position_indices_peak_unique = np.where(position_chosen_peak & (~position_chosen_region))[0]
position_indices_region_unique = np.where(position_chosen_region & (~position_chosen_peak))[0]
position_indices_common = np.where(position_chosen_region & position_chosen_peak)[0]
position_indices_intersect = np.where(position_chosen_region | position_chosen_peak)[0]

# %%
positions_region_unique = (position_indices_region_unique % (window[1] - window[0])) + window[0]
positions_region = (position_region % (window[1] - window[0])) + window[0]

positions_peak_unique = (position_indices_peak_unique % (window[1] - window[0])) + window[0]
positions_peak = (position_peak % (window[1] - window[0])) + window[0]

positions_common = (position_indices_common % (window[1] - window[0])) + window[0]

positions_intersect = (position_indices_intersect % (window[1] - window[0])) + window[0]

# %%
binmids = np.linspace(*window, 200 + 1)
cuts = (binmids + (binmids[1] - binmids[0])/2)[:-1]

# %%
positions_region_unique_bincounts = np.bincount(np.digitize(positions_region_unique, cuts, right = True), minlength = len(cuts) + 1)
positions_region_bincounts = np.bincount(np.digitize(positions_region, cuts), minlength = len(cuts) + 1)

positions_peak_unique_bincounts = np.bincount(np.digitize(positions_peak_unique, cuts), minlength = len(cuts) + 1)
positions_peak_bincounts = np.bincount(np.digitize(positions_peak, cuts), minlength = len(cuts) + 1)

positions_common_bincounts = np.bincount(np.digitize(positions_common, cuts), minlength = len(cuts) + 1)

positions_intersect_bincounts = np.bincount(np.digitize(positions_intersect, cuts), minlength = len(cuts) + 1)

# %%
fig, ax = plt.subplots()
ax.plot(binmids, (positions_common_bincounts) / (positions_intersect_bincounts) / (positions_common_bincounts.sum() / positions_intersect_bincounts.sum()), label = "common")
ax.plot(binmids, (positions_peak_unique_bincounts / positions_peak_bincounts) / (positions_peak_unique_bincounts.sum() / positions_peak_bincounts.sum()), label = "peak_unique")
ax.plot(binmids, (positions_region_unique_bincounts / positions_region_bincounts) / (positions_region_unique_bincounts.sum() / positions_region_bincounts.sum()), label = "region_unique")
ax.axhline(1, dashes = (2, 2), color = "#333")
ax.set_yscale("log")
ax.legend()
ax.set_yticks([1/2, 1, 2, 1/4])
ax.set_yticklabels([1/2, 1, 2, 1/4])

# %%
plotdata = pd.DataFrame({
    "common":positions_common_bincounts,
    "peak_unique":positions_peak_unique_bincounts,
    "region_unique":positions_region_unique_bincounts,
    "intersect":positions_intersect_bincounts,
    "position":binmids
})
plotdata["peak_unique_density"] = plotdata["peak_unique"]/plotdata["intersect"]
plotdata["common_density"] = plotdata["common"]/plotdata["intersect"]
plotdata["region_unique_density"] = plotdata["region_unique"]/plotdata["intersect"]

# %%
plotdata_last = plotdata[["peak_unique_density", "common_density", "region_unique_density"]].iloc[-10].to_frame(name = "density")
plotdata_last["cumulative_density"] = np.cumsum(plotdata_last["density"]) - plotdata_last["density"]/2
plotdata_mean = pd.Series({
    "peak_unique_density":positions_peak_unique_bincounts.sum() / positions_intersect_bincounts.sum(),
    "region_unique_density":positions_region_unique_bincounts.sum() / positions_intersect_bincounts.sum(),
    "common_density":positions_common_bincounts.sum() / positions_intersect_bincounts.sum()
})

# %%
fig, ax = plt.subplots(figsize = (3, 3))
ax.stackplot(
    binmids,
    plotdata["peak_unique_density"],
    plotdata["common_density"],
    plotdata["region_unique_density"],
    baseline = "zero",
    colors = method_info["color"],
    lw = 1,
    ec = "#FFFFFF33"
)
ax.set_xlim(*window)
ax.set_ylim(0, 1)
transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
x = 1.02
ax.text(
    x,
    plotdata_last.loc["peak_unique_density", "cumulative_density"],
    f"{method_info.loc['peak', 'label']}\n{plotdata_mean['peak_unique_density']:.0%}",
    transform = transform, ha = "left", va = "center",
    color = method_info.loc["peak", "color"]
)
ax.text(
    x,
    plotdata_last.loc["common_density", "cumulative_density"],
    f"{method_info.loc['common', 'label']}\n{plotdata_mean['common_density']:.0%}",
    transform = transform, ha = "left", va = "center",
    color = method_info.loc["common", "color"]
)
ax.text(
    x,
    plotdata_last.loc["region_unique_density", "cumulative_density"],
    f"{method_info.loc['region', 'label']}\n{plotdata_mean['region_unique_density']:.0%}",
    transform = transform, ha = "left", va = "center",
    color = method_info.loc["region", "color"]
)

ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_xlabel("Distance to TSS")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6, 3), sharey=True)
ax1.stackplot(
    binmids,
    plotdata["peak_unique"],
    plotdata["common"],
    plotdata["region_unique"],
    baseline = "zero",
    colors = method_info["color"],
    lw = 1,
    ec = "#FFFFFF33"
)
ax1.set_xlim(*window)

ax2.stackplot(
    binmids,
    plotdata["peak_unique"],
    plotdata["common"],
    plotdata["region_unique"],
    baseline = "zero",
    colors = method_info["color"],
    lw = 1,
    ec = "#FFFFFF33"
)
ax2.set_xlim(-3000, 3000)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6, 3), sharey=True)
ax = ax1
ax.fill_between(binmids, plotdata["common"], color = method_info.loc["common", "color"])
ax.fill_between(binmids, plotdata["common"], plotdata["region_unique"], color = method_info.loc["region", "color"])
ax.fill_between(binmids, -plotdata["peak_unique"], color = method_info.loc["peak", "color"])
ax.axvline(0, dashes = (2, 2), color = "#333", lw = 1)

ax = ax2
ax.fill_between(binmids, plotdata["common"], color = method_info.loc["common", "color"])
ax.fill_between(binmids, plotdata["common"], plotdata["region_unique"], color = method_info.loc["region", "color"])
ax.fill_between(binmids, -plotdata["peak_unique"], color = method_info.loc["peak", "color"])
ax.axvline(0, dashes = (2, 2), color = "#333", lw = 1)
ax.set_xlim(-3000, 3000)

transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
x = 1.02
ax.text(
    x,
    0.3,
    f"{method_info.loc['peak', 'label']}\n{plotdata_mean['peak_unique_density']:.0%}",
    transform = ax.transAxes, ha = "left", va = "center",
    color = method_info.loc["peak", "color"]
)
ax.text(
    x,
    0.6,
    f"{method_info.loc['common', 'label']}\n{plotdata_mean['common_density']:.0%}",
    transform = ax.transAxes, ha = "left", va = "center",
    color = method_info.loc["common", "color"]
)
ax.text(
    x,
    0.8,
    f"{method_info.loc['region', 'label']}\n{plotdata_mean['region_unique_density']:.0%}",
    transform = ax.transAxes, ha = "left", va = "center",
    color = method_info.loc["region", "color"]
)

# %% [markdown]
# ## Bystander regions

# %%
probs_diff_interpolated = probs_interpolated - probs_interpolated.mean(1, keepdims = True)

# %%
probs_interpolated.shape

# %%
print(
    probs_diff_interpolated.flatten()[position_indices_peak_unique].mean(),
    probs_diff_interpolated.flatten()[position_indices_region_unique].mean(),
    probs_diff_interpolated.flatten()[position_indices_common].mean()
)

# %%
fig, ax = plt.subplots(figsize = (2, 2))
sns.ecdfplot(np.exp(probs_diff_interpolated.flatten()[position_indices_peak_unique][:1000000]), color = method_info.loc["peak", "color"], ax = ax)
sns.ecdfplot(np.exp(probs_diff_interpolated.flatten()[position_indices_region_unique][:1000000]), color = method_info.loc["region", "color"], ax = ax)
sns.ecdfplot(np.exp(probs_diff_interpolated.flatten()[position_indices_common][:1000000]), color = method_info.loc["common", "color"], ax = ax)
ax.set_xscale("log")
ax.set_xlim(0.5, 8)
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(["½", 1, 2, 4, 8])
ax.axvline(1, color = "grey", zorder = -1)
ax.set_xlabel("Accessibility fold change")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))

# %% [markdown]
# ### Per-peak look

# %%
peakscores = peakresult.get_slicescores().join(peakresult.get_slicedifferential(probs_interpolated)).join(peakresult.get_sliceaverages(probs_interpolated))

# %%
fig, ax = plt.subplots(figsize = (2, 2))
sns.ecdfplot(peakscores["differential_positions"])
ax.set_xlabel("% differential positions\nwithin peak")
ax.set_ylabel("% peaks")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_yticks([0, 0.5, 1])

# %%
upstream_cutoff = -1000 - window[0]
downstream_cutoff = +1000 - window[0]

def classify_position_group(x, upstream_cutoff, downstream_cutoff):
    return np.array(["upstream", "downstream", "promoter"])[np.argmax(np.vstack([x.end < upstream_cutoff, x.start > downstream_cutoff, np.repeat(True, len(x))]), 0)]

position_group_info = pd.DataFrame([
    ["upstream", "Upstream -10kb→-1kb"],
    ["promoter", "Promoter -1kb→+1kb"],
    ["downstream", "Downstream +1kb→+10kb"],
], columns = ["position_group", "label"]).set_index("position_group")

# %%
peakscores["position_group"] = classify_position_group(peakscores, upstream_cutoff, downstream_cutoff)

# %%
fig, ax = plt.subplots(figsize = (2, 2))
for position_group, peakscores_group in peakscores.groupby("position_group"):
    sns.ecdfplot(peakscores_group["differential_positions"], label = position_group)
plt.legend(bbox_to_anchor=(0.5, -0.4), loc="upper center", fontsize = 8, ncol = 3)
ax.set_xlabel("% differential positions\nwithin peak")
ax.set_ylabel("% peaks")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_yticks([0, 0.5, 1])

# %% [markdown]
# ### Examples

# %%
slices_oi = peakscores.query("max_lfc > log(4)").sort_values("differential_positions").head(10)
# slices_oi = peakscores.query("max_lfc > log(4)").assign(group = lambda x:(x.passenger * 10)//1).groupby("group").first()
# https://media.tenor.com/wn2_Qq6flogAAAAM/magical-magic.gif
slices_oi = slices_oi.sort_values("length").iloc[np.concatenate([np.arange(len(slices_oi), step = 2), np.arange(len(slices_oi), 0, step = -2) - 1])]

# %%
probs_diff = probs - probs.mean(1, keepdims=True)

# %%
region_position_chosen = pureregionresult.position_chosen.reshape((fragments.n_genes, len(cluster_info), (window[1] - window[0])))
peak_position_chosen = peakresult.position_chosen.reshape((fragments.n_genes, len(cluster_info), (window[1] - window[0])))

# %%
import chromatinhd.grid
main = chd.grid.Grid()
wrap = main[0, 0] = chd.grid.Wrap(5, padding_width = 0.1, padding_height = 0.3)
fig = chd.grid.Figure(main)

padding_height = 0.001
resolution = 0.0005

panel_height = 0.5

total_width_cutoff = 10

hatch_color = "#FFF4"

for slice_oi in slices_oi.to_dict(orient="records"):
    slice_oi = dict(slice_oi)

    expanded_slice_oi = slice_oi.copy()
    expanded_slice_oi["start"] = np.clip(slice_oi["start"] - 1000, *(window - window[0]))
    expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 1000, *(window - window[0]))

    window_oi = np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]

    gene_oi = expanded_slice_oi["gene_ix"]
    cluster_ix = expanded_slice_oi["cluster_ix"]
    cluster_info_oi = cluster_info.iloc[[cluster_ix]]

    plotdata_atac = design.query("gene_ix == @gene_oi").copy().rename(columns = {"active_latent":"cluster"}).set_index(["coord", "cluster"]).drop(columns = ["batch", "gene_ix"])
    plotdata_atac["prob"] = probs[gene_oi].flatten()
    plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

    plotdata_atac["prob"] = plotdata_atac["prob"] - np.log(plotdata_atac.reset_index().groupby(["cluster"]).apply(lambda x:np.trapz(np.exp(x["prob"]), x["coord"].astype(float) / (window[1] - window[0])))).mean()
    plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

    resolution = 0.0005
    panel_width = (window_oi[1] - window_oi[0]) * resolution

    # differential atac
    wrap_differential = chd.differential.plot.Differential(
        plotdata_atac,
        plotdata_atac_mean,
        cluster_info_oi,
        window_oi,
        panel_width,
        panel_height,
        padding_height = padding_height,
        title = False
    )
    wrap.add(wrap_differential)

    ax = wrap_differential.elements[0].ax
    
    # add region and peak unique spans  
    region_position_chosen_oi = region_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]
    peak_position_chosen_oi = peak_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]
    
    chd.differential.plot.CommonUnique(ax, peak_position_chosen_oi, region_position_chosen_oi, expanded_slice_oi, window, method_info)
    
    # add labels
    start = slice_oi['start'] + window[0]
    end = slice_oi['end'] + window[0]
    
    gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
    cluster_label = cluster_info.query("dimension == @cluster_ix")["label"][0]
    position_label = str(int(slice_oi["start"] + slice_oi["length"]/2) + window[0])
    text = ax.annotate(
        f"$\\it{{{gene_label}}}$ $\\bf{{{cluster_label}}}$",
        (0,1),
        (2,2),
        va = "bottom",
        ha = "left",

        xycoords = "axes fraction",
        textcoords = "offset points",
        fontsize = 6,
        color = "#999",
        zorder = 200
    )
    text.set_path_effects([
        mpl.patheffects.Stroke(foreground = "#FFFFFFFF", linewidth = 2),
        mpl.patheffects.Normal()
    ])

    trans = mpl.transforms.blended_transform_factory(y_transform=ax.transAxes, x_transform=ax.transData)
    text = ax.annotate(
        f"{start:+}", (start,1), (-2,-2),va = "top",ha = "right",
        xycoords = trans,
        textcoords = "offset points",
        fontsize = 6,
        color = "#999",
        zorder = 200
    )
    text.set_path_effects([
        mpl.patheffects.Stroke(foreground = "white", linewidth = 2),
        mpl.patheffects.Normal()
    ])
    text = ax.annotate(
        f"{end:+}", (end,1), (2,-2),va = "top",ha = "left",
        xycoords = trans,
        textcoords = "offset points",
        fontsize = 6,
        color = "#999",
        zorder = 200
    )
    text.set_path_effects([
        mpl.patheffects.Stroke(foreground = "white", linewidth = 2),
        mpl.patheffects.Normal()
    ])
    

    
legend_panel = main[0, 1] = chd.grid.Panel((1, panel_height * 2))
legend_panel.ax.axis("off")
# legend_panel.ax.plot([0, 1], [0, 1])

fig.plot()

# %% [markdown]
# ### Enrichment

# %% [markdown]
# TODO

# %%
motifscan_name = "cutoff_0001"
# motifscan_name = "onek1k_0.2"
# motifscan_name = "gwas_immune"
# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_cns"
# motifscan_name = "gtex"

# %%
motifscan_folder = chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
motifscan = chd.data.Motifscan(motifscan_folder)
motifscan.n_motifs = len(motifscan.motifs)

# %%
# position_indices = np.repeat(np.arange(fragments.n_genes * (window[1] - window[0])), np.diff(motifscan.indptr))[(motifscan.indices == motif_ix_oi)]
# position_chosen = np.zeros(fragments.n_genes * (window[1] - window[0]), dtype = bool)
# position_chosen[position_indices] = True

# %%
# nuc = torch.where(onehot_promoters[:-1])[1]

# %%
# plt.plot(torch.bincount(nuc[1:] + (nuc[:-1]*4)))

# %%
peakscores_oi = peakscores

# cluster_oi = "leiden_0"
# cluster_oi = "Monocytes"
# cluster_oi = "B"
cluster_ixs = [cluster_info.loc[cluster_oi, "dimension"]]

cluster_ixs = cluster_info["dimension"]

# %%
peakscores_oi = peakscores_oi.assign(differential_positions_group = lambda x:(x["differential_positions"] * 5)//1)

# %%
groupenrichments = []
for differential_positions_group, peakscores_group in peakscores_oi.groupby("differential_positions_group"):
    print(differential_positions_group, peakscores_group.shape)
    motifscores_region = chd.differential.enrichment.enrich_windows(
        motifscan,
        peakscores_oi[["start", "end"]].values,
        peakscores_oi["gene_ix"].values,
        oi_slices = (peakscores_oi["cluster_ix"].isin(cluster_ixs) & (peakscores_oi["differential_positions_group"] == differential_positions_group)).values,
        background_slices = (~peakscores_oi["cluster_ix"].isin([cluster_ix])).values,
        n_genes = fragments.n_genes,
        window = window,
        n_background = 1
    )
    motifscores_region["differential_positions_group"] = differential_positions_group
    
    groupenrichments.append(motifscores_region)
groupenrichments = pd.concat(groupenrichments).reset_index()
groupenrichments = groupenrichments.reset_index().set_index(["differential_positions_group", "motif"])

# %%
groupenrichments["perc_gene_mean"] = [x.mean() for x in groupenrichments["perc_gene"]]
# typeenrichments["perc_gene_mean"] = [x[transcriptome.var["chr"] == "chr6"].mean() for x in typeenrichments["perc_gene"]]

# %%
groupenrichments.sort_values("logodds", ascending = False).head(15)

# %%
groupenrichments["significant"] = groupenrichments["qval"] < 0.05

# %%
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("SPI1")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("HXA13")][0]
motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("monoc")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("Rheumatoid arthritis")][0]
# motif_id = motifscan.motifs.index[motifscan.motifs.index.str.contains("Whole Blood")][0]

# %%
fig, ax = plt.subplots(figsize = (3, 3))
plotdata = groupenrichments.xs(motif_id, level = "motif")
ax.bar(np.arange(len(plotdata)), plotdata["logodds"])
# ax.set_xticks(np.arange(len(plotdata)))
# ax.set_xticklabels(plotdata.index)
# chd.slicetypes.label_axis(ax, ax.xaxis)

# %%
sns.histplot(data = groupenrichments, x = "logodds", hue = "differential_positions_group")

# %% [markdown]
# ## "Background"

# %% [markdown]
# ### Examples

# %%
region_position_chosen = pureregionresult.position_chosen.reshape((fragments.n_genes, len(cluster_info), (window[1] - window[0])))
peak_position_chosen = peakresult.position_chosen.reshape((fragments.n_genes, len(cluster_info), (window[1] - window[0])))

# %%
slicetopologies = regionresult.get_slicetopologies(probs_interpolated).join(regionresult.get_slicescores())

# %%
slices_oi = slicetopologies.query("dominance < 0.4").query("(mid > (@window[0] + 500)) & (mid < (@window[1] - 500))").query("length > 100").sort_values("differentialdominance", ascending = False).head(10)

# %%
peak_methods = pd.DataFrame([
    ["cellranger", False, "Cellranger"],
    ["macs2_improved", False, "MACS2"],
    ["macs2_" + latent_name, True, "MACS2 (cluster)"],
    ["genrich", False, "Genrich"],
], columns = ["method", "cluster_specific", "label"]).set_index("method")
peak_methods["ix"] = np.arange(len(peak_methods))

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)


# %%
def center_peaks(peaks, promoter):
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns = ["start", "end", "method"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] - promoter["tss"]) * promoter["strand"],
                (peak["end"] - promoter["tss"]) * promoter["strand"]
            ][::promoter["strand"]]

            for _, peak in peaks.iterrows()
        ]
    return peaks

def get_promoter_peaks(promoter, peak_methods):
    peaks = []

    import pybedtools
    promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(promoter).T[["chr", "start", "end"]])

    for peaks_name, (cluster_specific, ) in peak_methods[["cluster_specific"]].iterrows():
        peaks_bed = pybedtools.BedTool(chd.get_output() / "peaks" / dataset_name / peaks_name / "peaks.bed")

        if cluster_specific:
            usecols = [0, 1, 2, 6]
            names = ["chr", "start", "end", "name"]
        else:
            usecols = [0, 1, 2]
            names = ["chr", "start", "end"]
        peaks_cellranger = promoter_bed.intersect(peaks_bed, wb = True, nonamecheck = True).to_dataframe(usecols = usecols, names = names)

        if cluster_specific:
            peaks_cellranger = peaks_cellranger.rename(columns = {"name":"cluster"})
            peaks_cellranger["cluster"] = peaks_cellranger["cluster"].astype(int)

        # peaks_cellranger = promoter_bed.intersect(peaks_bed).to_dataframe()
        if len(peaks_cellranger) > 0:
            peaks_cellranger["peak"] = peaks_cellranger["chr"] + ":" + peaks_cellranger["start"].astype(str) + "-" + peaks_cellranger["end"].astype(str)
            peaks_cellranger["method"] = peaks_name
            peaks_cellranger = center_peaks(peaks_cellranger, promoter)
            peaks.append(peaks_cellranger.set_index("peak"))

    peaks = pd.concat(peaks).reset_index()
    peaks["method"] = pd.Categorical(peaks["method"], peak_methods.index)
    peaks["size"] = peaks["end"] - peaks["start"]
    
    return peaks.set_index(["method", "peak"])


# %%
import chromatinhd.grid
main = wrap = chd.grid.Wrap(5, padding_width = 0.1, padding_height = 0.3)
fig = chd.grid.Figure(main)

padding_height = 0.001
resolution = 0.0005

panel_height = 0.5

total_width_cutoff = 10

for slice_ix, slice_oi in enumerate(slices_oi.to_dict(orient="records")):
    grid = chd.grid.Grid(margin_height=0, margin_width=0, padding_height=0, padding_width=0)
    main.add(grid)
    
    slice_oi = dict(slice_oi)

    expanded_slice_oi = slice_oi.copy()
    expanded_slice_oi["start"] = np.clip(slice_oi["start"] - 1000, *(window - window[0]))
    expanded_slice_oi["end"] = np.clip(slice_oi["end"] + 1000, *(window - window[0]))

    window_oi = np.array([expanded_slice_oi["start"], expanded_slice_oi["end"]]) + window[0]

    gene_oi = expanded_slice_oi["gene_ix"]
    cluster_ix = expanded_slice_oi["cluster_ix"]
    cluster_info_oi = cluster_info.iloc[[cluster_ix]]

    plotdata_atac = design.query("gene_ix == @gene_oi").copy().rename(columns = {"active_latent":"cluster"}).set_index(["coord", "cluster"]).drop(columns = ["batch", "gene_ix"])
    plotdata_atac["prob"] = probs[gene_oi].flatten()
    plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

    plotdata_atac["prob"] = plotdata_atac["prob"] - np.log(plotdata_atac.reset_index().groupby(["cluster"]).apply(lambda x:np.trapz(np.exp(x["prob"]), x["coord"].astype(float) / (window[1] - window[0])))).mean()
    plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

    resolution = 0.0005
    panel_width = (window_oi[1] - window_oi[0]) * resolution

    # differential atac
    wrap_differential = chd.differential.plot.Differential(
        plotdata_atac,
        plotdata_atac_mean,
        cluster_info_oi,
        window_oi,
        panel_width,
        panel_height,
        padding_height = padding_height,
        title = False
    )
    grid[0, 0] = wrap_differential

    ax = wrap_differential.elements[0].ax
        
    # add labels
    gene_label = transcriptome.var.iloc[gene_oi]["symbol"]
    cluster_label = cluster_info.query("dimension == @cluster_ix")["label"][0]
    position_label = str(int(slice_oi["start"] + slice_oi["length"]/2) + window[0])
    
    chd.differential.plot.LabelSlice(ax, gene_label, cluster_label, slice_oi, window)
    
    # peaks
    promoter = promoters.iloc[expanded_slice_oi["gene_ix"]]
    peaks = get_promoter_peaks(promoter, peak_methods)
    
    grid[1, 0] = chd.differential.plot.Peaks(peaks, peak_methods, window_oi, panel_width, label_methods = slice_ix == 0, panel_height = 0.5)
    
    # add common/unique annotation
    region_position_chosen_oi = region_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]
    peak_position_chosen_oi = peak_position_chosen[
        gene_oi, cluster_ix, expanded_slice_oi["start"] : expanded_slice_oi["end"]
    ]
    
    chd.differential.plot.CommonUnique(ax, peak_position_chosen_oi, region_position_chosen_oi, expanded_slice_oi, window, method_info)
fig.plot()

# %% [markdown]
# ## Characteristics

# %% [markdown]
# ### Size

# %%
logbins = np.logspace(np.log10(1),np.log10(10000),50)
sns.histplot(peakresult.get_slicescores()["length"], bins = logbins, color = method_info["color"]["peak"])
sns.histplot(pureregionresult.get_slicescores()["length"], bins = logbins, color = method_info["color"]["region"])
plt.xscale('log')

# %%
fig, ax = plt.subplots(figsize = (2, 2))
sns.ecdfplot(peakresult.get_slicescores()["length"], ax = ax, color = method_info["color"]["peak"])
sns.ecdfplot(regionresult.get_slicescores()["length"], ax = ax, color = method_info["color"]["region"])
sns.ecdfplot(pureregionresult.get_slicescores()["length"], ax = ax, color = method_info["color"]["region"])
ax.set_xscale("log")
ax.set_xticks([10, 100, 1000, 10000])
ax.set_xlim(10, 10000)

# %%
pureregionresult.get_slicescores()["length"].median(), pureregionresult.get_slicescores()["length"].mean()

# %%
pureregionresult.get_slicescores()["length"].quantile([0.25, 0.75])

# %%
mid_bins = np.linspace(*window, 11)[:-1]

# %%
slicescores_region = pureregionresult.get_slicescores()
slicescores_region["mid_bin"] = np.digitize(slicescores_region["mid"], mid_bins)

# %%
logbins = np.logspace(np.log10(20),np.log10(1000),50)

# %%
fig, axes = plt.subplots(len(mid_bins), 1, figsize = (5, len(mid_bins)/3), sharex = True, sharey = True, gridspec_kw={"hspace":0})
for i, (mid_bin, plotdata) in enumerate(slicescores_region.groupby("mid_bin")):
    ax = axes[i]
    ax.hist(plotdata["length"], density = True, bins = logbins, lw = 0)
    ax.axvline(plotdata["length"].median(), color = "red")
    ax.set_xscale("log")

# %%
fig, ax = plt.subplots(figsize = (3, 3))
sns.ecdfplot(data = slicescores_region, y = "length", hue = "mid_bin")
ax.set_yscale("log")
ax.set_ylim(1, 10000)
ax.legend([])

# %%
slicescores_region["loglength"] = np.log1p(slicescores_region["length"])

# %%
midbinscores_region = pd.DataFrame({
    "position":mid_bins,
    "length":np.exp(slicescores_region.groupby("mid_bin")["loglength"].mean()),
    "length_std":np.exp(slicescores_region.groupby("mid_bin")["loglength"].std())
}).set_index("position")

# %%
fig, ax = plt.subplots(figsize = (2, 2))
midbinscores_region["length"].plot(ax = ax)
fig, ax = plt.subplots(figsize = (2, 2))
midbinscores_region["length_std"].plot(ax = ax)

# %%
sns.histplot(plotdata["length"], binrange = (0, 500), bins = 50)#.plot(kind = "hist", bins = 100)

# %% [markdown]
# ### Height

# %%
peakaverages = peakresult.get_sliceaverages(probs_interpolated).join(peakresult.get_slicescores().reset_index())
regionaverages = regionresult.get_sliceaverages(probs_interpolated).join(regionresult.get_slicescores().reset_index())
pureregionaverages = pureregionresult.get_sliceaverages(probs_interpolated).join(pureregionresult.get_slicescores().reset_index())

# %%
fig, axes = plt.subplots(1, 2, figsize = (4, 2), sharey = True)

ax = axes[0]
sns.ecdfplot(np.exp(peakaverages["average"]), ax = ax, color = method_info.loc["peak", "color"])
sns.ecdfplot(np.exp(regionaverages["average"]), ax = ax, color = method_info.loc["region", "color"])
# sns.ecdfplot(np.exp(pureregionaverages["average"]), ax = ax, color = method_info.loc["region", "color"], dashes = (2, 2))
ax.set_xlabel("Average accessibility")
ax.set_xscale("log")

ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))

ax = axes[1]
sns.ecdfplot(np.exp(peakresult.get_sliceaverages(probs_interpolated)["max"]), ax = ax, color = method_info.loc["peak", "color"])
sns.ecdfplot(np.exp(regionresult.get_sliceaverages(probs_interpolated)["max"]), ax = ax, color = method_info.loc["region", "color"])
# sns.ecdfplot(np.exp(pureregionresult.get_sliceaverages(probs_interpolated)["max"]), ax = ax, color = method_info.loc["region", "color"], dashes = (2, 2))
ax.set_xscale("log")
ax.set_xlabel("Maximum accessibility")

# %%
upstream_cutoff = -1000 - window[0]
downstream_cutoff = +1000 - window[0]

def classify_position_group(x, upstream_cutoff, downstream_cutoff):
    return np.array(["upstream", "downstream", "promoter"])[np.argmax(np.vstack([x.end < upstream_cutoff, x.start > downstream_cutoff, np.repeat(True, len(x))]), 0)]

position_group_info = pd.DataFrame([
    ["upstream", "Upstream -10kb→-1kb"],
    ["promoter", "-1kb→+1kb"],
    ["downstream", "Downstream +1kb→+10kb"],
], columns = ["position_group", "label"]).set_index("position_group")

# %%
peakaverages["position_group"] = classify_position_group(peakaverages, upstream_cutoff, downstream_cutoff)
regionaverages["position_group"] = classify_position_group(regionaverages, upstream_cutoff, downstream_cutoff)

# %%
import textwrap

# %%
fig, axes = plt.subplots(1, 3, figsize = (1.5 * 3, 1.5), sharey = True, sharex = True)

score = "max";xlim = (0.1, 100); xlabel = "Maximum accessibility"
# score = "average";xlim = (0.1, 100); xlabel = "Average accessibility"
# score = "average_lfc";xlim = (1., 10); xlabel = "Average fold-change"
# score = "max_lfc";xlim = (1., 10); xlabel = "Max fold-change"
plotdata_all = {
    "peak":np.exp(peakaverages[score]),
    "region":np.exp(regionaverages[score])
}
for ax, position_group in zip(axes, position_group_info.index):
    plotdata = {
        "peak":np.exp(peakaverages.loc[peakaverages["position_group"] == position_group][score]),
        "region":np.exp(regionaverages.loc[regionaverages["position_group"] == position_group][score])
    }
    plotdata_mean = {method:np.exp(np.log(values).median()) for method, values in plotdata.items()}
    print(plotdata_mean)
    sns.ecdfplot(plotdata["peak"], ax = ax, color = method_info.loc["peak", "color"])
    sns.ecdfplot(plotdata["region"], ax = ax, color = method_info.loc["region", "color"])
    
    sns.ecdfplot(plotdata_all["peak"], ax = ax,color = method_info.loc["peak", "color"], zorder = -1, alpha = 0.1)
    sns.ecdfplot(plotdata_all["region"], ax = ax, color = method_info.loc["region", "color"], zorder = -1, alpha = 0.1)
    
    ax.set_xscale("log")
    ax.set_xlabel("")
    ax.set_xlim(*xlim)
    ax.set_title("\n".join(textwrap.wrap(position_group_info.loc[position_group]["label"], width = 10)))
    ax.annotate(
        None,
        (plotdata_mean["peak"], 0.5),
        (plotdata_mean["region"], 0.5),
        xycoords = "data",
        textcoords = "data",
        ha = "center",
        va = "center",
        arrowprops = dict(arrowstyle = "->", ec = "black", shrinkA=0, shrinkB = 0)
    )
    ax.annotate(
        f'$\\times${plotdata_mean["peak"] / plotdata_mean["region"]:.2f}',
        (max(plotdata_mean["peak"], plotdata_mean["region"]), 0.5), va = "center", ha = "left"
    )
ax = axes[0]
ax.set_xlabel(xlabel)
ax.set_yticks([0, 0.5, 1])
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

# %% [markdown]
# ## Gene-wise overlap

# %%
plotdata = pd.DataFrame({
    'n_region':regionresult.get_slicescores().groupby("gene_ix")["length"].sum(),
    'n_peak':peakresult.get_slicescores().groupby("gene_ix")["length"].sum()
}).fillna(0)
plotdata = plotdata.reindex(np.arange(transcriptome.var.shape[0]), fill_value = 0)
plotdata.index = transcriptome.var.index[plotdata.index]

# %%
fig, ax = plt.subplots()
ax.scatter(
    plotdata["n_region"],
    plotdata["n_peak"]
)
ax.set_xscale("symlog", linthresh=100)
ax.set_yscale("symlog", linthresh=100)

# %%
plotdata["diff"] = plotdata["n_region"] - plotdata["n_peak"]

# %%
plotdata.sort_values("diff").assign(symbol = lambda x:transcriptome.symbol(x.index).values)

# %% [markdown]
# ## Classification

# %%
import chromatinhd.slicetypes
chromatinhd.slicetypes.types_info

# %% [markdown]
# ### Classify

# %%
slicetopologies = pd.concat([
    pureregionresult.get_slicetopologies(probs_interpolated),
    pureregionresult.get_sliceaverages(probs_interpolated),
    pureregionresult.get_slicescores()
], axis = 1)

# %%
promoters

# %%
slicetopologies["flank"] = slicetopologies["prominence"] <= 0.5
slicetopologies["hill"] = slicetopologies["dominance"] <= 0.5
slicetopologies["chain"] = (slicetopologies["length"] > 800) & (slicetopologies["n_subpeaks"] >= 2)
slicetopologies["canyon"] = (slicetopologies["balance"] >= 0.2) | (slicetopologies["balances_raw"] < np.log(2))
slicetopologies["ridge"] = (slicetopologies["length"] > 800) & (slicetopologies["shadow"] > 0.5)
slicetopologies["volcano"] = (slicetopologies["max"] < np.log(1.0))

# %%
slicetopologies["type"] = "peak"
slicetopologies.loc[slicetopologies["volcano"], "type"] = "volcano"
slicetopologies.loc[slicetopologies["hill"], "type"] = "hill"
slicetopologies.loc[slicetopologies["canyon"], "type"] = "canyon"
slicetopologies.loc[slicetopologies["flank"], "type"] = "flank"
slicetopologies.loc[slicetopologies["chain"], "type"] = "chain"
slicetopologies.loc[slicetopologies["ridge"], "type"] = "ridge"
slicetopologies["type"] = pd.Categorical(slicetopologies["type"], categories = chd.slicetypes.types_info.index)

# %%
slicetopologies["loglength"] = np.log(slicetopologies["length"])

# %% [markdown]
# ### Store for David

# %%
slicetopologies_mapped = slicetopologies.copy()
slicetopologies_mapped["gene"] = promoters.index[slicetopologies_mapped["gene_ix"]]
slicetopologies_mapped["cluster"] = cluster_info.index[slicetopologies_mapped["cluster_ix"]]

# %%
slicetopologies_mapped["start"] = (
    promoters.loc[slicetopologies_mapped.gene, "tss"].values + 
    (slicetopologies["start"] + window[0]) * (promoters.loc[slicetopologies_mapped.gene, "strand"] == 1).values - 
    (slicetopologies["end"] + window[0]) * (promoters.loc[slicetopologies_mapped.gene, "strand"] == -1).values
)

slicetopologies_mapped["end"] = (
    promoters.loc[slicetopologies_mapped.gene, "tss"].values + 
    (slicetopologies["end"] + window[0]) * (promoters.loc[slicetopologies_mapped.gene, "strand"] == 1).values - 
    (slicetopologies["start"] + window[0]) * (promoters.loc[slicetopologies_mapped.gene, "strand"] == -1).values
)
slicetopologies_mapped["chr"] = promoters.loc[slicetopologies_mapped.gene, "chr"].values

# %%
scores_dir = prediction.path / "scoring" / "significant_up"
slicetopologies_mapped.to_csv(scores_dir / "slicetopologies.csv")

# %%
slicetopologies_mapped[["chr", "start", "end", "cluster", "type"]]

# %%
from_file = scores_dir / "slicetopologies.csv"
to_output = pathlib.Path("/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output")
to_file = to_output / from_file.relative_to(chd.get_output())

to_file.parent.mkdir(parents = True, exist_ok = True)

import shutil
shutil.copy(from_file, to_file)

# %% [markdown]
# ### Positions

# %%
fig, axes = plt.subplots(
    chd.slicetypes.types_info.shape[0],
    1,
    figsize = (5, chd.slicetypes.types_info.shape[0] * 0.5),
    sharex = True,
    gridspec_kw = {"hspace":0}
)
nbins = 100
density_lim = 1/((window[1] - window[0])/nbins) / 25
for ax, (type, plotdata) in zip(axes, slicetopologies.groupby("type")):
    color = chd.slicetypes.types_info.loc[type, "color"]
    sns.histplot(plotdata["mid"], bins = nbins, stat = "density", label = type, lw = 0, ax = ax, color = color)
    # ax.text(0.02, 0.95, type, color = color, transform = ax.transAxes, va = "top", ha = "left")
    ax.set_yticks([])
    ax.set_xlim(*window)
    
    w, h = fig.transFigure.inverted().transform([[1, 1]])[0] * 20
    x, y = fig.transFigure.inverted().transform(ax.transAxes.transform([0.1, 0.9]))
    y -= h
    
    l = ax.yaxis.get_label()
    inset = chd.plotting.replace_patch(ax, l, points = 25, ha = "right")
    l.set_visible(False)
    inset.axis("off")
    chd.slicetypes.plot_type(inset, type)
    ax.set_ylim(0,density_lim)
    ax.axvline(0, dashes = (2, 2), lw = 1, color = "#333")
axes[-1].set_xlabel("    ← upstream    Variant    downstream →")

# %% [markdown]
# ### Frequencies

# %%
fig, ax = plt.subplots(figsize = (1.5, 3))

plotdata = pd.DataFrame({
    "n_regions":slicetopologies.groupby("type").size(),
    "n_positions":slicetopologies.groupby("type")["length"].sum()
})
plotdata["rel_n_regions"] = plotdata["n_regions"] / plotdata["n_regions"].sum()
plotdata["cum_n_regions"] = np.cumsum(plotdata["rel_n_regions"]) - plotdata["rel_n_regions"]
plotdata["rel_n_positions"] = plotdata["n_positions"] / plotdata["n_positions"].sum()
plotdata["cum_n_positions"] = np.cumsum(plotdata["rel_n_positions"]) - plotdata["rel_n_positions"]

ax.bar(
    0,
    plotdata["rel_n_regions"],
    bottom =plotdata["cum_n_regions"],
    color = chd.slicetypes.types_info.loc[plotdata.index, "color"],
    lw = 0
)
ax.bar(
    1,
    plotdata["rel_n_positions"],
    bottom =plotdata["cum_n_positions"],
    color = chd.slicetypes.types_info.loc[plotdata.index, "color"],
    lw = 0
)

texts = []
for type, plotdata_type in plotdata.iterrows():
    color = chd.slicetypes.types_info.loc[type, "color"]
    text = ax.text(-0, plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"]/2, f"{plotdata_type['rel_n_regions']:.1%}", ha = "center", va = "center", color = "white", fontweight = "bold")
    text.set_path_effects([mpl.patheffects.Stroke(linewidth=3, foreground = color), mpl.patheffects.Normal()])
    # texts.append(text)

    text = ax.text(1., plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"]/2, f"{plotdata_type['rel_n_positions']:.1%}", ha = "center", va = "center", color = "white", fontweight = "bold")
    text.set_path_effects([mpl.patheffects.Stroke(linewidth=3, foreground = color), mpl.patheffects.Normal()])
    # texts.append(text)
    # texts_left.append(ax.text(-0.5, plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"]/2, f"{plotdata_type['rel_n_regions']:.1%}", ha = "right"))
    # texts_right.append(ax.text(2.5, plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"]/2, f"{plotdata_type['rel_n_positions']:.1%}", ha = "left"))
    # texts.append(ax.text(0, plotdata_type["cum_n_regions"] + plotdata_type["rel_n_regions"]/2, f"{plotdata_type['rel_n_regions']:.1%} {type}", ha = "center", va = "center"))
    # ax.text(1, plotdata_type["cum_n_positions"] + plotdata_type["rel_n_positions"]/2, f"{plotdata_type['rel_n_positions']:.1%} {type}", ha = "center", va = "center")

ax.set_xticks([0, 1])
ax.set_xticklabels(["regions", "positions"])
ax.set_ylim(1, 0)
ax.set_xlim(-0.4, 1.4)
ax.set_yticks([0, 0.5, 1])
sns.despine(ax = ax)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.xaxis.tick_top()
# adjustText.adjust_text(texts, autoalign=False, only_move = {"text":"y"}, ha = "center", lim = 3000)
# adjustText.adjust_text(texts_right, autoalign=False, only_move = {"text":"y"}, ha = "left")

# %%
scores.groupby("variant")["significant"].any()

# %%
promoters

# %%
plotdata = pd.DataFrame({
    "n_regions":slicetopologies.groupby(["type", "cluster_ix"]).size(),
    "n_positions":slicetopologies.groupby(["type", "cluster_ix"])["length"].sum()
})

# %% [markdown]
# ### Enrichment with cell-type specific

# %%
import scipy

# %%
slicetopologies["gene"] = promoters.iloc[slicetopologies["gene_ix"]].index

# %%
scores = pickle.load((chd.get_git_root() / "code" / "6-eqtl" / "scores.pkl").open("rb"))

# %%
scores["significant"] = scores["bf"] > np.log(10)

# %%
cluster_oi = "B"
cluster_ix = cluster_info.loc[cluster_oi, "dimension"]

# %%
significant = scores["significant"].unstack().groupby("variant").any()
slicetypes = slicetopologies.groupby(["gene", "type"]).size().unstack().reindex(significant.index, fill_value = 0.) > 1

# %%
contingencies = [
    (~significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
    (~significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
    (significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
    (significant[cluster_oi].values[:, None] & slicetypes.values).sum(0)
]
contingencies = np.stack(contingencies)

# %%
import fisher

# %%
slicetype_enrichments = []
for cluster_oi in cluster_info.index:
    cluster_ix = cluster_info.loc[cluster_oi, "dimension"]
    slicetopologies_oi = slicetopologies.query("cluster_ix == @cluster_ix")
    slicetopologies_oi = slicetopologies_oi.query("(start <= 5000) & (end >= -5000)")
    significant = scores["significant"].unstack().groupby("variant").any()
    slicetypes = slicetopologies_oi.groupby(["gene", "type"]).size().unstack().reindex(significant.index, fill_value = 0.) > 1
    contingencies = [
        (~significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
        (~significant[cluster_oi].values[:, None] & slicetypes.values).sum(0),
        (significant[cluster_oi].values[:, None] & ~slicetypes.values).sum(0),
        (significant[cluster_oi].values[:, None] & slicetypes.values).sum(0)
    ]
    contingencies = np.stack(contingencies)
    for slicetype, cont in zip(slicetypes.columns, contingencies.T):
        slicetype_enrichments.append({
            "cont":cont,
            # "odds":(cont[0] * cont[3])/(cont[1] * cont[2]),
            "odds":scipy.stats.contingency.odds_ratio(cont.reshape((2, 2))).statistic,
            "p":fisher.pvalue(*cont).right_tail,
            "cluster":cluster_oi,
            "type":slicetype
        })

# %%
slicetype_enrichments = pd.DataFrame(slicetype_enrichments)

# %%
overall_slicetype_enrichments = []
for type, slicetype_enrichments_type in slicetype_enrichments.groupby("type"):
    cont = np.vstack(slicetype_enrichments_type["cont"]).sum(0)
    p = fisher.pvalue(*cont).right_tail
    odds = scipy.stats.contingency.odds_ratio(cont.reshape(2, 2)).statistic
    print(type, odds)
    
    overall_slicetype_enrichments.append({"p":p, "odds":odds, "type":type})

# %%
overall_slicetype_enrichments = pd.DataFrame(overall_slicetype_enrichments)

# %%
overall_slicetype_enrichments["logodds"] = np.log(overall_slicetype_enrichments["odds"])

# %%
fig, ax = plt.subplots(figsize = (2, 2))
plotdata = overall_slicetype_enrichments.groupby("type")[["logodds"]].mean()
ax.barh(np.arange(len(plotdata)), plotdata["logodds"], color = chd.slicetypes.types_info.loc[plotdata.index]["color"])
# ax.set_xscale("log")
ax.set_xticks(np.log([1, 1.5, 2]))
ax.set_xticklabels(["1", "1.5", "2"])
ax.set_xticks([], minor = True)
ax.set_yticks(np.arange(len(plotdata)))
ax.set_yticklabels(plotdata.index)
ax.set_xlabel("Fold enrichment\n near functional mutations")
chd.slicetypes.label_axis(ax, ax.yaxis)

# %%
