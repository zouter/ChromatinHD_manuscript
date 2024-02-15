# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python-3.9.5
#     language: python
#     name: python3
# ---

# %%
folder_to_chromatinhd_manuscript = "/home/pavel/Deplancke-lab/immersion/ChromatinHD_manuscript/" # add here the folder to ChromatinHD_manuscript, make sure it ends with /
# !ln -s /data/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output {folder_to_chromatinhd_manuscript}

# if this worked correctly, this should contain several subfolders
# !ls {folder_to_chromatinhd_manuscript}/output

# %%
# install chromatinhd package (in developer mode)
# you will first have to run this to install our package
# after running this, you will likely have to restart the notebook
folder_to_chromatinhd = "/home/pavel/Deplancke-lab/immersion/ChromatinHD/" # add here the folder to ChromatinHD, make sure it ends with /
# !pip install -e {folder_to_chromatinhd}

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

from statannotations.Annotator import Annotator

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

import chromatinhd as chd

# %%
FULL_NB = False

# %% [markdown]
# ## Information on histone marks

# %% [markdown]
# Based on research (and a bit of help from ChatGPT). Feel free at add your own information:
#
# - H3K4me3 - This modification is typically associated with active gene transcription. It is often found at the promoter regions of genes, where it can help recruit transcription factors and other proteins that initiate transcription.
# - H3K27ac - This modification is also associated with active gene transcription. It is often found near enhancer regions, which are regulatory regions that can increase gene expression.
# - H3K27me3 - This modification is associated with gene repression. It is often found at regions of chromatin that are tightly packed and inaccessible to transcriptional machinery.
# - H3K4me1 - This modification is associated with enhancer regions and is thought to help stabilize their structure.
# - H3K9me3 - This modification is also associated with gene repression. It is often found at regions of chromatin that are densely packed and transcriptionally inactive.
#
#  - H3K36me3 - This modification is associated with gene expression and is often found at the bodies of actively transcribed genes. It may help recruit proteins that are involved in elongation of the transcript during transcription. It is often correlated with high methylation in gene bodies (https://www.embopress.org/doi/full/10.15252/embj.201796812) and because it promotes DNA methylation, it is likely involved in avoiding aberrant transcription from intragenic transcription start sites.
#
# It's important to note that the functions of these modifications can vary depending on the specific cell type and context in which they are found.

# %% [markdown]
# ## Load different ChiP-seq files

# %% [markdown]
# In the `bw_folder`, I have downloaded a bunch of bigwig files from ENCODE, that measured histone marks using ChIP-seq from different blood cell types. You can find the overview of all data (which is only a subset of what we downloaded) here: https://www.encodeproject.org/immune-cells/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.cell_slims=hematopoietic+cell&biosample_ontology.classification=primary+cell&control_type!=*&status!=replaced&status!=revoked&status!=archived&biosample_ontology.system_slims=immune+system&biosample_ontology.system_slims=circulatory+system&config=immune

# %%
encode_folder = chd.get_output() / "data" / "encode2"
bw_folder = encode_folder / "immune"

# %% [markdown]
# This following dataframe links all the identifiers form ENCODE, with the celltype/cluster identifiers we use in our ATAC-seq data. E.g. CD4 T = "naive thymus-derived CD4-positive, alpha-beta T cell" in ENCODE data.

# %%
biosamples_oi = pd.DataFrame([
    ["naive thymus-derived CD4-positive, alpha-beta T cell", "CD4 T"],
    ["naive thymus-derived CD8-positive, alpha-beta T cell", "CD8 T"],
    ["CD14-positive monocyte", "Monocytes"],
    ["naive B cell", "B"],
    ["B cell", "B"],
    ["immature natural killer cell", "NK"],
    ["natural killer cell", "NK"],
    ["dendritic cell", "cDCs"]
], columns = ["Biosample term name", "cluster"])

# %% [markdown]
# The following dataframe will contain information on the clusters, e.g. for now only what color they will have for plotting.

# %%
clusters_info = pd.DataFrame({"cluster":biosamples_oi["cluster"].unique()}).set_index("cluster")
clusters_info["color"] = sns.color_palette(n_colors = len(clusters_info))

# %% [markdown]
# The following file contains the information for all the files that we downloaded from ENCODE

# %%
files = pd.read_csv(bw_folder / "files.csv", index_col = 0)
files["target"] = files["Experiment target"].str.split("-").str[0]

# DNA methylation will have as "Experiment target" = NA, but we change it here to DNAme for consistency downstream
files.loc[pd.isnull(files["target"]), "target"] = "DNAme"
files.head()

# %% [markdown]
# We got a bunch of marks:

# %%
files["target"].value_counts()

# %% [markdown]
# And a bunch of cell types:

# %%
files["Biosample term name"].value_counts()

# %% [markdown]
# And what output they contain. For ChIP-seq it is signal p-value, while for methylation it is the raw signal.

# %%
files["Output type"].value_counts()

# %% [markdown]
# ## Load promoter positions

# %%
dataset_name = "pbmc10k"

folder_dataset = chd.get_output() / "data" / dataset_name

promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_dataset / ("promoters_" + promoter_name + ".csv"), index_col = 0)

transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")

# %% [markdown]
# ### Load at a single file

# %% [markdown]
# Let's visualize the ChIP-seq data for one sample close to one gene.

# %%
import pyBigWig

# %%
idx_oi = files.query("`Biosample term name` == 'natural killer cell'").query("`target` == 'H3K4me3'").index[0]
idx_oi = files.query("`Biosample term name` == 'natural killer cell'").query("`target` == 'DNAme'").index[0]

# %%
bw = pyBigWig.open(str(bw_folder / files.iloc[idx_oi]["filename"]))

# %%
promoter = promoters.loc[transcriptome.gene_id("CCL4")]

# %%
plotdata = pd.DataFrame({
    "signal":np.array(bw.values(promoter.chr, promoter.start, promoter.end)),
    "position":np.arange(window[0], window[1])[::promoter["strand"]],
    "chrposition":np.arange(promoter.start, promoter.end)
})

# %%
fig, ax = plt.subplots(figsize = (15, 2))
ax.plot(plotdata["position"], plotdata["signal"])
ax.set_xlim(plotdata["position"].min(), plotdata["position"].max())
ax2 = ax.twiny()
ax2.set_xlim([plotdata["chrposition"].max(), plotdata["chrposition"].min()][::promoter["strand"]])

# %% [markdown]
# 0 represents the transcription start site, while positive numbers represent (part of) the gene body, negative numbers is everything upstream from the promoter. I also add a second x-axis that gives you the real coordinates on the chromosome (according to the Grch38 assembly).

# %% [markdown]
# ### Explore multiple tracks

# %% [markdown]
# Let's visualize the ChIP-seq data for a couple of samples from different cell types

# %%
display(files)

# %%
files_oi = pd.merge(files, biosamples_oi)

clusters_oi = clusters_info.loc[['NK', 'Monocytes', 'CD4 T', 'B', 'CD8 T']]
files_oi = files_oi.query("cluster in @clusters_oi.index")

targets = ['H3K36me3']
files_oi = files_oi.query("target in @targets")

files_oi = files_oi.copy()
files_oi["file"] = [pyBigWig.open(str(bw_folder / file["filename"])) for _, file in files_oi.iterrows()]
files_oi.index.name = "file_ix"

files_oi = files_oi.sort_values("cluster")

# %%
display(files_oi)

# %%
signals = {}
for file_ix, file in files_oi.iterrows():
    bw = file["file"]
    signal = np.array(bw.values(promoter.chr, promoter.start, promoter.end))
    signal_file = pd.Series(np.log(signal+0.01))
    signals[file_ix] = signal_file
signals = pd.concat(signals, axis = 1).T
signals.columns = np.arange(promoter.start, promoter.end)

# %%
fig, ax = plt.subplots(figsize = (10, 3))
plotdata = signals
ax.matshow(plotdata, aspect = 100)

# %%
signals_tmp = signals.copy()
signals_tmp.index = files_oi.loc[signals_tmp.index, "cluster"]
signals_clusters = signals_tmp.groupby(level = 0).mean()

# %%
fig, ax = plt.subplots(figsize = (15, 2))
for cluster in clusters_oi.index:
    ax.plot(signals_clusters.columns, np.exp(signals_clusters.loc[cluster]), label = cluster)
ax.legend()

# %%
sns.heatmap(signals_clusters)

# %% [markdown]
# ## Combine with different types differential ATAC-seq regions

# %% [markdown]
# Let's visualize the signal for a particular gene (CCL4) across different cell types and markers (`targets`)

# %%
files_oi = pd.merge(files, biosamples_oi)

clusters_oi = clusters_info.loc[['NK', 'Monocytes', 'CD4 T', 'B', 'CD8 T']]
files_oi = files_oi.query("cluster in @clusters_oi.index")

targets = [
    "DNAme",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me3",
    "H3K9me3",
]
files_oi = files_oi.query("target in @targets")

files_oi = files_oi.copy()
files_oi["file"] = [pyBigWig.open(str(bw_folder / file["filename"])) for _, file in files_oi.iterrows()]
files_oi.index.name = "file_ix"

files_oi = files_oi.sort_values("cluster")

# %%
promoter = promoters.loc[transcriptome.gene_id("CCL4")]

# %% [markdown]
# Load in the "slicetopologies".
# This contains all the regions that were differentially accessible in a particular cell type (=`cluster`), and how these slices looked like (=`type`). It also contains where these regions are located (`chr`, `start`, `end`). The goal is now to compare how these different shapes of differentially accessibly regions are related to chromatin marks.

# %%
prediction_path = chd.get_output() / "prediction_likelihood/pbmc10k/10k10k/leiden_0.1/v9_128-64-32"
scores_dir = prediction_path / "scoring" / "significant_up"
scores_file = scores_dir / "slicetopologies.csv"
slicetopologies = pd.read_csv(scores_file, index_col = 0)

# %%
analysis_folder = prediction_path / "encode_marks"
analysis_folder.mkdir(exist_ok = True, parents = True)


# %% [markdown]
# Some functions that extract all the ChIP-seq data around a gene, and that will aggregate this across different ChIP-seq samples.

# %%
def get_signals(files_oi, promoter):
    signals = {}
    for file_ix, file in files_oi.iterrows():
        bw = file["file"]
        signal = np.array(bw.values(promoter.chr, promoter.start, promoter.end))
        signal_file = pd.Series(signal)
        signals[file_ix] = signal_file
    signals = pd.concat(signals, axis = 1).T
    signals.columns = np.arange(promoter.start, promoter.end)
    return signals
signals = get_signals(files_oi, promoter)


# %%
def get_signals_clusters(signals):
    signals_tmp = signals.copy()
    signals_tmp.index = pd.MultiIndex.from_frame(files_oi.loc[signals_tmp.index, ["cluster", "target"]])
    signals_clusters = signals_tmp.groupby(level = ["cluster", "target"]).mean()
    return signals_clusters
signals_clusters = get_signals_clusters(signals)
normsignal_clusters = (
    signals_clusters / 
    signals_clusters.groupby("target").max().max(1)[signals_clusters.index.get_level_values("target")].values[:, None]
)

# %%
target_info = pd.DataFrame({"target":targets}).set_index("target")
target_info["color"] = sns.color_palette(n_colors = len(target_info))
target_info["label"] = target_info.index

# %% [markdown]
# Plot for each cell type and for each mark its average signal.
# Also plot on top the different types of differential accessibility regions.
# See powerpoint for color code

# %%
slicetopologies_oi = slicetopologies.query("gene == @promoter.name")
main = chd.grid.Grid(padding_height = 0)
fig = chd.grid.Figure(main)

ymax = 1

for cluster_ix, cluster in enumerate(clusters_oi.index):
    panel = main[cluster_ix, 0] = chd.grid.Ax(dim = (10, 0.5))
    ax = panel.ax
    
    normsignal_cluster = normsignal_clusters.loc[cluster]
    
    for target, normsignal_target in normsignal_cluster.iterrows():
        ax.plot(
            normsignal_target.index,
            normsignal_target,
            color = target_info.loc[target, "color"],
        )
    
    for _, slice in slicetopologies_oi.query("cluster == @cluster").iterrows():
        ax.axvspan(slice["start"], slice["end"], fc = chd.slicetypes.types_info.loc[slice["type"], "color"], alpha = 0.5)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(cluster, rotation = 0, ha = "right", va = "center")
    ax.axvline(promoter["tss"], dashes = (2, 2), color = "grey")
    
    # invert the xaxis if the gene would go in the other direction
    # this makes it so that the gene body is also to the right of the plot
    if promoter.strand == -1:
        ax.xaxis.invert_xaxis()
    
fig.plot()


# %% [markdown]
# Color code for all the tracks

# %%
# create a standalone legend
def create_color_legend(data):
    fig_fake, ax_fake = plt.subplots()
    lines = []
    for _, data_ in data.iterrows():
        lines.append(ax_fake.plot([0, 1], [0, 1], color = data_["color"])[0])
    plt.close()

    fig, ax = plt.subplots(figsize=(3,2))
    ax.axis("off")
    fig.legend(lines, data["label"], bbox_to_anchor=(0.5, 1.), loc = "upper center")
    fig.tight_layout()
    fig.show()
    return fig
create_color_legend(target_info);

# %% [markdown]
# ## Absolute signal within each region

# %% [markdown]
# We want to answer the question: how are different shapes (or "topologies") of differential DNA accessibility associated with different histone marks?
#
# There are multiple ways to answer this question.
#
# We're first going to do it in the most simplistic way, and perhaps dig deeper later.
#
# The simplest way would be to take all the regions that are differential in a particular cell type, and look at the average signal for a chromatin mark in the particular region in the samples from the particular cell type. In other words, we will average both across a whole region and across all samples.

# %% [markdown]
# For speed, we're at first only doing it for the first 10000 slices/regions. To do the actual analysis, we will of course look at all regions in the future, but you can just look at these regions for now to do some initial exploration. Also, because this takes some time I'm first extracting the information from the bigwig files, saving it, and reading it back in to do the plotting, normalization and analysis later. That way you can play around with the interpretation/plotting without having to rerun this code block all the time.

# %%
slicetopologies_oi = slicetopologies.query("cluster in @clusters_oi.index").head(10000) # select the first 10k regions
slicetopologies_oi.index.name = "slice_ix"

# %%
if FULL_NB:
    
    signals_slices = {}
    # for each slice
    for slice_ix, slicetopology in tqdm.tqdm(slicetopologies_oi.iterrows(), total = len(slicetopologies_oi)):
        signals = []
        file_ixs = []

        # select  for only the celltypes in which the slice was differential
        for file_ix, file in files_oi.loc[files_oi["cluster"] == slicetopology["cluster"]]["file"].items():
            signals.append(np.array(file.values(slicetopology.chr, slicetopology.start, slicetopology.end)))
            file_ixs.append(file_ix)

        # stack the signal from the different files (i.e. samples) and then calculate the mean for each target (=mark)
        signals = pd.DataFrame(np.stack(signals), index = file_ixs)
        signals.index = files_oi.loc[file_ixs, "target"]
        signals_slices[slice_ix] = signals

# %%
print(len(slicetopologies.query("cluster in @clusters_oi.index")))

# %%
if FULL_NB:
    pickle.dump(signals_slices, (analysis_folder / "signals_slices.pkl").open("wb"))

# %% [markdown]
# Load in the data again

# %%
signals_slices = pickle.load((analysis_folder / "signals_slices.pkl").open("rb"))

# %% [markdown]
# Take the mean for each 

# %%
signal_topologies = []
for slice_ix, slicetopology in tqdm.tqdm(slicetopologies_oi.iterrows(), total = len(slicetopologies_oi)):
    # to take the
    # signals = np.exp(np.log(signals_slices[slice_ix] + 1e-5).groupby(level = 0).mean().mean(1))
    signals = signals_slices[slice_ix].groupby(level = 0).mean().mean(1)
    signal_topologies.append(pd.DataFrame({"slice_ix":slice_ix, "signal":signals.values, "target":signals.index}))
signal_topologies = pd.concat(signal_topologies)

# %%
signal_topologies_mean = signal_topologies.join(slicetopologies_oi[["type"]], on = "slice_ix").groupby(["type", "target"])["signal"].mean().unstack()

# %%
# do max normalization for each mark
signal_topologies_mean_norm = signal_topologies_mean / signal_topologies_mean.values.max(0)

# %%
sns.heatmap(signal_topologies_mean_norm.T)

# %% [markdown]
# You can see that on average there is a clear link between some slice topologies and histone marks. For example, ridges are clearly linked with H3K36me3 and DNA methylation, probably because they are often found in highly expressed genes, and there is therefore a need to shut off aberrant transcription start sites (see e.g. https://www.embopress.org/doi/full/10.15252/embj.201796812). 

# %% [markdown]
# For now we just looked at the signal of a mark within the cell type itself: for a given differential accessibility slice within cell type X, we just looked at the ChIP-seq mark signal within cell type X.
#
# The question I would want you to explore now is: could you look at this differentially? In other words, how much higher is a mark in cell type X versus the same mark in the same region for all other cell types?
#
# For this, you will need to:
# - Extract the signal for each slice across all cell types.
# - Calculate the average signal within each slice for each cell type
# - Compare these average for cell type X (the cell type where the slice was found) with the average for all other cell types. I would calculate the fold change here, i.e. `(signal in cell type X)/(average signal in all other cell types)
# - Take the log of this fold change.
# - Calculate the average of the fold changes across each combination of slice type and target.
# - Create a heatmap as before

# %% [markdown]
# ## Starting here

# %%
display(clusters_oi.index)

# %%
slicetopologies_oi = slicetopologies.query("cluster in @clusters_oi.index").head(1000) # select the first 1000 regions
slicetopologies_oi.index.name = "slice_ix"

# %%
signals_slices_ = {}

if FULL_NB:
    # for each slice
    for slice_ix, slicetopology in tqdm.tqdm(slicetopologies_oi.iterrows(), total = len(slicetopologies_oi)):
        signals = []
        file_ixs = []

        # select all celltypes
        for file_ix, file in files_oi["file"].items():
            signals.append(np.array(file.values(slicetopology.chr, slicetopology.start, slicetopology.end)))
            file_ixs.append(file_ix)

        # stack the signal from the different files (i.e. samples)
        signals = pd.DataFrame(np.stack(signals), index = [files_oi.loc[file_ixs,"target"],files_oi.loc[file_ixs,"cluster"]]) 
        signals_slices_[slice_ix] = signals


# %%
if FULL_NB:
    pickle.dump(signals_slices_, (analysis_folder / "signals_slices_.pkl").open("wb"))

# %% [markdown]
# Load in the data again

# %%
signals_slices_ = pickle.load((analysis_folder / "signals_slices_.pkl").open("rb"))

# %% [markdown]
# Take the mean for each 

# %%
# Dictionary of the original cluster where the slice was found
og_cluster_dict = slicetopologies_oi["cluster"]
og_cluster_dict = og_cluster_dict.rename("og_cluster")
og_cluster_dict = og_cluster_dict.to_dict()

# %%
signal_topologies_ = []
for slice_ix, slicetopology in tqdm.tqdm(slicetopologies_oi.iterrows(), total = len(slicetopologies_oi)):
    # For each slice, take the average signal of each cell type 
    signals = signals_slices_[slice_ix].groupby(["target", "cluster"]).mean().mean(1)
    signals.name = "signal"
    signals = signals.to_frame()
    signals.reset_index(inplace=True)
    
    # (Mask column to calculate fold change)
    signals["is_og_cluster"] = signals["cluster"] == og_cluster_dict[slice_ix]
    
    signal_topologies_.append(pd.DataFrame({"slice_ix":slice_ix, "signal":signals.signal, "target":signals.target, "cluster": signals.cluster, "is_og_cluster": signals.is_og_cluster}))
signal_topologies_ = pd.concat(signal_topologies_)

# %%
display(signal_topologies_)

# %% [markdown]
# We can now calculate the fold change across different types of slices and different targets:

# %%
# Average signal of the original cell type
signal_topologies_og = signal_topologies_.loc[signal_topologies_["is_og_cluster"]][["slice_ix", "target", "signal"]].set_index(["slice_ix", "target"])
# Average signal in all other cell types
signal_topologies_mean_ = signal_topologies_.loc[~signal_topologies_["is_og_cluster"]][["slice_ix", "target", "signal"]].groupby(["slice_ix", "target"]).mean()

# Calculate fold change, apply log (not good for DNAme)
signal_topologies_fold_change = signal_topologies_og.join(signal_topologies_mean_, lsuffix='_og', rsuffix="_mean")
signal_topologies_fold_change["fc"] = signal_topologies_fold_change["signal_og"]/signal_topologies_fold_change["signal_mean"]
signal_topologies_fold_change["log_fc"] = np.log(signal_topologies_fold_change["fc"])
signal_topologies_fold_change.reset_index(inplace=True)
display(signal_topologies_fold_change)

# %%
# Calculate the average of the fold changes across each combination of slice type and target.
signal_topologies_fold_change_mean = signal_topologies_fold_change.join(slicetopologies_oi[["type"]], on = "slice_ix").groupby(["type", "target"])["log_fc"].mean().unstack()
display(signal_topologies_fold_change_mean)
sns.heatmap(signal_topologies_fold_change_mean[["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]].T)

# %%
# (comparison with the previous heatmap)
sns.heatmap(signal_topologies_mean_norm[["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]].T)

# %%
signal_topologies_fold_change_df = signal_topologies_fold_change.join(slicetopologies_oi[["type"]], on = "slice_ix")
#exclude DNAme:
signal_topologies_fold_change_df = signal_topologies_fold_change_df.loc[signal_topologies_fold_change_df["target"] != "DNAme"]

# %%
pairs = (
    [("H3K27ac", "canyon"), ("H3K27ac", "chain")],
    [("H3K27ac", "canyon"), ("H3K27ac", "flank")],
    [("H3K27ac", "canyon"), ("H3K27ac", "hill")],
    [("H3K27ac", "canyon"), ("H3K27ac", "peak")],
    [("H3K27ac", "canyon"), ("H3K27ac", "ridge")],
    [("H3K27ac", "canyon"), ("H3K27ac", "volcano")],
    [("H3K27ac", "chain"), ("H3K27ac", "flank")],
    [("H3K27ac", "chain"), ("H3K27ac", "hill")],
    [("H3K27ac", "chain"), ("H3K27ac", "peak")],
    [("H3K27ac", "chain"), ("H3K27ac", "ridge")],
    [("H3K27ac", "chain"), ("H3K27ac", "volcano")],
    [("H3K27ac", "flank"), ("H3K27ac", "hill")],
    [("H3K27ac", "flank"), ("H3K27ac", "peak")],
    [("H3K27ac", "flank"), ("H3K27ac", "ridge")],
    [("H3K27ac", "flank"), ("H3K27ac", "volcano")],
    [("H3K27ac", "hill"), ("H3K27ac", "peak")],
    [("H3K27ac", "hill"), ("H3K27ac", "ridge")],
    [("H3K27ac", "hill"), ("H3K27ac", "volcano")],
    [("H3K27ac", "peak"), ("H3K27ac", "ridge")],
    [("H3K27ac", "peak"), ("H3K27ac", "volcano")],
    [("H3K27ac", "ridge"), ("H3K27ac", "volcano")],
    
    [("H3K27me3", "canyon"), ("H3K27me3", "chain")],
    [("H3K27me3", "canyon"), ("H3K27me3", "flank")],
    [("H3K27me3", "canyon"), ("H3K27me3", "hill")],
    [("H3K27me3", "canyon"), ("H3K27me3", "peak")],
    [("H3K27me3", "canyon"), ("H3K27me3", "ridge")],
    [("H3K27me3", "canyon"), ("H3K27me3", "volcano")],
    [("H3K27me3", "chain"),  ("H3K27me3", "flank")],
    [("H3K27me3", "chain"),  ("H3K27me3", "hill")],
    [("H3K27me3", "chain"),  ("H3K27me3", "peak")],
    [("H3K27me3", "chain"),  ("H3K27me3", "ridge")],
    [("H3K27me3", "chain"),  ("H3K27me3", "volcano")],
    [("H3K27me3", "flank"),  ("H3K27me3", "hill")],
    [("H3K27me3", "flank"),  ("H3K27me3", "peak")],
    [("H3K27me3", "flank"),  ("H3K27me3", "ridge")],
    [("H3K27me3", "flank"),  ("H3K27me3", "volcano")],
    [("H3K27me3", "hill"),   ("H3K27me3", "peak")],
    [("H3K27me3", "hill"),   ("H3K27me3", "ridge")],
    [("H3K27me3", "hill"),   ("H3K27me3", "volcano")],
    [("H3K27me3", "peak"),   ("H3K27me3", "ridge")],
    [("H3K27me3", "peak"),   ("H3K27me3", "volcano")],
    [("H3K27me3", "ridge"),  ("H3K27me3", "volcano")],
    
    [("H3K36me3", "canyon"), ("H3K36me3", "chain")],
    [("H3K36me3", "canyon"), ("H3K36me3", "flank")],
    [("H3K36me3", "canyon"), ("H3K36me3", "hill")],
    [("H3K36me3", "canyon"), ("H3K36me3", "peak")],
    [("H3K36me3", "canyon"), ("H3K36me3", "ridge")],
    [("H3K36me3", "canyon"), ("H3K36me3", "volcano")],
    [("H3K36me3", "chain"),  ("H3K36me3", "flank")],
    [("H3K36me3", "chain"),  ("H3K36me3", "hill")],
    [("H3K36me3", "chain"),  ("H3K36me3", "peak")],
    [("H3K36me3", "chain"),  ("H3K36me3", "ridge")],
    [("H3K36me3", "chain"),  ("H3K36me3", "volcano")],
    [("H3K36me3", "flank"),  ("H3K36me3", "hill")],
    [("H3K36me3", "flank"),  ("H3K36me3", "peak")],
    [("H3K36me3", "flank"),  ("H3K36me3", "ridge")],
    [("H3K36me3", "flank"),  ("H3K36me3", "volcano")],
    [("H3K36me3", "hill"),   ("H3K36me3", "peak")],
    [("H3K36me3", "hill"),   ("H3K36me3", "ridge")],
    [("H3K36me3", "hill"),   ("H3K36me3", "volcano")],
    [("H3K36me3", "peak"),   ("H3K36me3", "ridge")],
    [("H3K36me3", "peak"),   ("H3K36me3", "volcano")],
    [("H3K36me3", "ridge"),  ("H3K36me3", "volcano")],
    
    [("H3K4me1", "canyon"), ("H3K4me1", "chain")],
    [("H3K4me1", "canyon"), ("H3K4me1", "flank")],
    [("H3K4me1", "canyon"), ("H3K4me1", "hill")],
    [("H3K4me1", "canyon"), ("H3K4me1", "peak")],
    [("H3K4me1", "canyon"), ("H3K4me1", "ridge")],
    [("H3K4me1", "canyon"), ("H3K4me1", "volcano")],
    [("H3K4me1", "chain"),  ("H3K4me1", "flank")],
    [("H3K4me1", "chain"),  ("H3K4me1", "hill")],
    [("H3K4me1", "chain"),  ("H3K4me1", "peak")],
    [("H3K4me1", "chain"),  ("H3K4me1", "ridge")],
    [("H3K4me1", "chain"),  ("H3K4me1", "volcano")],
    [("H3K4me1", "flank"),  ("H3K4me1", "hill")],
    [("H3K4me1", "flank"),  ("H3K4me1", "peak")],
    [("H3K4me1", "flank"),  ("H3K4me1", "ridge")],
    [("H3K4me1", "flank"),  ("H3K4me1", "volcano")],
    [("H3K4me1", "hill"),   ("H3K4me1", "peak")],
    [("H3K4me1", "hill"),   ("H3K4me1", "ridge")],
    [("H3K4me1", "hill"),   ("H3K4me1", "volcano")],
    [("H3K4me1", "peak"),   ("H3K4me1", "ridge")],
    [("H3K4me1", "peak"),   ("H3K4me1", "volcano")],
    [("H3K4me1", "ridge"),  ("H3K4me1", "volcano")],
    
    [("H3K4me3", "canyon"), ("H3K4me3", "chain")],
    [("H3K4me3", "canyon"), ("H3K4me3", "flank")],
    [("H3K4me3", "canyon"), ("H3K4me3", "hill")],
    [("H3K4me3", "canyon"), ("H3K4me3", "peak")],
    [("H3K4me3", "canyon"), ("H3K4me3", "ridge")],
    [("H3K4me3", "canyon"), ("H3K4me3", "volcano")],
    [("H3K4me3", "chain"),  ("H3K4me3", "flank")],
    [("H3K4me3", "chain"),  ("H3K4me3", "hill")],
    [("H3K4me3", "chain"),  ("H3K4me3", "peak")],
    [("H3K4me3", "chain"),  ("H3K4me3", "ridge")],
    [("H3K4me3", "chain"),  ("H3K4me3", "volcano")],
    [("H3K4me3", "flank"),  ("H3K4me3", "hill")],
    [("H3K4me3", "flank"),  ("H3K4me3", "peak")],
    [("H3K4me3", "flank"),  ("H3K4me3", "ridge")],
    [("H3K4me3", "flank"),  ("H3K4me3", "volcano")],
    [("H3K4me3", "hill"),   ("H3K4me3", "peak")],
    [("H3K4me3", "hill"),   ("H3K4me3", "ridge")],
    [("H3K4me3", "hill"),   ("H3K4me3", "volcano")],
    [("H3K4me3", "peak"),   ("H3K4me3", "ridge")],
    [("H3K4me3", "peak"),   ("H3K4me3", "volcano")],
    [("H3K4me3", "ridge"),  ("H3K4me3", "volcano")],
    
    [("H3K9me3", "canyon"), ("H3K9me3", "chain")],
    [("H3K9me3", "canyon"), ("H3K9me3", "flank")],
    [("H3K9me3", "canyon"), ("H3K9me3", "hill")],
    [("H3K9me3", "canyon"), ("H3K9me3", "peak")],
    [("H3K9me3", "canyon"), ("H3K9me3", "ridge")],
    [("H3K9me3", "canyon"), ("H3K9me3", "volcano")],
    [("H3K9me3", "chain"),  ("H3K9me3", "flank")],
    [("H3K9me3", "chain"),  ("H3K9me3", "hill")],
    [("H3K9me3", "chain"),  ("H3K9me3", "peak")],
    [("H3K9me3", "chain"),  ("H3K9me3", "ridge")],
    [("H3K9me3", "chain"),  ("H3K9me3", "volcano")],
    [("H3K9me3", "flank"),  ("H3K9me3", "hill")],
    [("H3K9me3", "flank"),  ("H3K9me3", "peak")],
    [("H3K9me3", "flank"),  ("H3K9me3", "ridge")],
    [("H3K9me3", "flank"),  ("H3K9me3", "volcano")],
    [("H3K9me3", "hill"),   ("H3K9me3", "peak")],
    [("H3K9me3", "hill"),   ("H3K9me3", "ridge")],
    [("H3K9me3", "hill"),   ("H3K9me3", "volcano")],
    [("H3K9me3", "peak"),   ("H3K9me3", "ridge")],
    [("H3K9me3", "peak"),   ("H3K9me3", "volcano")],
    [("H3K9me3", "ridge"),  ("H3K9me3", "volcano")],
)


# %%
sig_pairs = (
    [("H3K27ac", "canyon"), ("H3K27ac", "chain")],
    [("H3K27ac", "canyon"), ("H3K27ac", "flank")],
    [("H3K27ac", "canyon"), ("H3K27ac", "hill")],
    [("H3K27ac", "canyon"), ("H3K27ac", "peak")],
    [("H3K27ac", "canyon"), ("H3K27ac", "ridge")],
    [("H3K27ac", "canyon"), ("H3K27ac", "volcano")],
    [("H3K27ac", "chain"), ("H3K27ac", "flank")],
    [("H3K27ac", "chain"), ("H3K27ac", "hill")],
    [("H3K27ac", "chain"), ("H3K27ac", "peak")],
    [("H3K27ac", "flank"), ("H3K27ac", "volcano")],
    [("H3K27ac", "peak"), ("H3K27ac", "volcano")],
    [("H3K27ac", "ridge"), ("H3K27ac", "volcano")],
    
    [("H3K27me3", "flank"),  ("H3K27me3", "volcano")],
    
    [("H3K36me3", "canyon"), ("H3K36me3", "chain")],
    [("H3K36me3", "canyon"), ("H3K36me3", "hill")],
    [("H3K36me3", "canyon"), ("H3K36me3", "ridge")],
    [("H3K36me3", "canyon"), ("H3K36me3", "volcano")],
    [("H3K36me3", "flank"),  ("H3K36me3", "volcano")],
    [("H3K36me3", "peak"),   ("H3K36me3", "volcano")],
    
    [("H3K4me1", "canyon"), ("H3K4me1", "volcano")],
    [("H3K4me1", "chain"),  ("H3K4me1", "volcano")],
    [("H3K4me1", "flank"),  ("H3K4me1", "volcano")],
    [("H3K4me1", "hill"),   ("H3K4me1", "volcano")],
    [("H3K4me1", "peak"),   ("H3K4me1", "volcano")],
    [("H3K4me1", "ridge"),  ("H3K4me1", "volcano")],
    
    [("H3K4me3", "canyon"), ("H3K4me3", "chain")],
    [("H3K4me3", "canyon"), ("H3K4me3", "flank")],
    [("H3K4me3", "canyon"), ("H3K4me3", "hill")],
    [("H3K4me3", "canyon"), ("H3K4me3", "peak")],
    [("H3K4me3", "canyon"), ("H3K4me3", "ridge")],
    [("H3K4me3", "canyon"), ("H3K4me3", "volcano")],
    [("H3K4me3", "chain"),  ("H3K4me3", "flank")],
    [("H3K4me3", "chain"),  ("H3K4me3", "hill")],
    [("H3K4me3", "chain"),  ("H3K4me3", "peak")],
    [("H3K4me3", "flank"),  ("H3K4me3", "volcano")],
    [("H3K4me3", "hill"),   ("H3K4me3", "volcano")],
    [("H3K4me3", "peak"),   ("H3K4me3", "volcano")],
)

# %%
sns.set(rc={'figure.figsize':(16,9)})
types = ["canyon", "chain", "flank", "hill", "peak", "ridge", "volcano"]
customPalette = sns.set_palette(sns.color_palette([chd.slicetypes.types_info.loc[type_, "color"] for type_ in types]))
        

with sns.plotting_context("notebook", font_scale = 1.4):
    # Create new plot
    fig, ax = plt.subplots(1, 1, figsize=(16,9))
    fig.patch.set_alpha(1)
    #getattr(ax, set_scale)("log")

    # Plot with seaborn
    ax = sns.boxplot(ax=ax, data= signal_topologies_fold_change_df, x = "target", y = "log_fc", hue = "type", palette = customPalette, hue_order = types)

    # Add annotations
    annotator = Annotator(ax, sig_pairs, data= signal_topologies_fold_change_df, x = "target", y = "log_fc", hue = "type", palette = customPalette, hue_order = types)
    annotator.configure(test="t-test_ind", comparisons_correction="bonferroni")
    _, corrected_results = annotator.apply_and_annotate()

    # Label and show
    ax.legend()
    plt.title("Log fold change of topology types for different targets")
    ax.set_ylabel("log_fc")
    ax.set_xlabel("Target", labelpad=20)
    plt.show()

# %%
signal_topologies_df = signal_topologies.join(slicetopologies_oi[["type"]], on = "slice_ix")

# %%
sns.boxplot(data= signal_topologies_df.loc[signal_topologies_df["target"] != "DNAme"], x = "target", y = "signal", hue = "type", palette = customPalette, hue_order = types)
plt.yscale('log')

# %% [markdown]
# ### All datapoints

# %%
all_slicetopologies_oi = slicetopologies.query("cluster in @clusters_oi.index") # select all regions
all_slicetopologies_oi.index.name = "slice_ix"

# %%
all_signals_slices_ = {}

if FULL_NB:
    # for each slice
    for slice_ix, slicetopology in tqdm.tqdm(all_slicetopologies_oi.iterrows(), total = len(all_slicetopologies_oi)):
        signals = []
        file_ixs = []

        # select all celltypes
        for file_ix, file in files_oi["file"].items():
            signals.append(np.array(file.values(slicetopology.chr, slicetopology.start, slicetopology.end)))
            file_ixs.append(file_ix)

        # stack the signal from the different files (i.e. samples)
        signals = pd.DataFrame(np.stack(signals), index = [files_oi.loc[file_ixs,"target"],files_oi.loc[file_ixs,"cluster"]]) 
        all_signals_slices_[slice_ix] = signals


# %%
if FULL_NB:
    pickle.dump(all_signals_slices_, (analysis_folder / "all_signals_slices_.pkl").open("wb"))

# %% [markdown]
# Load in the data again

# %%
all_signals_slices_ = pickle.load((analysis_folder / "all_signals_slices_.pkl").open("rb"))

# %% [markdown]
# Take the mean for each 

# %%
# Dictionary of the original cluster where the slice was found
all_og_cluster_dict = all_slicetopologies_oi["cluster"]
all_og_cluster_dict = all_og_cluster_dict.rename("og_cluster")
all_og_cluster_dict = all_og_cluster_dict.to_dict()

# %%

all_signal_topologies_ = []
for slice_ix, slicetopology in tqdm.tqdm(all_slicetopologies_oi.iterrows(), total = len(all_slicetopologies_oi)):
    # For each slice, take the average signal of each cell type 
    signals = all_signals_slices_[slice_ix].groupby(["target", "cluster"]).mean().mean(1)
    signals.name = "signal"
    signals = signals.to_frame()
    signals.reset_index(inplace=True)
    
    # (Mask column to calculate fold change)
    signals["is_og_cluster"] = signals["cluster"] == all_og_cluster_dict[slice_ix]
    
    all_signal_topologies_.append(pd.DataFrame({"slice_ix":slice_ix, "signal":signals.signal, "target":signals.target, "cluster": signals.cluster, "is_og_cluster": signals.is_og_cluster}))
all_signal_topologies_ = pd.concat(all_signal_topologies_)

# %%
# Average signal of the original cell type
all_signal_topologies_og = all_signal_topologies_.loc[all_signal_topologies_["is_og_cluster"]][["slice_ix", "target", "signal"]].set_index(["slice_ix", "target"])
# Average signal in all other cell types
all_signal_topologies_mean_ = all_signal_topologies_.loc[~all_signal_topologies_["is_og_cluster"]][["slice_ix", "target", "signal"]].groupby(["slice_ix", "target"]).mean()

# Calculate fold change, apply log (not good for DNAme)
all_signal_topologies_fold_change = all_signal_topologies_og.join(all_signal_topologies_mean_, lsuffix='_og', rsuffix="_mean")
all_signal_topologies_fold_change["fc"] = all_signal_topologies_fold_change["signal_og"]/all_signal_topologies_fold_change["signal_mean"]
all_signal_topologies_fold_change["log_fc"] = np.log(all_signal_topologies_fold_change["fc"])
all_signal_topologies_fold_change.reset_index(inplace=True)
display(all_signal_topologies_fold_change)

# %%
# !pwd

# %%
# Calculate the average of the fold changes across each combination of slice type and target.
all_signal_topologies_fold_change_mean = all_signal_topologies_fold_change.join(all_slicetopologies_oi[["type"]], on = "slice_ix").groupby(["type", "target"])["log_fc"].mean().unstack()
display(all_signal_topologies_fold_change_mean)
sns.heatmap(all_signal_topologies_fold_change_mean[["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]].T)

# %%
sns.heatmap(signal_topologies_fold_change_mean[["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]].T)

# %%
all_signal_topologies_fold_change_df = all_signal_topologies_fold_change.join(all_slicetopologies_oi[["type"]], on = "slice_ix")
#exclude DNAme:
all_signal_topologies_fold_change_df = all_signal_topologies_fold_change_df.loc[all_signal_topologies_fold_change_df["target"] != "DNAme"]

# %%
sns.boxplot(data= all_signal_topologies_fold_change_df.loc[all_signal_topologies_fold_change_df["target"] != "DNAme"], x = "target", y = "log_fc", hue = "type", palette = customPalette, hue_order = types)

# %% [markdown]
# ### Absolute signal values

# %%
if not FULL_NB:
    
    all_signals_slices = {}
    # for each slice
    for slice_ix, slicetopology in tqdm.tqdm(all_slicetopologies_oi.iterrows(), total = len(all_slicetopologies_oi)):
        signals = []
        file_ixs = []

        # select  for only the celltypes in which the slice was differential
        for file_ix, file in files_oi.loc[files_oi["cluster"] == slicetopology["cluster"]]["file"].items():
            signals.append(np.array(file.values(slicetopology.chr, slicetopology.start, slicetopology.end)))
            file_ixs.append(file_ix)

        # stack the signal from the different files (i.e. samples) and then calculate the mean for each target (=mark)
        signals = pd.DataFrame(np.stack(signals), index = file_ixs)
        signals.index = files_oi.loc[file_ixs, "target"]
        all_signals_slices[slice_ix] = signals

# %%
if notFULL_NB:
    pickle.dump(signals_slices, (analysis_folder / "all_signals_slices.pkl").open("wb"))

# %% [markdown]
# Load in the data again

# %%
all_signals_slices = pickle.load((analysis_folder / "all_signals_slices.pkl").open("rb"))

# %% [markdown]
# Take the mean for each 

# %%
all_signal_topologies = []
for slice_ix, slicetopology in tqdm.tqdm(all_slicetopologies_oi.iterrows(), total = len(all_slicetopologies_oi)):
    # to take the
    # signals = np.exp(np.log(signals_slices[slice_ix] + 1e-5).groupby(level = 0).mean().mean(1))
    signals = all_signals_slices[slice_ix].groupby(level = 0).mean().mean(1)
    all_signal_topologies.append(pd.DataFrame({"slice_ix":slice_ix, "signal":signals.values, "target":signals.index}))
all_signal_topologies = pd.concat(all_signal_topologies)

# %%
all_signal_topologies_mean = all_signal_topologies.join(all_slicetopologies_oi[["type"]], on = "slice_ix").groupby(["type", "target"])["signal"].mean().unstack()

# %%
# do max normalization for each mark
all_signal_topologies_mean_norm = all_signal_topologies_mean / all_signal_topologies_mean.values.max(0)

# %%
sns.heatmap(all_signal_topologies_mean_norm.T)

# %%
