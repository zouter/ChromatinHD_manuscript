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

import seaborn as sns
sns.set_style('ticks')

import pickle

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k_gran"
# dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
# dataset_name = "brain"
dataset_name = "hspc"
dataset_name = "hspc_gmp"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
software_folder = chd.get_git_root() / "software"

# %% [markdown]
# ## MACS2

# %%
# loading
latent_name = "leiden_1"
latent_name = "leiden_0.1"
# latent_name = "celltype"
# latent_name = "overexpression"
dataset_folder = chd.get_output() / "datasets" / dataset_name
latent_folder = dataset_folder / "latent"

clustering = chd.data.Clustering(latent_folder / latent_name)

# %%
peaks_folder = folder_root / "peaks" / dataset_name / ("macs2_" + latent_name)
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
tmp_fragments_folder = peaks_folder / "tmp"
tmp_fragments_folder.mkdir(exist_ok = True)

# %%
import gzip

# %%
latent = pd.get_dummies(clustering.labels)

# %%
# extract fragments for each cluster
# should take a couple of minutes for pbmc10k
cluster_fragments = [(tmp_fragments_folder / (str(cluster_ix) + ".tsv")).open("w") for cluster_ix in range(clustering.n_clusters)]
cell_to_cluster = dict(zip(latent.index, np.where(latent)[1]))

n = 0
for l in gzip.GzipFile((folder_data_preproc / "atac_fragments.tsv.gz"), "r"):
    l = l.decode("utf-8")
    if l.startswith("#"):
        continue
    cell = l.split("\t")[3].strip()
    
    if cell in cell_to_cluster:
        n += 1
        cluster_fragments[cell_to_cluster[cell]].write(l)
    else:
        pass
        # print("wat")

# %%
# !ls {peaks_folder}/tmp

# %%
for cluster_ix in range(clustering.n_clusters):
    # !echo 'cd {peaks_folder} && macs2 callpeak --nomodel -t {peaks_folder}/tmp/{cluster_ix}.tsv -f BEDPE && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks_{cluster_ix}.bed'

# %%
with (peaks_folder / "peaks.bed").open("w") as outfile:
    for cluster_ix in range(clustering.n_clusters):
        bed = pd.read_table(peaks_folder / ("peaks_" + str(cluster_ix) + ".bed"), names = ["chr", "start", "end"], usecols = range(3))
        bed["strand"] = cluster_ix
        outfile.write(bed.to_csv(sep = "\t", header = False, index = False))

# %%
# !head -n 20 {(peaks_folder / "peaks.bed")}

# %%
# !rm -r {peaks_folder}/tmp

# %% [markdown]
# ## Merged

# %%
original_peaks_folder = folder_root / "peaks" / dataset_name / ("macs2_" + latent_name)

# %%
peaks_folder = folder_root / "peaks" / dataset_name / ("macs2_" + latent_name + "_merged")
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
import pybedtools
pybedtools.BedTool(original_peaks_folder / "peaks.bed").sort().merge().saveas(peaks_folder / "peaks.bed")

# %%
x = pybedtools.BedTool(original_peaks_folder / "peaks.bed").sort().merge()
x.to_dataframe().to_csv(peaks_folder / "peaks.bed", header = False, index = False, sep = "\t")
