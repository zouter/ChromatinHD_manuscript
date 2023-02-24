# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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

import tqdm.auto as tqdm
# -

import chromatinhd as chd
import tempfile

import requests

url = "https://www.encodeproject.org/search/?type=Experiment&assay_title=Mint-ChIP-seq&biosample_ontology.term_name=naive+thymus-derived+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=naive+thymus-derived+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=naive+B+cell&biosample_ontology.term_name=B+cell&biosample_ontology.term_name=CD14-positive+monocyte&biosample_ontology.term_name=natural+killer+cell&biosample_ontology.term_name=immature+natural+killer+cell&biosample_ontology.term_name=dendritic+cell&limit=1000&assay_title=Histone+ChIP-seq&control_type!=*&target.label=H3K27ac&target.label=H3K4me3&target.label=H3K27me3&target.label=H3K36me3&target.label=H3K9me3&target.label=H3K4me1&files.file_type=bigWig"
url = "https://www.encodeproject.org/metadata/?assay_title=WGBS&assay_title=Mint-ChIP-seq&biosample_ontology.term_name=naive+thymus-derived+CD8-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=naive+thymus-derived+CD4-positive%2C+alpha-beta+T+cell&biosample_ontology.term_name=naive+B+cell&biosample_ontology.term_name=B+cell&biosample_ontology.term_name=CD14-positive+monocyte&biosample_ontology.term_name=natural+killer+cell&biosample_ontology.term_name=immature+natural+killer+cell&biosample_ontology.term_name=dendritic+cell&assay_title=Histone+ChIP-seq&control_type%21=%2A&files.file_type=bigWig&type=Experiment&files.analyses.status=released&files.preferred_default=true"
# &target.label=H3K27ac&target.label=H3K4me3&target.label=H3K27me3&target.label=H3K36me3&target.label=H3K9me3&target.label=H3K4me1

obj = requests.get(url)

import json

import io

files = pd.read_table(io.StringIO(obj.content.decode("utf-8")))

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

assert biosamples_oi["Biosample term name"].isin(files["Biosample term name"]).all()
assert files["Biosample term name"].isin(biosamples_oi["Biosample term name"]).all()

# ### Download files

encode_folder = chd.get_output() / "data" / "encode"
encode_folder_relative = encode_folder.relative_to(chd.get_git_root())

# + tags=[]
# !ln -s ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/{encode_folder_relative} {encode_folder}
# -

bw_folder = encode_folder / "immune"
bw_folder.mkdir(exist_ok = True, parents = True)

files["filename"] = files["File download URL"].str.split("/").str[-1]

import urllib

# + tags=[]
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
# -

for _, file in files.iterrows():
    if not (bw_folder / file["filename"]).exists():
        print(file["filename"])
        urllib.request.urlretrieve(file["File download URL"], bw_folder / file["filename"], )

# ### Check out

files["Experiment target"].value_counts()

files["Biosample term name"].value_counts()

files["Output type"].value_counts()

import pyBigWig

import chromatinhd as chd

import chromatinhd.conservation

# +
idx_oi = files.query("`Biosample term name` == 'natural killer cell'").query("`Experiment target` == 'H3K4me3-human'").index[0]
idx_oi = files.query("`Biosample term name` == 'dendritic cell'").query("`Experiment target` == 'H3K4me3-human'").index[0]
idx_oi = files.query("`Biosample term name` == 'CD14-positive monocyte'").query("`Experiment target` == 'H3K4me3-human'").index[0]
idx_oi = files.query("`Biosample term name` == 'naive thymus-derived CD8-positive, alpha-beta T cell'").query("`Experiment target` == 'H3K4me3-human'").index[0]

# repressive
# idx_oi = files.query("`Biosample term name` == 'natural killer cell'").query("`Experiment target` == 'H3K27me3-human'").index[1]
# idx_oi = files.query("`Biosample term name` == 'naive thymus-derived CD4-positive, alpha-beta T cell'").query("`Experiment target` == 'H3K27me3-human'").index[1]
# idx_oi = files.query("`Biosample term name` == 'CD14-positive monocyte'").query("`Experiment target` == 'H3K27me3-human'").index[5]
# idx_oi = files.query("`Biosample term name` == 'dendritic cell'").query("`Experiment target` == 'H3K27me3-human'").index[0]
# -

conservation = chd.conservation.BigWig(files.iloc[idx_oi]["File download URL"])

# +
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"

folder_data_preproc = folder_data / dataset_name
# -

promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

promoter = promoters.loc[transcriptome.gene_id("CD244")]

plotdata = pd.DataFrame({
    "conservation":conservation.get_values(promoter.chr, promoter.start, promoter.end),
    "position":np.arange(window[0], window[1])[::promoter["strand"]]
})

fig, ax = plt.subplots(figsize = (15, 2))
ax.plot(plotdata["position"], plotdata["conservation"])

files_oi = pd.merge(files, biosamples_oi).query("cluster in ['B', 'CD4 T']")
# files_oi = pd.merge(files, biosamples_oi).query("cluster in ['NK', 'cDCs']")
files_oi = files_oi.query("`Experiment target` == 'H3K4me3-human'")
# files_oi = files_oi.query("`Experiment target` == 'H3K27me3-human'")
files_oi = files_oi.copy()

files_oi["file"] = [chd.conservation.BigWig(file["File download URL"]) for _, file in files_oi.iterrows()]

plotdata = []
for file_ix, file in files_oi.iterrows():
    conservation = file["file"]
    cons = conservation.get_values(promoter.chr, promoter.start, promoter.end)
    plotdata_file = pd.DataFrame({
        "conservation":np.log(cons+0.01),
        "position":np.arange(window[0], window[1])[::promoter["strand"]],
        "file_ix":file_ix
    })
    plotdata.append(plotdata_file)
plotdata = pd.concat(plotdata)

cluster_info = pd.DataFrame({"cluster":files_oi["cluster"].unique()}).set_index("cluster")
cluster_info["color"] = sns.color_palette(n_colors = len(cluster_info))

files_oi.index.name = "file_ix"

fig, ax = plt.subplots(figsize = (15, 2))
for file_ix, file in files_oi.iterrows():
    plotdata_file = plotdata.query("file_ix == @file_ix")
    ax.plot(plotdata_file["position"], plotdata_file["conservation"], color = cluster_info.loc[file["cluster"], "color"])

plotdata_mean = plotdata.join(files_oi, on = "file_ix").groupby(["cluster", "position"])["conservation"].mean().to_frame()
np.exp(plotdata_mean.unstack().T).plot()

plotdata_mean = plotdata.join(files_oi, on = "file_ix").groupby(["cluster", "position"])["conservation"].mean().to_frame()
np.exp(plotdata_mean.unstack().T).plot()

fig, ax = plt.subplots(figsize = (15, 2))
for file_ix, file in files_oi.iterrows():
    plotdata_file = plotdata.query("file_ix == @file_ix")
    ax.plot(plotdata_file["position"], plotdata_file["conservation"], color = cluster_info.loc[file["cluster"], "color"])
