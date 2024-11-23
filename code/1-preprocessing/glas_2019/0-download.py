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


# %% metadata={}
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

import pysam


# %% metadata={}
import chromatinhd as chd
data_folder = chd.get_output() /"data"/"glas_2019"
dataset_folder = chd.get_output() / "datasets" / "glas_2019"
chd.set_default_device("cuda:0")

# %% metadata={}
data_folder.mkdir(exist_ok = True)

# %% [markdown]
# ## Download

# %%
# !ls {data_folder}/peaks

# %%
if not pathlib.Path(f"{data_folder}/GSE128662_RAW.tar").exists():
    import os
    os.system(f"wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE128nnn/GSE128662/suppl/GSE128662_RAW.tar -O {data_folder}/GSE128662_RAW.tar")

# %%
(data_folder / "peaks").mkdir(exist_ok=True)
# ! tar -xvf {data_folder}/GSE128662_RAW.tar -C {data_folder}/peaks

# %%
# convert peaks
import glob
import re
for file in glob.glob(str(data_folder / "peaks" / "GSM*.peak.txt.gz")):
    peaks = pd.read_table(file, comment = "#", names ="PeakID	chr	start	end	strand	Normalized Tag Count	focus ratio	findPeaks Score	Fold Change vs Local	p-value vs Local	Clonal Fold Change".split("\t"))[["chr", "start", "end"]]
    peaks.to_csv(pathlib.Path(re.sub("GSM[0-9]*_", "", file.split(".")[0])).with_suffix(".bed"), sep = "\t", index = False, header = False)

# %%
data_folder = pathlib.Path("/srv/data/liesbetm/Projects/u_mgu/Wouter/ChrisGlass_atac/outputBowtie")

# %% [markdown]
# ## Load obs

# %% metadata={}
obs = pd.DataFrame([
    ["BloodLy6cHi_mouse8_ATAC_notx.bam", "blood", "monocyte", "0", "WT", "Ly6cHi"],
    ["BloodLy6cHi_mouse3_ATAC_notx.bam", "blood", "monocyte", "0", "WT", "Ly6cHi"],
    ["RLM_mouse58_ATAC_DT24h.bam", "liver", "RLM", "24", "WT", ""],
    ["RLM_mouse215_ATAC_DT24h.bam", "liver", "RLM", "24", "WT", ""],
    ["RLM_mouse58_ATAC_DT24h.bam", "liver", "RLM", "24", "WT", ""],
    ["RLM_mouse106_ATAC_DT24h.bam", "liver", "RLM", "24", "WT", ""],
    ["RLM_mouse212_ATAC_DT48h.bam", "liver", "RLM", "48", "WT", ""],
    ["RLM_mouse195_ATAC_DT48h.bam", "liver", "RLM", "48", "WT", ""],
    ["KC_mouse60_ATAC_PBS.bam", "liver", "KC", "final", "WT", ""],
    ["KC_mouse216_ATAC_PBS.bam", "liver", "KC", "final", "WT", ""],
    ["KcClecPosTim4Neg_ATAC_NoTx_KO_rep1.bam", "liver", "KC", "final", "Nr1h3-KO", "Clec4f+Tim4-"],
    ["KcClecPosTim4Neg_ATAC_NoTx_KO_rep2.bam", "liver", "KC", "final", "Nr1h3-KO", "Clec4f+Tim4-"],
    ["KcClecPosTim4Pos_ATAC_NoTx_KO_rep1.bam", "liver", "KC", "final", "Nr1h3-KO", "Clec4f+Tim4+"],
    ["KcClecPosTim4Pos_ATAC_NoTx_KO_rep2.bam", "liver", "KC", "final", "Nr1h3-KO", "Clec4f+Tim4+"],
    ["Smad4KO_KC_ATAC_NoTx_366.bam", "liver", "KC", "final", "Smad4-KO", ""],
    ["Smad4KO_KC_ATAC_NoTx_366.bam", "liver", "KC", "final", "Smad4-KO", ""],
    ["KcClecNegTim4Pos_ATAC_NoTx_Control_rep1.bam", "liver", "KC", "final", "Nr1h3-WT", ""],
    ["KcClecNegTim4Pos_ATAC_NoTx_Control_rep1.bam", "liver", "KC", "final", "Nr1h3-WT", ""],
], columns = ["file", "tissue", "celltype", "timepoint", "genetic", "sort"])
# obs = pd.read_table('/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/glas_2019/samples.tsv', sep = " ", engine="python").fillna(np.nan)

# %% metadata={}
# add condition to obs
conditions = []
for _, row in obs.iterrows():
    condition = row["tissue"] + "_" + row["celltype"]
    if not pd.isnull(row["timepoint"]):
        condition += "_" + str(row["timepoint"])
    if not pd.isnull(row["genetic"]) and row["genetic"] != "":
        condition += "_" + str(row["genetic"])
    if not pd.isnull(row["sort"]) and row["sort"] != "":
        condition += "_" + str(row["sort"])
    conditions.append(condition)
obs["condition"] = conditions

obs["path"] = obs["file"].apply(lambda x: "/srv/data/liesbetm/Projects/u_mgu/Wouter/ChrisGlass_atac/outputBowtie/" + x)

# %% metadata={}
# import genomepy
# genomepy.install_genome("mm10", genomes_dir="/data/genome/")

# process all fragments
# if not (dataset_folder / "fragments" / "all").exists():
if True:
    fasta_file = "/srv/data/genomes/mm10/mm10.fa"
    chromosomes_file = "/srv/data/genomes/mm10/mm10.fa.sizes"

    regions = chd.data.Regions.from_chromosomes_file(chromosomes_file, path = dataset_folder / "regions" / "all")

    fragments_all = chd.data.Fragments.from_alignments(
        obs,
        regions=regions,
        path=dataset_folder / "fragments" / "all",
        overwrite = True,
        batch_size = 10e7,
        paired = False,
        remove_duplicates = True,
    )
    fragments_all.create_regionxcell_indptr()

# %% [markdown]
# ## Get diffexp

# %%
if not (dataset_folder / "mmc5.xlsx").exists():
    os.system(f"wget https://ars.els-cdn.com/content/image/1-s2.0-S1074761319303735-mmc5.xlsx -O {dataset_folder}/mmc5.xlsx")

# %% metadata={}
mmc5 = pd.read_excel(dataset_folder / "mmc5.xlsx", sheet_name = "Table S4")

# %% metadata={}
mmc5["symbol"] = mmc5["Annotation/Divergence"].str.split("|").str[0]

# %%
import os
if not (dataset_folder / "mmc2.xlsx").exists():
    os.system(f"wget https://www.cell.com/cms/10.1016/j.immuni.2019.09.002/attachment/75234d89-01cc-4cd0-8214-eccbb3b58e6e/mmc2.xlsx -O {dataset_folder}/mmc2.xlsx")

# %% metadata={}
mmc2 = pd.read_excel("mmc2.xlsx")

# %%
mmc2["significant_padj"] = (mmc2[mmc2.columns[mmc2.columns.str.endswith("padj")]] < 5e-2).any(axis = 1)
mmc2["significant_log2FoldChange"] = (np.abs(mmc2[mmc2.columns[mmc2.columns.str.endswith("log2FoldChange")]]) > 1).any(axis = 1)

mmc2["significant"] = mmc2["significant_padj"] & mmc2["significant_log2FoldChange"]

# %% metadata={}
mmc2["symbol"] = mmc2["Annotation/Divergence"].str.split("|").str[0]

# %% [markdown]
# ## Create dataset

# %% metadata={}
fragments_all = chd.data.Fragments.from_path(dataset_folder / "fragments" / "all")

# %% metadata={}
symbols = "Clec4f, Cd207, Cd5l, Cdh5, Cd38, Nr1h3, Id3, Itga9".split(", ")
symbols = mmc2.query("significant")["symbol"].values.tolist() + symbols
symbols = list(set(symbols))

# %% metadata={}
regions_name = "100k100k"
transcripts = chd.biomart.get_canonical_transcripts(chd.biomart.Dataset.from_genome("mm10"), filter_canonical = False, 
    symbols = symbols
)
regions = chd.data.Regions.from_transcripts(transcripts, [-100000, 100000], path = dataset_folder / "regions" / regions_name, overwrite = True)
# regions.coordinates["symbol"] = symbols
regions.coordinates = regions.coordinates

# %% metadata={}
fragments = chd.data.fragments.FragmentsView.from_fragments(
    fragments_all,
    regions = regions,
    path = dataset_folder / "fragments" / regions_name,
    overwrite = True
)
fragments.create_regionxcell_indptr2(overwrite = True)

# %%
fragments.var["symbol"] = transcripts["symbol"]
fragments.var = fragments.var

# %% [markdown]
# ## Motifscan

# %%
dataset_name = "liverphx"
dataset_name = "glas_2019"

# %%
motifscan_name = "hocomocov12" + "_" + "1e-4"
# motifscan_name = "hocomocov12" + "_" + "5"
genome_folder = pathlib.Path("/srv/data/genomes/GRCm39")
parent = chd.data.Motifscan(genome_folder / "motifscans" / motifscan_name)

motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
    parent,
    regions,
    path=chd.get_output()
    / "datasets"
    / dataset_name
    / "motifscans"
    / regions_name
    / motifscan_name,
    overwrite=True,
)

# %% [markdown]
# ## Clustering

# %%
clustering = chd.data.Clustering.from_labels(
    obs["condition"],
    var=obs.groupby("condition").first()[["tissue", "celltype", "timepoint"]],
    path=dataset_folder / "clusterings" / "cluster",
    overwrite=True,
)

# %%
clustering = chd.data.Clustering.from_labels(
    obs["file"],
    var=obs.groupby("file").first()[["tissue", "celltype", "timepoint"]],
    path=dataset_folder / "clusterings" / "file",
    overwrite=True,
)

# %% [markdown]
# ## Training

# %%
fold = {
    "cells_train": np.arange(len(fragments.obs)),
    "cells_test": np.arange(len(fragments.obs)),
    "cells_validation": np.arange(len(fragments.obs)),
}

# %%
latent_name = "cluster"
# latent_name = "file"
clustering = chd.data.Clustering(dataset_folder / "clusterings" / latent_name)
model_folder = (
    chd.get_output() / "diff" / "glas_2019" / "binary" / "split" / regions_name / latent_name
)


# %%
import chromatinhd.models.diff.model.binary

model = chd.models.diff.model.binary.Model.create(
    fragments,
    clustering,
    fold=fold,
    encoder="shared",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale=0.5,
        bias_regularization=True,
        bias_p_scale=0.5,
        binwidths=(5000, 1000, 500, 100, 50),
    ),
    path=model_folder / "model",
    overwrite = True,
)

# %%
model.train_model(
    n_epochs=100, n_regions_step=50, early_stopping=False, do_validation=True, lr=1e-2
)

# %%
model.trace.plot()

# %%
model.save_state()

# %% [markdown]
# ## Inference

# %%
model = chd.models.diff.model.binary.Model(model_folder / "model")

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    reset=True,
)

# %%
regionpositional.score(
    [model],
    fragments=fragments,
    clustering=clustering,
    device="cpu",
)
regionpositional

# %%
gene_id = fragments.regions.coordinates.query("symbol == 'Itga9'").index[0]
regionpositional.probs[gene_id].to_pandas().T.plot()

# %%
