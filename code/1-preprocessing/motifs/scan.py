# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
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

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import torch_scatter
import torch
import math

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "pbmc10k"
# organism = "hs"

dataset_name = "pbmc10k/subsets/top250"
organism = "hs"

# dataset_name = "hspc"
# organism = "hs"

folder_data_preproc = folder_data / dataset_name

# %%
import gzip
motifs_folder = chd.get_output() / "data" / "motifs" / organism / "hocomoco"
motifs_folder.mkdir(parents=True, exist_ok=True)

# %%
pwms = chd.data.motifscan.read_pwms(motifs_folder / "pwms.txt")

# %%
motifs = pd.DataFrame({"motif": pwms.keys()}).set_index("motif")
motif_cutoffs = pd.read_table(
    motifs_folder / "pwm_cutoffs.txt",
    names=["motif", "cutoff_001", "cutoff_0005", "cutoff_0001"],
    skiprows=1,
).set_index("motif")
motifs = motifs.join(motif_cutoffs)
annot = (
    pd.read_table(motifs_folder / "annot.txt")
    .rename(columns={"Model": "motif", "Transcription factor": "gene_label"})
    .set_index("motif")
)
motifs = motifs.join(annot)

# %%
import genomepy

# genomepy.install_genome("GRCh38", genomes_dir="/data/genome/")

fasta_file = "/data/genome/GRCh38/GRCh38.fa"

# %%
regions_name = "100k100k"
regions = chd.data.Regions(chd.get_output() / "datasets" / dataset_name / "regions" / regions_name)

# %%
motifscan_name = "hocomoco_0001"
motifscan = chd.data.Motifscan.from_pwms(
    pwms,
    regions,
    motifs=motifs,
    cutoff_col="cutoff_0001",
    fasta_file=fasta_file,
    path=chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
)

# %%
motifscan.create_region_indptr()

# %%
plt.hist(motifscan.coordinates[:])

# %%
