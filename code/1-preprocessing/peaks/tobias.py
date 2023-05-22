# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
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

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm

import chromatinhd as chd
import gzip

# %%
import chromatinhd as chd

# %%
software_folder = chd.get_git_root() / "software"

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "pbmc10k_gran"
dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
# dataset_name = "brain"
# dataset_name = "morf_20"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
software_folder = chd.get_git_root() / "software"

# %% [markdown]
# ## MACS2

# %% tags=[]
# loading
latent_name = "leiden_1"
latent_name = "leiden_0.1"
# latent_name = "celltype"
# latent_name = "overexpression"
latent_folder = folder_data_preproc / "latent"

# %% tags=[]
latent_folder.mkdir()

# %%
# !rsync -a wsaelens@updeplasrv7.epfl.ch:{latent_folder}/ {latent_folder}/ -v

# %% tags=[]
peaks_folder = folder_root / "peaks" / dataset_name / "macs2_improved"

# %% tags=[]
# !rsync -a wsaelens@updeplasrv7.epfl.ch:{peaks_folder}/ {peaks_folder}/ -v

# %% tags=[]
genome_folder = chd.get_output() / "data" / "genomes" / genome

# %%
tmpdir = 

# %% tags=[]
# !ls /data/genome/homo_sapiens/GRCh38.108/Homo_sapiens.GRCh38.dna.primary_assembly.fa

# %% tags=[]
# !ln -s /data/genome/homo_sapiens/GRCh38.108/Homo_sapiens.GRCh38.dna.primary_assembly.fa {folder_data_preproc}/genome/primary_assembly.fa

# %% tags=[]
# !cat {folder_data_preproc}/genome/dna.fa | sed -r 's/(N[CTW]_[0-9]*)/chr\1/g' > {folder_data_preproc}/genome/dna_chr.fa

# %% [markdown]
# https://github.com/10XGenomics/subset-bam/releases/download/v1.1.0/subset-bam_linux

# %% tags=[]
# !TOBIAS ATACorrect -b {folder_data_preproc}/bam/atac_possorted_bam.bam -g {folder_data_preproc}/genome/dna_chr.fa -p {peaks_folder}/peaks.bed

# %%
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

# %%
peaks_folder = folder_root / "peaks" / dataset_name / ("macs2_" + latent_name)
peaks_folder.mkdir(exist_ok = True, parents = True)
