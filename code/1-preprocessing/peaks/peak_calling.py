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

sns.set_style("ticks")

import pickle

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
genome = "GRCh38.107"

# dataset_name = "pbmc10k_gran"
# genome = "GRCh38.107"

dataset_name = "hspc"
genome = "GRCh38.107"

# dataset_name = "hspc_gmp"
# genome = "GRCh38.107"

# dataset_name = "e18brain"; genome = "mm10"
# dataset_name = "lymphoma"; genome = "GRCh38.107"
# dataset_name = "alzheimer"; genome = "mm10"
# dataset_name = "brain"; genome = "GRCh38.107"

# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"

# dataset_name = "morf_20"; genome = "GRCh38.107"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

# %%
software_folder = chd.get_git_root() / "software"

# %% [markdown]
# ## Cell ranger
#
# Peaks were already called by cell ranger

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "cellranger"
peaks_folder.mkdir(exist_ok=True, parents=True)

# %%
# !cp {folder_data_preproc}/peaks.tsv {peaks_folder}/peaks.bed

# %% [markdown]
# ## Genrich

# %% [markdown]
# A lot of people seem to be happy with this

# %% [markdown]
# If you don't have bam file, check out this: https://github.com/jsh58/Genrich/issues/95

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "genrich"
peaks_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
#
# Documentation:
#
# https://informatics.fas.harvard.edu/atac-seq-guidelines.html#peak

# %%
# !wget https://github.com/jsh58/Genrich/archive/refs/tags/v0.6.1.zip -P {software_folder}

# %%
# install
# !echo 'cd {software_folder} && unzip v0.6.1.zip'
# !echo 'cd {software_folder}/Genrich-0.6.1 && make'

# %%
# sort the reads
# !echo 'samtools sort -@ 20 -n {folder_data_preproc}/bam/atac_possorted_bam.bam -o {folder_data_preproc}/bam/atac_readsorted_bam.bam'

# create peaks folder
# !echo 'mkdir -p {peaks_folder}'

# run genrich
# !echo '{software_folder}/Genrich-0.6.1/Genrich -t {folder_data_preproc}/bam/atac_readsorted_bam.bam -j -f {peaks_folder}/log -o {peaks_folder}/peaks.bed -v'

# %%
# (run on updeplasrv7)
# create peaks folder
# !echo 'mkdir -p {peaks_folder}'

# sync from updeplasrv6 to updeplasrv7
# !echo 'rsync wsaelens@updeplasrv6.epfl.ch:{peaks_folder}/peaks.bed {peaks_folder}/peaks.bed'

# %% [markdown]
# ## MACS2 ENCODE
#
# MACS2 with encode parameters
# https://github.com/ENCODE-DCC/atac-seq-pipeline/blob/master/src/encode_task_macs2_atac.py

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2"
peaks_folder.mkdir(exist_ok=True, parents=True)

# %%
# !echo 'ls {peaks_folder}'
# !ls {peaks_folder}

# %%
# !echo 'mkdir -p {peaks_folder}'

# %%
# !echo 'ls {folder_data_preproc}'

# %%
# if BAM is available
# # !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/bam/atac_possorted_bam.bam --nomodel --shift -100 --extsize 200'

# if BAM is not available
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/atac_fragments.tsv.gz --nomodel --shift -100 --extsize 200 && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# !echo 'ls {peaks_folder}'

# %%
# !echo 'cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# from updeplasrv7
# !echo 'mkdir -p {peaks_folder}'
# !echo 'rsync wsaelens@updeplasrv6.epfl.ch:{peaks_folder}/peaks.bed {peaks_folder}/peaks.bed -v'

# %% [markdown]
# ## MACS2 Paired-end
#
# This uses the settings recommended by the authors of MACS2
# https://github.com/macs3-project/MACS/issues/145
#
# In the end, this visually doesn't make much difference from the ENCODE pipeline

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2_improved"
peaks_folder.mkdir(exist_ok=True, parents=True)

# %%
# !echo 'ls {peaks_folder}'
# !ls {peaks_folder}

# %%
# if BAM is available
# # !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/bam/atac_possorted_bam.bam -f BAMPE --nomodel'

# if BAM is not available
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/atac_fragments.tsv.gz -f BEDPE --nomodel && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# if BAM is not available
# alternative for other datasets
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/fragments.tsv.gz -f BEDPE && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# !echo 'ls {peaks_folder}'

# %%
# !echo 'cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# from updeplasrv7
# !echo 'mkdir -p {peaks_folder}'
# !echo 'rsync wsaelens@updeplasrv6.epfl.ch:{peaks_folder}/peaks.bed {peaks_folder}/peaks.bed -v'

# %% [markdown]
# ## MACS2 paired-end with different q parameters
# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2_q0.20"
peaks_folder.mkdir(exist_ok=True, parents=True)

# %%
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/atac_fragments.tsv.gz --nomodel -f BEDPE -q 0.20 && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/fragments.tsv.gz --nomodel -f BEDPE -q 0.20 && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2_q0.50"
peaks_folder.mkdir(exist_ok=True, parents=True)

# %%
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/atac_fragments.tsv.gz --nomodel -f BEDPE -q 0.50 && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'


# %% [markdown]
# ## ENCODE Screen

# %% [markdown]
# https://screen.encodeproject.org/

# %%
assert genome == "GRCh38.107"

# %%
# !wget https://downloads.wenglab.org/V3/GRCh38-cCREs.bed -O {chd.get_output()}/GRCh38-cCREs.bed
# # !wget https://downloads.wenglab.org/Registry-V3/mm10-cCREs.bed -O {chd.get_output()}/mm10-cCREs.bed

# %%
# !head  {chd.get_output()}/GRCh38-cCREs.bed

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "encode_screen"
peaks_folder.mkdir(exist_ok=True, parents=True)

# %%
# !cp {chd.get_output()}/GRCh38-cCREs.bed {peaks_folder}/peaks.bed

# %%

# %%
