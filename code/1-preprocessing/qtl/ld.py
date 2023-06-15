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
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import tqdm.auto as tqdm

import pathlib

import polars as pl

# %% tags=[]
import chromatinhd as chd

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
folder_qtl.mkdir(exist_ok = True, parents=True)

# %% tags=[]
# !wget https://www.dropbox.com/s/j72j6uciq5zuzii/all_hg38.pgen.zst?dl=1 -O {folder_qtl}/all_hg38.pgen.zst

# %% tags=[]
# !unzstd {folder_qtl}/all_hg38.pgen.zst

# %% tags=[]
# !wget https://www.dropbox.com/s/ngbo2xm5ojw9koy/all_hg38_noannot.pvar.zst?dl=1 -O {folder_qtl}/all_hg38.pvar.zst 

# %% tags=[]
# !unzstd {folder_qtl}/all_hg38.pvar.zst

# %% tags=[]
# !wget https://www.dropbox.com/s/2e87z6nc4qexjjm/hg38_corrected.psam?dl=1 -O {folder_qtl}/all_hg38.psam

# %% tags=[]
# !plink --file {folder_qtl}/all_hg38 --maf 0.05 --make-bed --out binary_fileset

# %% tags=[]
# !ls {folder_qtl}

# %% tags=[]
# !plink2 --pgen {folder_qtl}/all_hg38.pgen --pvar {folder_qtl}/all_hg38.pvar --psam {folder_qtl}/all_hg38.psam --make-bed --out {folder_qtl}/all_hg38 --allow-extra-chr --max-alleles 2

# %% tags=[]
# !plink --bfile {folder_qtl}/all_hg38 --ld rs2840528 rs7545940 --allow-extra-chr

# %% tags=[]
# !ls /data/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/qtl/hs/gwas/

# %% tags=[]
# !plink --bfile {folder_qtl}/all_hg38 --recode tab --out {folder_qtl}/all_hg38 --allow-extra-chr

# %% tags=[] language="bash"
# plink --file /data/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/qtl/hs/gwas/all_hg38  \
#       --r2  \
#       --ld-snp rs12345  \
#       --ld-window-kb 1000  \
#       --ld-window 99999  \
#       --ld-window-r2 0 \
#       --allow-extra-chr \

# %%
