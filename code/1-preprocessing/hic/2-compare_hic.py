# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Get Hi-C data

# %%
import IPython

if IPython.get_ipython() is not None:
    from IPython import get_ipython

    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm

import chromatinhd as chd
import chromatinhd_manuscript as chdm

import cooler

# %%
# ! cp ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/HiC/matrix_1kb.cool {chd.get_output()}/HiC/matrix_1kb.cool
# ! cp ~/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/HiC/matrix_1kb.mcool {chd.get_output()}/HiC/matrix_1kb.mcool

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"


promoter_name, window = "100k100k", np.array([-100000, 100000])
# promoter_name, promoter = "10k10k", np.array([-10000, 10000])

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

cool_name = "rao_2014_1kb"
step1 = 1000
c1 = cooler.Cooler(str(chd.get_output() / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000")

# cool_name = "gu_2021_500bp"
# step = 500

# cool_name = "matrix_1kb"
# step = 1000

c2 = cooler.Cooler(
    str(chd.get_output() / "HiC/hic_lcl_mega/LCL_mega_42B_500bp_30_cool.cool")
)
step2 = 500

# %%
promoter = promoters.loc[transcriptome.gene_id("BCL2")].copy()
promoter = promoters.loc[transcriptome.gene_id("CD74")].copy()
promoter = promoters.loc[transcriptome.gene_id("BANK1")].copy()
promoter = promoters.loc[transcriptome.gene_id("AFF3")].copy()
# promoter = promoters.loc["ENSG00000111728"].copy()

hic, bins_hic = chdm.hic.extract_hic(promoter, c=c1, step=step1)

promoter = promoter.copy()
promoter.chr = promoter.chr[3:]
hic2, bins_hic2 = chdm.hic.extract_hic(promoter, c=c2, step=step2)

# %%
hic["logbalanced"] = np.log(hic["balanced"])
hic2["logbalanced"] = np.log(hic2["balanced"])

# %%
hic["distance"] = np.abs(
    hic.index.get_level_values("window1") - hic.index.get_level_values("window2")
)
hic2["distance"] = np.abs(
    hic2.index.get_level_values("window1") - hic2.index.get_level_values("window2")
)

# %%
fig, ax = plt.subplots()
ax.matshow(
    hic.query("distance > 1000")["balanced"]
    .unstack()
    .reindex(index=bins_hic.index, columns=bins_hic.index)
)
fig, ax = plt.subplots()
ax.matshow(
    hic2.query("distance > 1000")["balanced"]
    .unstack()
    .reindex(index=bins_hic.index, columns=bins_hic.index)
)

# %%
plotdata = pd.DataFrame(
    {
        "hic": (
            hic.query("distance > 1000")["logbalanced"]
            .unstack()
            .reindex(index=bins_hic.index, columns=bins_hic.index)
            .values.flatten()
        ),
        "hic2": (
            hic2.query("distance > 1000")["logbalanced"]
            .unstack()
            .reindex(index=bins_hic.index, columns=bins_hic.index)
            .values.flatten()
        ),
    }
)

# %%
