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

regions_name, window = "100k100k", np.array([-100000, 100000])
# promoter_name, promoter = "10k10k", np.array([-10000, 10000])

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

regions = chd.data.Regions(chd.get_output() / "datasets" / dataset_name / "regions" / regions_name)

cool_name = "rao_2014_1kb"
step = 1000

# cool_name = "gu_2021_500bp"
# step = 500

# cool_name = "matrix_1kb"
# step = 1000

if cool_name == "rao_2014_1kb":
    c = cooler.Cooler(str(chd.get_output() / "HiC" / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000")
elif cool_name == "gu_2021_500bp":
    c = cooler.Cooler(str(chd.get_output() / "HiC/hic_lcl_mega/LCL_mega_42B_500bp_30_cool.cool"))
elif cool_name == "matrix_1kb":
    c = cooler.Cooler(
        # str(chd.get_output() / "HiC/matrix_1kb.cool")
        str(chd.get_output() / "HiC/matrix_1kb.mcool")
        + "::/resolutions/1000"
    )

hic_file = folder_data_preproc / "hic" / regions_name / f"{cool_name}.pkl"
hic_file.parent.mkdir(exist_ok=True, parents=True)

# %%
# load or create gene hics
import pickle
import cooler

gene_hics = {}
# if not hic_file.exists():
#     gene_hics = {}
# else:
#     gene_hics = pickle.load(open(hic_file, "rb"))
for gene in tqdm.tqdm(regions.coordinates.index):
    promoter = regions.coordinates.loc[gene]

    if cool_name == "gu_2021_500bp":
        promoter = promoter.copy()
        promoter.chr = promoter.chr[3:]

    balance = "weight" if cool_name == "matrix_1kb" else "VC_SQRT"

    try:
        hic, bins_hic = chdm.hic.extract_hic(promoter, c=c, step=step, balance=balance)
        print(bins_hic.shape[0])
        gene_hics[gene] = (hic, bins_hic)
    except ValueError:
        print(f"Could not extract Hi-C for {gene}")
        continue

pickle.dump(gene_hics, open(hic_file, "wb"))

# %%
promoter = regions.coordinates.loc["ENSG00000171791"]
if cool_name == "gu_2021_500bp":
    promoter = promoter.copy()
    promoter.chr = promoter.chr[3:]
hic, bins_hic = chdm.hic.extract_hic(promoter, c=c, step=step, balance=None)

if "balanced" not in hic.columns:
    hic["balanced"] = np.log1p(hic["count"])

# %%
hic["distance"] = np.abs(hic.index.get_level_values("window1") - hic.index.get_level_values("window2"))

sns.heatmap(np.log1p(hic.query("distance > 500")["balanced"].unstack()))
# %%
hic["distance"] = np.abs(hic.index.get_level_values("window1") - hic.index.get_level_values("window2"))
bins_hic_oi = bins_hic.loc[(bins_hic.index > 60000) & (bins_hic.index < 90000)]

sns.heatmap(np.log1p(hic.query("distance > 500")["balanced"].unstack().reindex(bins_hic_oi.index, bins_hic_oi.index)))
# %%
