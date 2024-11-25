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

# %%
import polyptich as pp
pp.setup_ipython()

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

# %%
import peakfreeatac as pfa
import tempfile

# %%
import os

# %% [markdown]
# ## Data

# %% [markdown]
# ### Dataset

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
dataset_name = "lymphoma"
dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
fragments.create_cut_data()

# %%
torch.bincount(fragments.cut_local_cell_ix, minlength = fragments.n_cells).numpy().shape

# %% [markdown]
# ## Differential

# %%
fragments.obs["lib"] = torch.bincount(fragments.cut_local_cell_ix, minlength = fragments.n_cells).numpy()

# %%
import peakfreeatac.peakcounts

# %%
import scipy.io
import subprocess as sp

# %%
R_location = pfa.get_git_root() / "software/R-4.2.2/bin/Rscript"
Rlib_location = pfa.get_git_root() / "software/R-4.2.2/lib"

# %%
R_location

# %%
import tempfile

# %%
R -c 'BiocManager::install("edgeR")'

# %%
R -c 'BiocManager::install("csaw")'

# %%
for latent_name in [
    "leiden_0.1",
    # "leiden_1",
    # "celltype",
]:
    folder_data_preproc = folder_data / dataset_name
    latent_folder = folder_data_preproc / "latent"
    latent = torch.from_numpy(pickle.load((latent_folder / (latent_name + ".pkl")).open("rb")).values).to(torch.float)
    cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))

    n_latent_dimensions = latent.shape[-1]
    
    transcriptome.adata.obs["cluster"] = fragments.obs["cluster"] = np.where(latent.cpu() > 0)[1]
    
    for peaks_name in [
        "cellranger",
        "macs2",
        # "macs2_leiden_0.1",
        # "macs2_leiden_0.1_merged",
        # "macs2_q0.20",
        # "macs2_q0.50",
        # "genrich",
        # "rolling_500"
    ]:
        print(f"{latent_name=} {peaks_name=}")
        peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)

        tmpdir = pathlib.Path(tempfile.mkdtemp())

        n = 5000
        scipy.io.mmwrite(tmpdir / "counts", peakcounts.counts[:n])
        fragments.obs.iloc[:n].to_csv(tmpdir / "obs.csv", index = False)
        peakcounts.var.to_csv(tmpdir / "var.csv")

        output = sp.run([R_location, "run_edger.R", str(tmpdir)], capture_output = True)
        print(output.stdout[-10:])
        print(output.stderr[-100:])

        results = pd.read_csv(tmpdir / "results.csv", index_col = 0)
        results["significant"] = (results["q_value"] < 0.05) & (results["logFC"] > 1)

        results.to_csv(peakcounts.path / ("diffexp_" + latent_name + ".csv"))

# %%
results["significant"]

# %%
results["significant"] =(results["logFC"] > 1)

# %%
results.groupby("cluster")["significant"].sum()
