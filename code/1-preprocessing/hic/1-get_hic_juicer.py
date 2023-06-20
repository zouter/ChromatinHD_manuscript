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
# # Get Hi-C data, but using hicstraw starting from a .hic file
# Note that this a much faster solution than using cooler

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

import hicstraw

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"

promoter_name, window = "100k100k", np.array([-100000, 100000])
# promoter_name, promoter = "10k10k", np.array([-10000, 10000])

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)

# cool_name = "rao_2014_1kb"
# step = 1000

# cool_name = "gu_2021_500bp"
# step = 500

cool_name = "harris_2023_500bp"
step = 500

# cool_name = "matrix_1kb"
# step = 1000

if cool_name == "harris_2023_500bp":
    c = hicstraw.HiCFile(str(chd.get_output() / "ENCFF555ISR.hic"))

hic_file = folder_data_preproc / "hic" / promoter_name / f"{cool_name}.pkl"
hic_file.parent.mkdir(exist_ok=True, parents=True)

# %%
# load or create gene hics
import pickle
import cooler

if (
    True or
    (not hic_file.exists())
):
    gene_hics = {}
    for gene in tqdm.tqdm(promoters.index):
        promoter = promoters.loc[gene]

        if cool_name == "gu_2021_500bp":
            promoter = promoter.copy()
            promoter.chr = promoter.chr[3:]

        balance = "weight" if cool_name == "matrix_1kb" else "VC_SQRT"

        try:
            mzd = c.getMatrixZoomData(promoter["chr"], promoter["chr"], "observed", "VC_SQRT", "BP", 500)
            numpy_matrix = mzd.getRecordsAsMatrix(promoter["start"], promoter["end"], promoter["start"], promoter["end"])

            bins_hic = pd.DataFrame(
                {
                    "window": np.arange(window[0], window[1]+1, step),
                    "start": np.arange(window[0], window[1]+1, step),
                    "end": np.arange(window[0]+step, window[1]+step+1, step),
                }
            ).set_index("window")

            hic = pd.DataFrame(numpy_matrix, index=bins_hic.index, columns=bins_hic.index).unstack().to_frame("balanced")
            hic.index.names = ["window1", "window2"]

            gene_hics[gene] = (hic, bins_hic)
        except ValueError:
            print(f"Could not extract Hi-C for {gene}")
            continue

    pickle.dump(gene_hics, open(hic_file, "wb"))

# %%
promoter = promoters.loc["ENSG00000171791"]

hic = gene_hics[promoter.name][0]
hic["distance"] = np.abs(
    hic.index.get_level_values("window1") - hic.index.get_level_values("window2")
)
plt.matshow(hic.query("distance > 500")["balanced"].unstack())

# %%
!rsync -a {hic_file} wsaelens@updeplasrv7.epfl.ch:{hic_file} -v