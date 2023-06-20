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
!wget https://www.encodeproject.org/files/ENCFF555ISR/@@download/ENCFF555ISR.hic -O {chd.get_output()}/ENCFF555ISR.hic

# %%
# ! pip install hic2cool
! pip install hicexplorer

# %%
!hic2cool convert -r 500 {chd.get_output()}/ENCFF555ISR.hic {chd.get_output()}/ENCFF555ISR.mcool
# %%
!hic2cool convert

# %%
!pip install hic-straw
# %%
import numpy as np
import hicstraw

# %%
hic = hicstraw.HiCFile(str(chd.get_output() / "ENCFF555ISR.hic"))

# %%
print(hic.getChromosomes()[0])
print(hic.getGenomeID())
print(hic.getResolutions())

# %%
mzd = hic.getMatrixZoomData('chr4', 'chr4', "observed", "VC_SQRT", "BP", 500)

# %%
numpy_matrix = mzd.getRecordsAsMatrix(0, 1000000, 0, 1000000)

# %%
import seaborn as sns
sns.heatmap(np.log1p(numpy_matrix))
# %%
