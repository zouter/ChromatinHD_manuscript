# %%
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().magic('load_ext autoreload')
    IPython.get_ipython().magic('autoreload 2')

import sys
import torch
import pickle
import subprocess
import numpy as np
import pandas as pd
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data_preproc = folder_root / "data" / "hspc"
dataset_name_sub = "MV2"

promoter_name, window = "10k10k", np.array([-10000, 10000])
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
annotation_original = pickle.load(open(folder_data_preproc / f"{dataset_name_sub}_celltype_colors.pkl", "rb"))

transcriptome = chd.data.Transcriptome(folder_data_preproc / f"{dataset_name_sub}_transcriptome")
adata = transcriptome.adata

bins = np.linspace(0, 1, 500)
binmids = (bins[1:] + bins[:-1])/2
binsize = binmids[1] - binmids[0]
pseudocoordinates = torch.linspace(0, 1, 1000)

#%%
lineages = ['lin_myeloid', 'lin_erythroid', 'lin_platelet']
transcriptome.var.index
combinations = [(key, value) for key in lineages for value in transcriptome.var.index]

for arg in combinations:
    subprocess.call([sys.executable, 'figure3diff.py', str(arg), 'from_fig_runner'])

# %%
