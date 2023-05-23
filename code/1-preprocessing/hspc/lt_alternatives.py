# %%
import os
import pathlib
import subprocess
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import unitvelo as utv

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
sc._settings.ScanpyConfig.figdir = pathlib.PosixPath('')

# %%
folder_root = pathlib.Path('/home/vifernan/projects/ChromatinHD_manuscript/output/')
folder_data = folder_root / "data"
dataset_name = "hspc"
organism = 'hs'
folder_data_preproc = folder_data / dataset_name

#%%
adata = sc.read_loom(folder_data_preproc / "GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom")

#%%
obs = pd.read_csv(folder_data_preproc / 'MV2' / 'barcodes.tsv.gz', header=None, sep="\t", compression="gzip")
celltypes = pd.read_csv(folder_data_preproc / "MV2_celltypes.csv")

obs.columns = ['barcode']
celltypes.columns = ['barcode', 'celltype']

obs = pd.merge(obs, celltypes, left_on='barcode', right_on='barcode', how='left')

# %%
adata.obs = obs.set_index('barcode')

# %%
nan_cells = adata.obs['celltype'].isna()
adata = adata[~nan_cells, :]

# %%
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
lin_erythroid = info_genes_cells['lin_erythroid'].dropna().tolist()

#%%
obs_mask = adata.obs['celltype'].isin(lin_erythroid)
subset_adata = adata[obs_mask, :]

#%%
velo_config = utv.config.Configuration()
velo_config.R2_ADJUST = True
velo_config.IROOT = None
velo_config.FIT_OPTION = '1'
velo_config.AGENES_R2 = 1

# `utv.run_model` is an integrated function for RNA velocity analysis. It includes the core velocity estimation process and a few pre-processing functions provided by scVelo (normalization, neighbor graph construction to be specific).
label = 'celltype'
exp_metrics = {}

cluster_edges = [
    ("HSC", "MEP"), 
    ("MEP", "Erythrocyte")
]

adata = utv.run_model(subset_adata, label, config_file=velo_config)
scv.pl.velocity_embedding_stream(subset_adata, color=adata.uns['label'], dpi=100, title='')

#%%
directory = './res/temp'
os.listdir(directory)

# Colors of each cluster can be changed manually if there is potential confusion on analysis.

#%%
adata = scv.read(directory + '/temp_1.h5ad')
adata.uns[f'{label}_colors'] = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']

# Evaluation metrics of UniTVelo illustrate the improved performance compare with previous method. Both CBDir and ICCoh value are higher.

#%%
scv.pp.neighbors(adata)
adata_velo = adata[:, adata.var.loc[adata.var['velocity_genes'] == True].index]
exp_metrics["model_unit"] = utv.evaluate(adata_velo, cluster_edges, label, 'velocity')