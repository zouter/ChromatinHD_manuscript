#%%
import os
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import multivelo as mv
import matplotlib.pyplot as plt
import chromatinhd as chd

scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')

folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "e18brain"
genome = "GRCh38.107"
organism = "hs"

folder_data_preproc = folder_data / dataset_name

#%%
adata_atac = sc.read_10x_mtx(folder_data_preproc, prefix='GSE209878_3423-MV-2_', var_names='gene_symbols', cache=True, gex_only=False)
adata_atac = adata_atac[:,adata_atac.var['feature_types'] == "Peaks"]

#%%
adata_atac = mv.aggregate_peaks_10x(
    adata_atac, 
    folder_data_preproc / 'GSM6403411_3423-MV-2_atac_peak_annotation.tsv', 
    folder_data_preproc / 'GSE209878_3423-MV-2_feature_linkage.bedpe', 
    verbose=True
)

#%%
plt.hist(adata_atac.X.sum(1), bins=100, range=(0, 100000));
# %%
sc.pp.filter_cells(adata_atac, min_counts=5000)
sc.pp.filter_cells(adata_atac, max_counts=100000)

# %%
shared_cells = pd.Index(np.intersect1d(adata.obs_names, adata_atac.obs_names))
shared_genes = pd.Index(np.intersect1d(adata.var_names, adata_atac.var_names))