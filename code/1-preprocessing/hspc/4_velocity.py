# %%
import pathlib
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import multivelo as mv
import chromatinhd as chd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('ticks')
scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
sc._settings.ScanpyConfig.figdir = pathlib.PosixPath('')

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
folder_data_preproc = folder_data / dataset_name

#%%
info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
s_genes = info_genes_cells['s_genes'].dropna().tolist()
g2m_genes = info_genes_cells['g2m_genes'].dropna().tolist()
hspc_marker_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()
lin_myeloid = info_genes_cells['lin_myeloid'].dropna().tolist()
lin_erythroid = info_genes_cells['lin_erythroid'].dropna().tolist()
lin_platelet = info_genes_cells['lin_platelet'].dropna().tolist()

adata_result = sc.read_h5ad(folder_data_preproc / "multivelo_result.h5ad")

#%%
# mv.pie_summary(adata_result)
# mv.switch_time_summary(adata_result)
# mv.likelihood_plot(adata_result)

#%%
mv.velocity_graph(adata_result)
mv.latent_time(adata_result)
mv.velocity_embedding_stream(adata_result, basis='umap', color='celltype', save=str(folder_data_preproc / "plots/UMAP_MV2_velo.png"))
scv.pl.scatter(adata_result, color='latent_time', color_map='gnuplot', size=80, save=str(folder_data_preproc / "plots/UMAP_MV2_latent.pdf"))
adata_result.obs['latent_time'].to_csv(folder_data_preproc / "MV2_latent_time.csv")

#%%
adata_result_lin = adata_result[adata_result.obs['celltype'].isin(lin_myeloid), ]
mv.velocity_graph(adata_result_lin)
mv.latent_time(adata_result_lin)
mv.velocity_embedding_stream(adata_result_lin, basis='umap', color='celltype', save=str(folder_data_preproc / "plots/UMAP_MV2_velo_myeloid.png"))
scv.pl.scatter(adata_result_lin, color='latent_time', color_map='gnuplot', size=80, save=str(folder_data_preproc / "plots/UMAP_MV2_latent_myeloid.png"))
adata_result_lin.obs['latent_time'].to_csv(folder_data_preproc / "MV2_latent_time_myeloid.csv")

#%%
adata_result_lin = adata_result[adata_result.obs['celltype'].isin(lin_erythroid), ]
mv.velocity_graph(adata_result_lin)
mv.latent_time(adata_result_lin)
mv.velocity_embedding_stream(adata_result_lin, basis='umap', color='celltype', save=str(folder_data_preproc / "plots/UMAP_MV2_velo_erythroid.png"))
scv.pl.scatter(adata_result_lin, color='latent_time', color_map='gnuplot', size=80, save=str(folder_data_preproc / "plots/UMAP_MV2_latent_erythroid.png"))
adata_result_lin.obs['latent_time'].to_csv(folder_data_preproc / "MV2_latent_time_erythroid.csv")

#%%
adata_result_lin = adata_result[adata_result.obs['celltype'].isin(lin_platelet), ]
mv.velocity_graph(adata_result_lin)
mv.latent_time(adata_result_lin)
mv.velocity_embedding_stream(adata_result_lin, basis='umap', color='celltype', save=str(folder_data_preproc / "plots/UMAP_MV2_velo_platelet.png"))
scv.pl.scatter(adata_result_lin, color='latent_time', color_map='gnuplot', size=80, save=str(folder_data_preproc / "plots/UMAP_MV2_latent_platelet.png"))
adata_result_lin.obs['latent_time'].to_csv(folder_data_preproc / "MV2_latent_time_platelet.csv")

#%%
obs_sorted = adata_result_lin.obs.sort_values('latent_time')

adata_result_lin = adata_result_lin[obs_sorted.index, :]

cell_max = adata_result_lin.X.argmax(axis=0)

adata_result_lin.var['order'] = list(cell_max)
var_sorted = adata_result_lin.var.sort_values('order')

adata_result_lin = adata_result_lin[:, var_sorted.index]

# test = adata_result_lin.layers['matrix'].todense()
test = adata_result_lin.X.T

X2 = np.clip((test - np.quantile(test, 0.01, keepdims=True))/(np.quantile(test, 0.99, keepdims=True) - np.quantile(test, 0.01, keepdims=True)), 0, 1)

sns.heatmap(X2, cmap='YlGnBu')

# %%
