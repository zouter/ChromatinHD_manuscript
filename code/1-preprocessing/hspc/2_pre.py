# %%
import pathlib
import subprocess
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import multivelo as mv
import chromatinhd as chd
import chromatinhd.data

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
sc._settings.ScanpyConfig.figdir = pathlib.PosixPath('')

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hspc"
organism = 'hs'

folder_data_preproc = folder_data / dataset_name

folder_plots = folder_data_preproc / 'plots'
folder_plots.mkdir(exist_ok = True, parents = True)

if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %%
### External Data
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)

info_genes_cells = pd.read_csv(folder_data_preproc / "info_genes_cells.csv")
s_genes = info_genes_cells['s_genes'].dropna().tolist()
g2m_genes = info_genes_cells['g2m_genes'].dropna().tolist()
hspc_marker_genes = info_genes_cells['hspc_marker_genes'].dropna().tolist()
lin_myeloid = info_genes_cells['lin_myeloid'].dropna().tolist()
lin_erythroid = info_genes_cells['lin_erythroid'].dropna().tolist()
lin_platelet = info_genes_cells['lin_platelet'].dropna().tolist()

#%%
adata_rna = sc.read_loom(folder_data_preproc / "GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom")

#%%
# check which layer is used
m1 = adata_rna.X.todense()
matrix = adata_rna.layers['matrix'].todense()
ambiguous = adata_rna.layers['ambiguous'].todense()
unspliced = adata_rna.layers['unspliced'].todense()
spliced = adata_rna.layers['spliced'].todense()

matrix_check = (m1 == spliced) - 1
print(matrix_check.sum(), '(must be 0 for correct layer)')

# adata_rna.X = adata_rna.layers['matrix']
# m1 = adata_rna.X.todense()

# matrix_check = (m1 == matrix) - 1
# print(matrix_check.sum(), '(must be 0)')

matrix_rep = ambiguous + unspliced + spliced
matrix_test = (matrix_rep == matrix) - 1
print(matrix_test.sum(), '(must be 0)')

matrix_sum = matrix.sum()
print(spliced.sum() / matrix_sum, 'spliced')
print(unspliced.sum() / matrix_sum, 'unspliced')
print(ambiguous.sum() / matrix_sum, 'ambiguous')

#%%
adata_atac = sc.read_10x_mtx(folder_data_preproc / "MV2", var_names='gene_symbols', cache=True, gex_only=False)
adata_atac = adata_atac[:,adata_atac.var['feature_types'] == "Peaks"]

#%%
adata_atac = mv.aggregate_peaks_10x(
    adata_atac, 
    folder_data_preproc / 'GSM6403411_3423-MV-2_atac_peak_annotation.tsv', 
    folder_data_preproc / 'GSE209878_3423-MV-2_feature_linkage.bedpe', 
    verbose=True
)

#%%
plt.hist(adata_atac.X.sum(1), bins=100, range=(0, 200000))
# %%
sc.pp.filter_cells(adata_atac, min_counts=5000)
sc.pp.filter_cells(adata_atac, max_counts=100000)

#%%
all_gene_ids = sorted(list(set(genes.loc[genes["chr"].isin(chromosomes)]['symbol']) & set(adata_rna.var.index)))

# %%
adata_rna.obs_names = [x.split(':')[1][:-1] + '-1' for x in adata_rna.obs_names]
adata_rna.var_names_make_unique()
adata_rna = adata_rna[:, all_gene_ids]
sc.pp.filter_cells(adata_rna, min_counts = 2000)
sc.pp.filter_cells(adata_rna, max_counts = 17500)
sc.pp.calculate_qc_metrics(adata_rna, percent_top=None, log1p=False, inplace=True)
adata_rna = adata_rna[(adata_rna.obs['n_genes_by_counts'] < 5000) & (adata_rna.obs['n_genes_by_counts'] > 1000),:]

#%%
# filtering, normalize_per_cell, log1p all included in function
scv.pp.filter_and_normalize(adata_rna, min_shared_counts=10, n_top_genes=1000)
# scv.pp.filter_genes(adata_rna, min_shared_counts=10)
# scv.pp.normalize_per_cell(adata_rna)
# scv.pp.log1p(adata_rna)
# adata_rna.raw = adata_rna
# scv.pp.filter_genes_dispersion(adata_rna, n_top_genes=1000)

#%%
shared_cells = pd.Index(np.intersect1d(adata_rna.obs_names, adata_atac.obs_names))
shared_genes = pd.Index(np.intersect1d(adata_rna.var_names, adata_atac.var_names))

#%%
adata_rna2 = adata_rna[shared_cells, shared_genes]
adata_atac2 = adata_atac[shared_cells, shared_genes]

s_genes_sub = adata_rna2.var[adata_rna2.var.index.isin(s_genes)].index
g2m_genes_sub = adata_rna2.var[adata_rna2.var.index.isin(g2m_genes)].index
#%%
# function includes PCA and neighbors
scv.pp.moments(adata_rna2, n_pcs=30, n_neighbors=50)
# sc.pp.pca(adata_rna2, n_comps=30)
# sc.pp.neighbors(adata_rna2, n_neighbors=50)
sc.tl.leiden(adata_rna2, resolution = 1)
sc.tl.umap(adata_rna2, n_components=2)

#%%
plt.ioff()

sc.pl.umap(adata_rna2, color="leiden", legend_loc="on data", show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic.pdf')

sc.pl.umap(adata_rna2, color=hspc_marker_genes, title=hspc_marker_genes, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_hspc_genes.pdf')

sc.pl.umap(adata_rna2, color=s_genes_sub, title=s_genes_sub, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_s_genes.pdf')

sc.pl.umap(adata_rna2, color=g2m_genes_sub, title=g2m_genes_sub, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_g2m_genes.pdf')

#%%
scv.tl.score_genes_cell_cycle(adata_rna2, s_genes=s_genes_sub, g2m_genes=g2m_genes_sub)

# %%
sc.pp.regress_out(adata_rna2, keys=['S_score', 'G2M_score'], n_jobs=4)

#%%
sc.pl.umap(adata_rna2, color=hspc_marker_genes, title=hspc_marker_genes, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_hspc_genes_ro.pdf')

sc.pl.umap(adata_rna2, color=s_genes_sub, title=s_genes_sub, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_s_genes_ro.pdf')

sc.pl.umap(adata_rna2, color=g2m_genes_sub, title=g2m_genes_sub, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_g2m_genes_ro.pdf')

#%%
del adata_rna2.uns
del adata_rna2.obsm
del adata_rna2.obsp
del adata_rna2.varm

#%%
scv.pp.moments(adata_rna2, n_pcs=30, n_neighbors=50)
# sc.pp.pca(adata_rna2, n_comps=30)
# sc.pp.neighbors(adata_rna2, n_neighbors=50)
sc.tl.leiden(adata_rna2, resolution = 1)
sc.tl.umap(adata_rna2, n_components=2)

#%%
sc.pl.umap(adata_rna2, color="leiden", legend_loc="on data", show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro.pdf')

sc.pl.umap(adata_rna2, color=hspc_marker_genes, title=hspc_marker_genes, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_hspc_genes.pdf')

sc.pl.umap(adata_rna2, color=s_genes_sub, title=s_genes_sub, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_s_genes.pdf')

sc.pl.umap(adata_rna2, color=g2m_genes_sub, title=g2m_genes_sub, show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_g2m_genes.pdf')

# %%
# sc.pl.umap(adata_rna2, color=['CD34', 'ATXN1'], title=['CD34', 'ATXN1'], use_raw=True)
#%%
# annotation = {
#     '3': 'LMPP', #
#     '2': 'HSC', #
#     '8': 'MEP', # 
#     '1': 'MPP', #
#     '0': 'Erythrocyte', #
#     '4': 'GMP', # 
#     '5': 'Prog MK', #
#     '6': 'Granulocyte', #
#     '9': 'Prog DC', #
#     '7': 'Prog B', # 
# }
annotation = {
    '0': 'LMPP', 
    '3': 'HSC', 
    '6': 'MEP', 
    '2': 'MPP', 
    '1': 'Erythrocyte', 
    '5': 'GMP', 
    '4': 'Prog MK', 
    '7': 'Granulocyte', 
    '9': 'Prog DC', 
    '8': 'Prog B', 
}
df_annotation = pd.DataFrame({'leiden': list(annotation.keys()), 'celltype': list(annotation.values())}).set_index("celltype")
df_annotation = df_annotation["leiden"].str.split(",").explode().to_frame().reset_index().set_index("leiden")["celltype"]
df_annotation = df_annotation.reindex(adata_rna2.obs["leiden"].unique())

adata_rna2.obs["celltype"] = df_annotation.loc[adata_rna2.obs["leiden"]].values

sc.pl.umap(adata_rna2, color="celltype", legend_loc="on data", show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_celltypes.pdf')

#%%
adata_rna2.obs_names.to_frame().to_csv(folder_data_preproc / 'filtered_cells.txt', header=False, index=False)
#%%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")
transcriptome.adata = adata_rna2
transcriptome.var = adata_rna2.var
transcriptome.obs = adata_rna2.obs
transcriptome.create_X()

#%%
# run seurat script
subprocess.call (folder_root.parent / "code/1-preprocessing/hspc/3_seurat_wnn.R")

#%%
nn_idx = np.loadtxt(folder_data_preproc / "nn_idx.txt", delimiter=',')
nn_dist = np.loadtxt(folder_data_preproc / "nn_dist.txt", delimiter=',')
nn_cells = pd.Index(pd.read_csv(folder_data_preproc / "nn_cells.txt", header=None)[0])

#%%
np.all(nn_cells == adata_atac2.obs_names)

mv.knn_smooth_chrom(adata_atac2, nn_idx, nn_dist)

adata_atac2

adata_result = mv.recover_dynamics_chrom(
    adata_rna2,
    adata_atac2,
    max_iter=5,
    init_mode="invert",
    verbose=False,
    parallel=True,
    save_plot=False,
    rna_only=False,
    fit=True,
    n_anchors=500,
    extra_color_key='celltype'
)

adata_result.write(folder_data_preproc / "multivelo_result.h5ad")

#%%