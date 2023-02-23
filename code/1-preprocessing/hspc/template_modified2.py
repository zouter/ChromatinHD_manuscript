# %%
# %load_ext autoreload
# %autoreload 2

import io
import os
import gzip
import scipy
import torch
import tabix
import pickle
import pathlib
import requests
import pybedtools
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import multivelo as mv
import tqdm.auto as tqdm
import chromatinhd as chd
import chromatinhd.data

import plotly.express as px
import matplotlib as mpl
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
genome = "GRCh38.107"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

files = {
    # 0 days, MV1
    'GSM6403408_3423-MV-1_gex_possorted_bam_0E7KE.loom.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403408/suppl/GSM6403408%5F3423%2DMV%2D1%5Fgex%5Fpossorted%5Fbam%5F0E7KE%2Eloom%2Egz',
    'GSM6403409_3423-MV-1_atac_fragments.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Ffragments%2Etsv%2Egz',
    'GSM6403409_3423-MV-1_atac_fragments.tsv.gz.tbi.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Ffragments%2Etsv%2Egz%2Etbi%2Egz',
    'GSM6403409_3423-MV-1_atac_peak_annotation.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz',
    'GSE209878_3423-MV-1_barcodes.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fbarcodes%2Etsv%2Egz',
    'GSE209878_3423-MV-1_feature_linkage.bedpe.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeature%5Flinkage%2Ebedpe%2Egz',
    'GSE209878_3423-MV-1_features.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeatures%2Etsv%2Egz',
    'GSE209878_3423-MV-1_matrix.mtx.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fmatrix%2Emtx%2Egz',
    # 7 days, MV2
    'GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403410/suppl/GSM6403410%5F3423%2DMV%2D2%5Fgex%5Fpossorted%5Fbam%5FICXFB%2Eloom%2Egz',
    'GSM6403411_3423-MV-2_atac_fragments.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Ffragments%2Etsv%2Egz',    
    'GSM6403411_3423-MV-2_atac_fragments.tsv.gz.tbi.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Ffragments%2Etsv%2Egz%2Etbi%2Egz',
    'GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz',
    'GSE209878_3423-MV-2_barcodes.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fbarcodes%2Etsv%2Egz',
    'GSE209878_3423-MV-2_feature_linkage.bedpe.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeature%5Flinkage%2Ebedpe%2Egz',
    'GSE209878_3423-MV-2_features.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeatures%2Etsv%2Egz',
    'GSE209878_3423-MV-2_matrix.mtx.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fmatrix%2Emtx%2Egz'
}

# for file in ['GSM6403411_3423-MV-2_atac_fragments.tsv.gz.tbi.gz', 'GSM6403409_3423-MV-1_atac_fragments.tsv.gz.tbi.gz']:
#     print(file)
#     r = requests.get(files[file])
#     with open(folder_data_preproc / file, 'wb') as f:
#         f.write(r.content) 
#%%
# if organism == "mm":
#     chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
# elif organism == "hs":
#     chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %% [markdown]
# Softlink relevant data
# !ln -s {folder_data_preproc}/../brain/genes.gff.gz {folder_data_preproc}/genes.gff.gz
# !ln -s {folder_data_preproc}/../brain//chromosome.sizes {folder_data_preproc}/chromosome.sizes
# !ln -s {folder_data_preproc}/../brain/genome.pkl.gz {folder_data_preproc}/genome.pkl.gz
# !ln -s {folder_data_preproc}/../brain/dna.fa.gz {folder_data_preproc}/dna.fa.gz
# !ln -s {folder_data_preproc}/../brain/genes.csv {folder_data_preproc}/genes.csv

# %%
### External Data
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)

### https://doi.org/10.1126/science.aad0501
cell_cycle_genes = pd.read_csv(folder_data_preproc / "cell_cycle_genes.csv")
s_genes = cell_cycle_genes['G1/S'].to_list()
s_genes = [x for x in s_genes if x == x]
s_genes = [x.replace(' ', '') for x in s_genes]
g2m_genes = cell_cycle_genes['G2/M'].to_list()
g2m_genes = [x.replace(' ', '') for x in g2m_genes]

#%%
### https://doi.org/10.1038/s41587-022-01476-y
hspc_marker_genes=["SPINK2", "AZU1", "MPO", "ELANE", "TUBB1", "PF4", "PPBP", "LYZ", "TCF4", "CD74", "HBB", "HBD", "KLF1", "PRG2", 'LMO4', 'EBF1']
lin_myeloid = ['HSC', 'MPP', 'LMPP', 'GMP']
lin_erythroid = ['HSC', 'MEP', 'Erythrocyte']
lin_platelet = ['HSC', 'MEP', 'Prog MK']

#%%
adata_rna = sc.read_loom(folder_data_preproc / "GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom")

#%%
adata_atac = sc.read_10x_mtx(folder_data_preproc / "MV2", var_names='gene_symbols', cache=True, gex_only=False)

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
adata_atac = adata_atac[:,adata_atac.var['feature_types'] == "Peaks"]

#%%
adata_atac = mv.aggregate_peaks_10x(
    adata_atac, 
    folder_data_preproc / 'GSM6403411_3423-MV-2_atac_peak_annotation.tsv', 
    folder_data_preproc / 'GSE209878_3423-MV-2_feature_linkage.bedpe', 
    verbose=True
)

#%%
plt.hist(adata_atac.X.sum(1), bins=100, range=(0, 200000));
# %%
sc.pp.filter_cells(adata_atac, min_counts=5000)
sc.pp.filter_cells(adata_atac, max_counts=100000)

# %%
adata_rna.obs_names = [x.split(':')[1][:-1] + '-1' for x in adata_rna.obs_names]
adata_rna.var_names_make_unique()
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
# plt.ioff()

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
# sc.pl.umap(adata_rna2, color=hspc_marker_genes, title=hspc_marker_genes, show=False)
# plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_hspc_genes_ro.pdf')

# sc.pl.umap(adata_rna2, color=s_genes_sub, title=s_genes_sub, show=False)
# plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_s_genes_ro.pdf')

# sc.pl.umap(adata_rna2, color=g2m_genes_sub, title=g2m_genes_sub, show=False)
# plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_basic_g2m_genes_ro.pdf')

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
# sc.pl.umap(adata_rna2, color="leiden", legend_loc="on data", show=False)
# plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro.pdf')

# sc.pl.umap(adata_rna2, color=hspc_marker_genes, title=hspc_marker_genes, show=False)
# plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_hspc_genes.pdf')

# sc.pl.umap(adata_rna2, color=s_genes_sub, title=s_genes_sub, show=False)
# plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_s_genes.pdf')

# sc.pl.umap(adata_rna2, color=g2m_genes_sub, title=g2m_genes_sub, show=False)
# plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_g2m_genes.pdf')

# %%
# sc.pl.umap(adata_rna2, color=['CD34', 'ATXN1'], title=['CD34', 'ATXN1'], use_raw=True)
#%%
annotation = {
    '3': 'LMPP', #
    '2': 'HSC', #
    '8': 'MEP', # 
    '1': 'MPP', #
    '0': 'Erythrocyte', #
    '4': 'GMP', # 
    '5': 'Prog MK', #
    '6': 'Granulocyte', #
    '9': 'Prog DC', #
    '7': 'Prog B', # 
}
df_annotation = pd.DataFrame({'leiden': list(annotation.keys()), 'celltype': list(annotation.values())}).set_index("celltype")
df_annotation = df_annotation["leiden"].str.split(",").explode().to_frame().reset_index().set_index("leiden")["celltype"]
df_annotation = df_annotation.reindex(adata_rna2.obs["leiden"].unique())

adata_rna2.obs["celltype"] = df_annotation.loc[adata_rna2.obs["leiden"]].values

sc.pl.umap(adata_rna2, color="celltype", legend_loc="on data", show=False)
plt.savefig(folder_data_preproc / 'plots/UMAP_MV2_ro_celltypes.pdf')

"""
story celltype annotations
"""
#%%
adata_rna2.obs_names.to_frame().to_csv(folder_data_preproc / 'filtered_cells.txt', header=False, index=False)
#%%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")
transcriptome.adata = adata_rna2
transcriptome.var = adata_rna2.var
transcriptome.obs = adata_rna2.obs
transcriptome.create_X()
### break for Seurat

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
adata_result = sc.read_h5ad(folder_data_preproc / "multivelo_result.h5ad")

#%%
# mv.pie_summary(adata_result)
# mv.switch_time_summary(adata_result)
# mv.likelihood_plot(adata_result)
mv.velocity_graph(adata_result)
mv.latent_time(adata_result)

mv.velocity_embedding_stream(adata_result, basis='umap', color='celltype')
scv.pl.scatter(adata_result, color='latent_time', color_map='gnuplot', size=80)
#%%
adata_result_lin = adata_result[adata_result.obs['celltype'].isin(lin_myeloid), ]
mv.latent_time(adata_result_lin)
mv.velocity_embedding_stream(adata_result_lin, basis='umap', color='celltype')
scv.pl.scatter(adata_result_lin, color='latent_time', color_map='gnuplot', size=80)

#%%
# adata_result_lin = adata_result[adata_result.obs['celltype'].isin(lin_erythroid), ]
# mv.latent_time(adata_result_lin)
# mv.velocity_embedding_stream(adata_result_lin, basis='umap', color='celltype')
# scv.pl.scatter(adata_result_lin, color='latent_time', color_map='gnuplot', size=80)

#%%
# adata_result_lin = adata_result[adata_result.obs['celltype'].isin(lin_platelet), ]
# mv.latent_time(adata_result_lin)
# mv.velocity_embedding_stream(adata_result_lin, basis='umap', color='celltype')
# scv.pl.scatter(adata_result_lin, color='latent_time', color_map='gnuplot', size=80)

#%%
# sort cells by latent_time
# sort genes 
#   find for each gene cell with highest expression and respective latent time
#   order genes by latent time (which genes reach highest expression first)

obs_sorted = adata_result_lin.obs.sort_values('latent_time')

adata_result_lin = adata_result_lin[obs_sorted.index, :]

cell_max = adata_result_lin.X.argmax(axis=0)

adata_result_lin.var['order'] = list(cell_max)
var_sorted = adata_result_lin.var.sort_values('order')

adata_result_lin = adata_result_lin[:, var_sorted.index]

test = adata_result_lin.layers['matrix'].todense()

X2 = np.clip((test - np.quantile(test, 0.01, keepdims=True))/(np.quantile(test, 0.99, keepdims=True) - np.quantile(test, 0.01, keepdims=True)), 0, 1)

sns.heatmap(X2.T, cmap='YlGnBu')


# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "GSM6403411_3423-MV-2_atac_fragments.tsv.gz"))

# %%
# promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)
# promoter_name, (padding_negative, padding_positive) = "1k1k", (1000, 1000)

# %%
all_gene_ids = adata_rna2.var['Accession']

all_gene_ids.index.name = "symbol"
all_gene_ids = all_gene_ids.reset_index()
all_gene_ids.index = all_gene_ids["Accession"]
all_gene_ids.index.name = "gene"

promoters = pd.DataFrame(index = all_gene_ids.index)

#%%
genes_missing = set(promoters.index).difference(set(genes.index))
genes_existing = set(promoters.index).intersection(set(genes.index))

promoters = promoters.loc[genes_existing]
# %%
promoters["tss"] = [genes_row["start"] if genes_row["strand"] == +1 else genes_row["end"] for _, genes_row in genes.loc[promoters.index].iterrows()]
promoters["strand"] = genes["strand"]
promoters["positive_strand"] = (promoters["strand"] == 1).astype(int)
promoters["negative_strand"] = (promoters["strand"] == -1).astype(int)
promoters["chr"] = genes.loc[promoters.index, "chr"]

# %%
promoters["start"] = promoters["tss"] - padding_negative * promoters["positive_strand"] - padding_positive * promoters["negative_strand"]
promoters["end"] = promoters["tss"] + padding_negative * promoters["negative_strand"] + padding_positive * promoters["positive_strand"]

# %%
promoters = promoters.drop(columns = ["positive_strand", "negative_strand"], errors = "ignore")

# %%
promoters

# %%
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))





# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
var = pd.DataFrame(index = promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
obs = transcriptome.adata.obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])

if "cell_original" in transcriptome.adata.obs.columns:
    cell_ix_to_cell = transcriptome.adata.obs["cell_original"].explode()
    cell_to_cell_ix = pd.Series(cell_ix_to_cell.index.astype(int), cell_ix_to_cell.values)
else:
    cell_to_cell_ix = obs["ix"].to_dict()

n_cells = obs.shape[0]

# %%
gene_to_fragments = [[] for i in var["ix"]]
cell_to_fragments = [[] for i in obs["ix"]]

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    gene_ix = var.loc[gene, "ix"]
    query = f"{promoter_info['chr']}:{promoter_info['start']}-{promoter_info['end']}"
    fragments_promoter = fragments_tabix.querys(query)
    
    for fragment in fragments_promoter:
        cell = fragment[3]
        
        # only store the fragment if the cell is actually of interest
        if cell in cell_to_cell_ix:
            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([
                cell_to_cell_ix[fragment[3]],
                gene_ix
            ])

 # %%
fragments.var = var
fragments.obs = obs

# %% [markdown]
# Create fragments tensor

# %%
coordinates = torch.tensor(np.array(coordinates_raw, dtype = np.int64))
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# %% [markdown]
# Sort `coordinates` and `mapping` according to `mapping`

# %%
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

# %% [markdown]
# Check size

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

# %%
np.product(coordinates.size()) * 64 / 8 / 1024 / 1024

# %% [markdown]
# Store

# %%
fragments.mapping = mapping
fragments.coordinates = coordinates

# %% [markdown]
# Create cellxgene index pointers

# %%
fragments.create_cellxgene_indptr()

# %% [markdown]
# #### Create training folds

# %%
n_bins = 5

# %%
# train/test split
transcriptome.var.index.name = "symbol"
transcriptome.var = transcriptome.var.reset_index()
transcriptome.var.index = transcriptome.var["Accession"]
transcriptome.var.index.name = "gene"

cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

cell_bins = np.floor((np.arange(len(cells_all))/(len(cells_all)/n_bins)))

chromosome_gene_counts = transcriptome.var.groupby("Chromosome").size().sort_values(ascending = False)
chromosome_bins = np.cumsum(((np.cumsum(chromosome_gene_counts) % (chromosome_gene_counts.sum() / n_bins + 1)).diff() < 0))

gene_bins = chromosome_bins[transcriptome.var["Chromosome"]].values

#%%
n_folds = 5
folds = []
for i in range(n_folds):
    cells_train = cells_all[cell_bins != i]
    cells_validation = cells_all[cell_bins == i]

    chromosomes_train = chromosome_bins.index[~(chromosome_bins == i)]
    chromosomes_validation = chromosome_bins.index[chromosome_bins == i]
    set1 = set(transcriptome.var.index[transcriptome.var["Chromosome"].isin(chromosomes_train)])
    set2 = set(fragments.var.index)
    genes_index = set1.intersection(set2)
    genes_train = fragments.var["ix"][genes_index].values
    set1 = set(transcriptome.var.index[transcriptome.var["Chromosome"].isin(chromosomes_validation)])
    set2 = set(fragments.var.index)
    genes_index = set1.intersection(set2)
    genes_validation = fragments.var["ix"][genes_index].values
    
    folds.append({
        "cells_train":cells_train,
        "cells_validation":cells_validation,
        "genes_train":genes_train,
        "genes_validation":genes_validation
    })

pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))


# %%
sizes = []
with gzip.GzipFile(folder_data_preproc / "GSM6403411_3423-MV-2_atac_fragments.tsv.gz", "r") as fragment_file:
    i = 0
    for line in fragment_file:
        line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        split = line.split("\t")
        sizes.append(int(split[2]) - int(split[1]))
        i += 1
        if i > 1000000:
            break

# %%
sizes = np.array(sizes)

# %%
np.isnan(sizes).sum()

# %%
fig, ax = plt.subplots()
ax.hist(sizes, range = (0, 1000), bins = 100)
ax.set_xlim(0, 1000)

# %%
gamma_params = scipy.stats.gamma.fit(sizes)

# %%
dist = scipy.stats.gamma(*gamma_params)

# %%
xs = np.linspace(0, 1000)
ys = dist.pdf(xs)

# %%
fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.hist(sizes, range = (0, 1000), bins = 100, density = True)
ax.set_xlim(0, 1000)
