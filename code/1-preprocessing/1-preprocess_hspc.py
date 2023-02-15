# %%
# %load_ext autoreload
# %autoreload 2

import io
import os
import torch
import pickle
import requests
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

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "hspc"
genome = "GRCh38.107"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# files = {
#     # 0 days
#     'GSM6403408_3423-MV-1_gex_possorted_bam_0E7KE.loom.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403408/suppl/GSM6403408%5F3423%2DMV%2D1%5Fgex%5Fpossorted%5Fbam%5F0E7KE%2Eloom%2Egz',
#     'GSM6403409_3423-MV-1_atac_fragments.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Ffragments%2Etsv%2Egz',
#     'GSM6403409_3423-MV-1_atac_peak_annotation.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403409/suppl/GSM6403409%5F3423%2DMV%2D1%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz',
#     'GSE209878_3423-MV-1_barcodes.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fbarcodes%2Etsv%2Egz',
#     'GSE209878_3423-MV-1_feature_linkage.bedpe.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeature%5Flinkage%2Ebedpe%2Egz',
#     'GSE209878_3423-MV-1_features.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Ffeatures%2Etsv%2Egz',
#     'GSE209878_3423-MV-1_matrix.mtx.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D1%5Fmatrix%2Emtx%2Egz',
#     # 7 days
#     'GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403410/suppl/GSM6403410%5F3423%2DMV%2D2%5Fgex%5Fpossorted%5Fbam%5FICXFB%2Eloom%2Egz',
#     'GSM6403411_3423-MV-2_atac_fragments.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Ffragments%2Etsv%2Egz',    
#     'GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6403nnn/GSM6403411/suppl/GSM6403411%5F3423%2DMV%2D2%5Fatac%5Fpeak%5Fannotation%2Etsv%2Egz',
#     'GSE209878_3423-MV-2_barcodes.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fbarcodes%2Etsv%2Egz',
#     'GSE209878_3423-MV-2_feature_linkage.bedpe.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeature%5Flinkage%2Ebedpe%2Egz',
#     'GSE209878_3423-MV-2_features.tsv.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Ffeatures%2Etsv%2Egz',
#     'GSE209878_3423-MV-2_matrix.mtx.gz': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209878/suppl/GSE209878%5F3423%2DMV%2D2%5Fmatrix%2Emtx%2Egz'
# }
# 
# for file in ['GSM6403409_3423-MV-1_atac_peak_annotation.tsv.gz', 'GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz']:
#     print(file)
#     r = requests.get(files[file])
#     with open(folder_data_preproc / file, 'wb') as f:
#         f.write(r.content) 
#%%
if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %% [markdown]
# Softlink relevant data
# !ln -s {folder_data_preproc}/../brain/genes.gff.gz {folder_data_preproc}/genes.gff.gz
# !ln -s {folder_data_preproc}/../brain//chromosome.sizes {folder_data_preproc}/chromosome.sizes
# !ln -s {folder_data_preproc}/../brain/genome.pkl.gz {folder_data_preproc}/genome.pkl.gz
# !ln -s {folder_data_preproc}/../brain/dna.fa.gz {folder_data_preproc}/dna.fa.gz
# !ln -s {folder_data_preproc}/../brain/genes.csv {folder_data_preproc}/genes.csv

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)

#%%
adata_rna = sc.read_loom(folder_data_preproc / "GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom")
adata_atac = sc.read_10x_mtx(folder_data_preproc, prefix='GSE209878_3423-MV-2_', var_names='gene_symbols', cache=True, gex_only=False)

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
scv.pp.filter_and_normalize(adata_rna, min_shared_counts=10, n_top_genes=1000)

#%%
shared_cells = pd.Index(np.intersect1d(adata_rna.obs_names, adata_atac.obs_names))
shared_genes = pd.Index(np.intersect1d(adata_rna.var_names, adata_atac.var_names))
len(shared_cells), len(shared_genes)

#%%
adata_rna2 = adata_rna[shared_cells, shared_genes]
adata_atac2 = adata_atac[shared_cells, shared_genes]

#%%
scv.pp.moments(adata_rna2, n_pcs=30, n_neighbors=50)
sc.tl.leiden(adata_rna2, resolution = 1)
sc.tl.umap(adata_rna2, n_components=2)
sc.pl.umap(adata_rna2, color="leiden", legend_loc="on data")
#%%
marker_genes=["SPINK2", "AZU1", "MPO", "ELANE", "TUBB1", "PF4", "PPBP", "LYZ", "TCF4", "CD74", "HBB", "HBD", "KLF1", "PRG2", 'LMO4', 'EBF1']
sc.pl.umap(adata_rna2, color=marker_genes, title=marker_genes)

#%%
cell_cycle_genes = ['MKI67', 'CDK1']
sc.pl.umap(adata_rna2, color=cell_cycle_genes, title=cell_cycle_genes)




# %%
