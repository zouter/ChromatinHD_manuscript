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
# mv1 = sc.read_loom(folder_data_preproc / "GSM6403408_3423-MV-1_gex_possorted_bam_0E7KE.loom")
mv2 = sc.read_loom(folder_data_preproc / "GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom")
adata = mv2.copy()

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
#%%
# matrix = adata.layers['matrix'].todense()
# ambiguous = adata.layers['ambiguous'].todense()
# unspliced = adata.layers['unspliced'].todense()
# spliced = adata.layers['spliced'].todense()

# matrix_rep = ambiguous + unspliced + spliced
# matrix_test = (matrix_rep == matrix) - 1
# print(matrix_test.sum(), '(must be 0)')

# print(spliced.sum() / matrix.sum(), 'spliced')
# print(unspliced.sum() / matrix.sum(), 'unspliced')
# print(ambiguous.sum() / matrix.sum(), 'ambiguous')

#%%
del adata.layers['ambiguous']
del adata.layers['unspliced']
del adata.layers['spliced']

# %%
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["Accession"]
adata.var.index.name = "gene"

#%%
adata_reg = adata.copy()

#%%
all_gene_ids = sorted(list(set(genes.loc[genes["chr"].isin(chromosomes)].index) & set(adata.var.index)))
adata = adata[:, all_gene_ids]

#%%
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts = 2000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, max_counts = 17500)
print(adata.obs.shape[0])
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
adata = adata[(adata.obs['n_genes_by_counts'] < 5000) & (adata.obs['n_genes_by_counts'] > 1000),:]
print(adata.obs.shape[0])

# %%
# sc.external.pp.scrublet(adata)

# adata.obs["doublet_score"].plot(kind = "hist")
# adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")

# print(adata.obs.shape[0])
# adata = adata[~adata.obs["doublet"].astype(bool)]
# print(adata.obs.shape[0])

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

sc.pp.normalize_total(adata, size_factor)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.highly_variable_genes(adata)

# %%
adata.var["n_cells"] = np.array((adata.X > 0).sum(0))[0]
adata.var["chr"] = genes["chr"]

# %%
print(adata.var.shape[0])
genes_oi = adata.var.query("n_cells > 100")["dispersions_norm"].sort_values(ascending = False)[:5000].index.tolist()
adata = adata[:, genes_oi]
print(adata.var.shape[0])
all_gene_ids = adata.var.index

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata, n_components=2)
sc.tl.leiden(adata, resolution = 1)
sc.pl.umap(adata, color="leiden", legend_loc="on data")

#%%
sc.tl.umap(adata, n_components=3)
df = pd.DataFrame(adata.obsm['X_umap'], columns=['d1','d2', 'd3'])
df['leiden'] = adata.obs['leiden'].reset_index()['leiden']
fig = px.scatter_3d(df, x='d1', y='d2', z='d3', color='leiden')
fig.update_traces(marker={'size': 1})
fig.update_layout(template='plotly_white')
fig.show()
#%%
sc.pp.filter_cells(adata_reg, min_genes=200)
sc.pp.filter_genes(adata_reg, min_cells=3)
sc.pp.normalize_per_cell(adata_reg, counts_per_cell_after=1e4)
sc.pp.log1p(adata_reg)
sc.pp.scale(adata_reg)
# csv derived from Tirosh et al table s5
# scoring is based on subset of genes that survived filtering
cell_cycle_genes = pd.read_csv(folder_data_preproc / "cell_cycle_genes.csv")
s_genes = cell_cycle_genes['G1/S'].to_list()
s_genes = [x for x in s_genes if x == x]
s_genes = [x.replace(' ', '') for x in s_genes]
s_genes = adata_reg.var[adata_reg.var.symbol.isin(s_genes)].index
g2m_genes = cell_cycle_genes['G2/M'].to_list()
g2m_genes = [x.replace(' ', '') for x in g2m_genes]
g2m_genes = adata_reg.var[adata_reg.var.symbol.isin(g2m_genes)].index

scv.tl.score_genes_cell_cycle(adata_reg, s_genes=s_genes, g2m_genes=g2m_genes)

#%%
sc.pp.regress_out(adata_reg, keys=['S_score', 'G2M_score'], n_jobs=4)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution = 1)
sc.pl.umap(adata, color="leiden", legend_loc="on data")

#%%
marker_genes=["SPINK2", "AZU1", "MPO", "ELANE", "TUBB1", "PF4", "PPBP", "LYZ", "TCF4", "CD74", "HBB", "HBD", "KLF1", "PRG2", 'MKI67', 'CDK1']
marker_genes = list(set(marker_genes).intersection(set(adata.var.symbol)))
marker_genes_s = adata.var[adata.var.symbol.isin(marker_genes)]
sc.pl.umap(adata, color=marker_genes_s.index, title=marker_genes_s['symbol'])

#%%
annotation = {
    '1': 'LMPP',
    '2': 'HSC',
    '3': 'MEP',
    '4': 'MPP',
    '5': 'Erythrocyte',
    '6': 'GMP',
    '7': 'Prog MK',
    '8': 'Granulocyte',
    '9': 'Prog DC',
    '10': 'Prog B',
    '11': 'Platelet'
}
df_annotation = pd.DataFrame({'leiden': list(annotation.keys()), 'celltype': list(annotation.values())}).set_index("celltype")
df_annotation = df_annotation["leiden"].str.split(",").explode().to_frame().reset_index().set_index("leiden")["celltype"]
df_annotation = df_annotation.reindex(adata.obs["leiden"].unique())

adata.obs["celltype"] = df_annotation.loc[adata.obs["leiden"]].values
sc.pl.umap(adata, color="celltype", legend_loc="on data")

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")
transcriptome.adata = adata
transcriptome.var = adata.var
transcriptome.obs = adata.obs
transcriptome.create_X()

# %%
fig, ax = plt.subplots()
sns.scatterplot(adata.var, x = "means", y = "dispersions_norm")
ax.set_xscale("log")

# %%
# adata.var['mt'] = adata.var["symbol"].str.startswith('MT')  # annotate the group of mitochondrial genes as 'mt'
# sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# %% [markdown]
# ### Interpret E18 brain

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")


# %%
sc.pl.rank_genes_groups_matrixplot(adata, ["5"], gene_symbols = "symbol")

# %% [markdown]
# ### Interpret Lymphoma

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")

# %%
sc.pl.rank_genes_groups_matrixplot(adata, ["22"], gene_symbols = "symbol")

# %%
transcriptome.obs = adata.obs
transcriptome.adata = adata

# %% [markdown]
# ### Interpret PBMC10K

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")

# %%
import io

# %%
marker_annotation = pd.read_table(io.StringIO("""ix	symbols	celltype
0	IL7R, CD3D	CD4 naive T
0	IL7R, CD3D, ITGB1	CD4 memory T
1	CD14, LYZ	CD14+ Monocytes
2	MS4A1, IL4R, CD79A	naive B
2	MS4A1, CD79A, TNFRSF13B	memory B
3	CD8A, CD3D	CD8 naive T
4	GNLY, NKG7, GZMA, GZMB, NCAM1	NK
4	GNLY, NKG7, CD3D, CCL5, GZMA, CD8A	CD8 activated T
4	SLC4A10	MAIT
5	FCGR3A, MS4A7	FCGR3A+ Monocytes
5	CD27, JCHAIN	Plasma
6	TCF4	pDCs
6	FCER1A, CST3	cDCs
""")).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")
# marker_annotation = marker_annotation.explode("symbols")

# %%
marker_annotation = pd.read_table(io.StringIO("""ix	symbols	celltype
0	IL7R, CD3D	CD4 naive T
0	IL7R, CD3D, ITGB1	CD4 memory T
1	CD14, LYZ	CD14+ Monocytes
2	MS4A1, IL4R, CD79A	naive B
2	MS4A1, CD79A, TNFRSF13B	memory B
3	CD8A, CD3D	CD8 naive T
4	GNLY, NKG7, GZMA, GZMB, NCAM1	NK
4	GNLY, NKG7, CD3D, CCL5, GZMA, CD8A	CD8 activated T
4	SLC4A10	MAIT
5	FCGR3A, MS4A7	FCGR3A+ Monocytes
5	CD27, JCHAIN	Plasma
6	TCF4	pDCs
6	CST3	cDCs
""")).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")
# marker_annotation = marker_annotation.explode("symbols")

# %%
sc.pl.umap(
    adata,
    color = transcriptome.gene_id(marker_annotation.query("celltype == 'pDCs'")["symbols"].explode())
)


# %%
#Define cluster score for all markers
def evaluate_partition(anndata, marker_dict, gene_symbol_key=None, partition_key='louvain_r1'):
    # Inputs:
    #    anndata         - An AnnData object containing the data set and a partition
    #    marker_dict     - A dictionary with cell-type markers. The markers should be stores as anndata.var_names or 
    #                      an anndata.var field with the key given by the gene_symbol_key input
    #    gene_symbol_key - The key for the anndata.var field with gene IDs or names that correspond to the marker 
    #                      genes
    #    partition_key   - The key for the anndata.obs field where the cluster IDs are stored. The default is
    #                      'louvain_r1' 

    #Test inputs
    if partition_key not in anndata.obs.columns.values:
        print('KeyError: The partition key was not found in the passed AnnData object.')
        print('   Have you done the clustering? If so, please tell pass the cluster IDs with the AnnData object!')
        raise

    if (gene_symbol_key != None) and (gene_symbol_key not in anndata.var.columns.values):
        print('KeyError: The provided gene symbol key was not found in the passed AnnData object.')
        print('   Check that your cell type markers are given in a format that your anndata object knows!')
        raise
        

    if gene_symbol_key:
        gene_ids = anndata.var[gene_symbol_key]
    else:
        gene_ids = anndata.var_names

    clusters = np.unique(anndata.obs[partition_key])
    n_clust = len(clusters)
    n_groups = len(marker_dict)
    
    marker_res = np.zeros((n_groups, n_clust))
    z_scores = sc.pp.scale(anndata, copy=True)

    i = 0
    for group in marker_dict:
        # Find the corresponding columns and get their mean expression in the cluster
        j = 0
        for clust in clusters:
            cluster_cells = np.in1d(z_scores.obs[partition_key], clust)
            marker_genes = np.in1d(gene_ids, marker_dict[group])
            marker_res[i,j] = z_scores.X[np.ix_(cluster_cells,marker_genes)].mean()
            j += 1
        i+=1

    variances = np.nanvar(marker_res, axis=0)
    if np.all(np.isnan(variances)):
        print("No variances could be computed, check if your cell markers are in the data set.")
        print("Maybe the cell marker IDs do not correspond to your gene_symbol_key input or the var_names")
        raise

    marker_res_df = pd.DataFrame(marker_res, columns=clusters, index=marker_dict.keys())

    #Return the median of the variances over the clusters
    return(marker_res_df)

# %%
cluster_celltypes = evaluate_partition(adata, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden").idxmax()

# %%
adata.obs["celltype"] = cluster_celltypes[adata.obs["leiden"]].values
adata.obs["celltype"] = adata.obs["celltype"].astype(str)
# adata.obs.loc[adata.obs["leiden"] == "4", "celltype"] = "NKT"

# %%
transcriptome.adata.obs["log_n_counts"] = np.log(transcriptome.adata.obs["n_counts"])

# %%
sc.pl.umap(
    adata,
    color = ["celltype", "log_n_counts", "leiden"]
)
sc.pl.umap(
    adata,
    color = transcriptome.gene_id(marker_annotation["symbols"].explode()),
    title = marker_annotation["symbols"].explode()
)

# %%
transcriptome.obs = adata.obs
transcriptome.adata = adata

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Creating promoters

# %%
# !zcat {folder_data_preproc}/atac_fragments.tsv.gz | head -n 100

# %%
import tabix

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "atac_fragments.tsv.gz"))

# %% [markdown]
# #### Define promoters

# %%
promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)
# promoter_name, (padding_negative, padding_positive) = "1k1k", (1000, 1000)

# %%
import pybedtools

# %%
all_gene_ids = transcriptome.var.index

# %%
promoters = pd.DataFrame(index = all_gene_ids)

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

# %% [markdown]
# #### Create fragments

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = chromatinhd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
import pathlib
import chromatinhd.data
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
    fragments_promoter = fragments_tabix.query(*promoter_info[["chr", "start", "end"]])
    
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
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

cell_bins = np.floor((np.arange(len(cells_all))/(len(cells_all)/n_bins)))

chromosome_gene_counts = transcriptome.var.groupby("chr").size().sort_values(ascending = False)
chromosome_bins = np.cumsum(((np.cumsum(chromosome_gene_counts) % (chromosome_gene_counts.sum() / n_bins + 1)).diff() < 0))

gene_bins = chromosome_bins[transcriptome.var["chr"]].values

n_folds = 5
folds = []
for i in range(n_folds):
    cells_train = cells_all[cell_bins != i]
    cells_validation = cells_all[cell_bins == i]

    chromosomes_train = chromosome_bins.index[~(chromosome_bins == i)]
    chromosomes_validation = chromosome_bins.index[chromosome_bins == i]
    genes_train = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_train)]].values
    genes_validation = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_validation)]].values
    
    folds.append({
        "cells_train":cells_train,
        "cells_validation":cells_validation,
        "genes_train":genes_train,
        "genes_validation":genes_validation
    })
pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))

# %% [markdown]
# ## Fragment distribution

# %%
import gzip

# %%
sizes = []
with gzip.GzipFile(folder_data_preproc / "atac_fragments.tsv.gz", "r") as fragment_file:
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
import scipy

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
