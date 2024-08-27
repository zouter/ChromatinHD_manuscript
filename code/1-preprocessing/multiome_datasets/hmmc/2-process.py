# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import scanpy as sc

import chromatinhd as chd
import chromatinhd_manuscript as chdm
import gimmedata

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"
dataset_name = "hmmc"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)
genome = "GRCh38"


# %%
# adata = sc.read_10x_h5("/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/hmmc/GSE219057_Aged2_HSPC.filtered_feature_bc_matrix.h5")
# adata = sc.read_10x_h5("/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/hmmc/GSE219106_Young1_BMMC.filtered_feature_bc_matrix.h5")
# adata = sc.read_10x_h5("/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/hmmc/GSE219106_Young1_HSPC.filtered_feature_bc_matrix.h5")
# adata = sc.read_10x_h5("/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/hmmc/GSE219248_Young2_HSPC.filtered_feature_bc_matrix.h5")

# %%
def convert_gene_ids(adata):
    # Indeed, ensembl ids are proevided in the "gene_ids" column. Let's see if they are unique
    adata.var["gene_ids"].nunique() == adata.var.shape[0]

    # They are unique, we will use them as index
    adata.var["symbol"] = adata.var.index
    adata.var.index = adata.var["gene_ids"]
    adata.var.index.name = "gene"
    return adata


# %%
samples = pd.DataFrame({
    "sample": ["GSE219106_Young1_HSPC", "GSE219167_Young1_T2_HSPC", "GSE219248_Young2_HSPC"],
}).set_index("sample")

# %%
adatas = [
    convert_gene_ids(sc.read_10x_h5(f"/home/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/hmmc/{sample}.filtered_feature_bc_matrix.h5")) for sample in samples.index
]

# %%
for i, adata in enumerate(adatas):
    adata.obs.index = adata.obs.index.str.replace("-1", f"-{i+1}")

# %%
adata = sc.concat(adatas)

# fix var
var = pd.concat([adata.var for adata in adatas]).drop_duplicates(keep = "first")
adata.var = var.loc[adata.var.index]

# fix obs
adata.obs["batch"] = adata.obs.index.str.split("-").str[1].astype(int)
adata.obs["sample"] = samples.index[adata.obs["batch"]-1]

# %%
# Now we can do some basic filtering
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts=1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes=200)
print(adata.obs.shape[0])
sc.pp.filter_genes(adata, min_cells=10)
print(adata.var.shape[0])

# %%
sc.external.pp.scrublet(adata)
adata.obs["doublet_score"].plot(kind="hist")
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
genome = "GRCh38"

# %%
# transcripts = chd.biomart.get_transcripts(chd.biomart.Dataset.from_genome(genome), gene_ids=adata.var.index.unique())
# pickle.dump(transcripts, (folder_data_preproc / 'transcripts.pkl').open("wb"))

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

# %%
adata.raw = adata

# %%
sc.pp.normalize_total(adata, size_factor)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata, n_top_genes = 2000)

# %%
sc.pp.pca(adata)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)


# %%
# sc.pl.umap(adata, color = "doublet_score")

# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene"]
def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene").loc[gene, "symbol"]


# %%
gene_id(adata, "CD34") in adata.var.index.tolist()

# %%
adata.obs["logn_counts"] = np.log10(adata.raw.to_adata().X.sum(1).A1)

# %%
sc.pl.umap(adata, color = "logn_counts")

# %%
adata.obs["sample"] == "GSE219248_Young2_HSPC"

# %%
sc.pl.umap(adata, color = "sample")

# %%
genes_oi = adata.var.index[[
    *np.argmax(adata.varm["PCs"], 0)[:10],
    *np.argmin(adata.varm["PCs"], 0)[:10],
]]
sc.pl.umap(adata, color = genes_oi, use_raw = False, title = symbol(adata, genes_oi))

# %%
genes_oi = gene_id(adata, ["CD34", "ERG", "HDAC9", "GATA1", "SPI1", "CDK1", "MCM5", "MCM3", "MKI67", "FLT3", "CCR2", "MPO", "SPIB", "IKZF2", "KLF1", "CSF1R", "ITGAX", "PTPRC"])
sc.pl.umap(adata, color = genes_oi, use_raw = False, title = symbol(adata, genes_oi))

# %%
genes_oi = gene_id(adata, ["MKI67", "MCM3", "PCNA", "CDKN1A", "BIRC5", "CCNE2", "CCNB2"])
sc.pl.umap(adata, color = genes_oi, use_raw = True, title = symbol(adata, genes_oi))

# %%
adata2 = adata

# %%
