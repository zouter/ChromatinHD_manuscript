# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile
import requests

# %% [markdown]
# ## Load data

# %% tags=[]
data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"

# %% tags=[]
transcriptome = chd.data.transcriptome.transcriptome.ClusterTranscriptome(data_folder / "transcriptome")

# %% tags=[]
adata2 = transcriptome.adata

# %% tags=[]
variants_info = pd.read_pickle(data_folder / "variants_info.pkl")
variants_info["ix"] = np.arange(len(variants_info))

# %% tags=[]
samples = pd.read_csv(data_folder / "raw" / "samples.csv")

# %% tags=[]
genes = pd.read_csv(data_folder / "genes.csv", index_col = 0)

# %% tags=[]
cluster_info = pd.DataFrame({"cluster":transcriptome.obs["cluster"].unique().sort_values()}).set_index("cluster")
cluster_info["ix"] = np.arange(len(cluster_info))

# %% tags=[]
final_file = data_folder /"final.bcf.gz"

# %% tags=[]
import cyvcf2
vcf = cyvcf2.VCF(final_file)

# %% tags=[]
adata2.var["std"] = adata2.X.std(0)

# %% tags=[]
genes["std"] = adata2.var["std"]

# %% tags=[]
(genes
     # .query("biotype == 'lncRNA'")
     .sort_values("std", ascending = False)
     .head(20)
)

# %% tags=[]
import xarray as xr

expression = pd.DataFrame(adata2.raw.X, columns = adata2.var.index)
expression.index = pd.MultiIndex.from_frame(adata2.obs[["cluster", "donor"]])
expression = xr.DataArray(expression).unstack()
expression.values[np.isnan(expression.values)] = 0.


# %% tags=[]
def get_types(window):
    # f"{gene_oi.chr}:{window[0]}-{window[1]}"
    # get types
    types = []
    variant_ids = []
    for variant in vcf(window):
        types.append(variant.gt_types.copy()) #! Copy is important due to repurposing of gt_types by cyvcf2
        variant_ids.append(variant.ID)
    types = np.stack(types, -1)
    types[types == 2] = 1
    types[types == 3] = 2
    return types, variant_ids

def get_expression(gene_id):
    expression = sc.get.obs_df(adata2, gene_oi.name)
    expression.index = pd.MultiIndex.from_frame(adata2.obs[["cluster", "donor"]])
    expression = expression.unstack().T
    expression.index = samples.set_index("donor")["old"].values
    expression = expression.reindex(vcf.samples)
    return expression


# %% tags=[]
gene_oi = genes.query("symbol == 'CTLA4'").iloc[0]

# %% tags=[]
expression_oi = expression.sel(gene = gene_oi.name).transpose("donor", "cluster")

# %% tags=[]
window_size = 10**6
window = (gene_oi["tss"] - window_size, gene_oi["tss"] + window_size)

# get types
window_str = f"{gene_oi.chr}:{window[0]}-{window[1]}"
genotype, variant_ids = get_types(window_str)

# %% tags=[]
variants_info_oi = variants_info.loc[variant_ids]

# %% tags=[]
clusters_info = pd.DataFrame({"cluster":pd.Series(adata2.obs["cluster"].unique()).sort_values().dropna()}).set_index("cluster")
clusters_info["ix"] = np.arange(len(clusters_info))
cluster_to_cluster_ix = clusters_info["ix"].to_dict()

genes["ix"] = np.arange(len(genes))
gene_to_gene_ix = genes["ix"].to_dict()

donors_info = samples.set_index("donor")

# %% tags=[]
import torch

# %% tags=[]
lib = expression_oi.sum("donor") / 10**6

# %% tags=[]
n_clusters = len(clusters_info)
n_donors = len(donors_info)
n_variants = len(variants_info_oi)

# %%
from model import Model
lib_torch = torch.from_numpy(lib.values)
model = Model(n_clusters, n_donors, n_variants, lib_torch)

# %% tags=[]
genotype_torch = torch.from_numpy(genotype)
expression_oi_torch = torch.from_numpy(expression_oi.values)

# %% tags=[]
model.forward(genotype_torch, expression_oi_torch)
