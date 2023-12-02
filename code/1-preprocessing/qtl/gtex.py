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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import tqdm.auto as tqdm

import pathlib

import polars as pl

# %%
import chromatinhd as chd

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gtex"
folder_qtl.mkdir(exist_ok = True, parents=True)

# %%
# !curl --location https://storage.googleapis.com/gtex_analysis_v8/single_tissue_qtl_data/GTEx_Analysis_v8_eQTL.tar > {folder_qtl}/GTEx_Analysis_v8_eQTL.tar

# %%
# !tar -tvf {folder_qtl}/GTEx_Analysis_v8_eQTL.tar

# %%
# !tar -xf {folder_qtl}/GTEx_Analysis_v8_eQTL.tar -C {folder_qtl} GTEx_Analysis_v8_eQTL/Whole_Blood.v8.signif_variant_gene_pairs.txt.gz

# %%
# !tar -xf {folder_qtl}/GTEx_Analysis_v8_eQTL.tar -C {folder_qtl} GTEx_Analysis_v8_eQTL/Brain_Cerebellum.v8.signif_variant_gene_pairs.txt.gz

# %%
# # !zcat {folder_qtl}/GTEx_Analysis_v8_eQTL/Whole_Blood.v8.signif_variant_gene_pairs.txt.gz | head -n 5

# %%
motifscan_name = "gtex_immune"; files = [
    ["Whole_Blood.v8.signif_variant_gene_pairs.txt.gz", "whole_blood"]
]

motifscan_name = "gtex_cerebellum"; files = [
    ["Brain_Cerebellum.v8.signif_variant_gene_pairs.txt.gz", "cerebellum"]
]

# %%
qtl = pd.concat([
    pd.read_table(folder_qtl/"GTEx_Analysis_v8_eQTL"/file).assign(tissue = tissue) for file, tissue in files
])

# %%
qtl["chr"], qtl["pos"] = qtl["variant_id"].str.split("_").str[0], qtl["variant_id"].str.split("_").str[1].astype(int)

# %%
qtl["abs_tss_distance"] = np.abs(qtl["tss_distance"])

# %% [markdown]
# ## Create 

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
dataset_name = "brain"

# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"

folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %% [markdown]
# ### Link QTLs to SNP location

# %%
chromosomes = promoters["chr"].sort_values().unique().tolist()

# %%
chromosome_mapping = pd.Series(np.arange(len(chromosomes)), chromosomes)
promoters["chr_int"] = chromosome_mapping[promoters["chr"]].values

# %%
qtl = qtl.loc[qtl.chr.isin(chromosomes)].copy()

# %%
qtl["chr_int"] = chromosome_mapping[qtl["chr"]].values

# %%
qtl = qtl.sort_values(["chr_int", "pos"])

# %%
assert np.all(np.diff(qtl["chr_int"].to_numpy()) >= 0), "Should be sorted by chr"

# %%
motif_col = "tissue"
qtl[motif_col] = qtl[motif_col].astype("category")
assert qtl[motif_col].dtype.name == "category"

# %%
n = []

position_ixs = []
motif_ixs = []
scores = []

for gene_ix, promoter_info in enumerate(promoters.itertuples()):
    chr_int = promoter_info.chr_int
    chr_start = np.searchsorted(qtl["chr_int"].to_numpy(), chr_int)
    chr_end = np.searchsorted(qtl["chr_int"].to_numpy(), chr_int + 1)
    
    pos_start = chr_start + np.searchsorted(qtl["pos"].iloc[chr_start:chr_end].to_numpy(), promoter_info.start)
    pos_end = chr_start + np.searchsorted(qtl["pos"].iloc[chr_start:chr_end].to_numpy(), promoter_info.end)
    
    qtls_promoter = qtl.iloc[pos_start:pos_end].copy()
    qtls_promoter["relpos"] = qtls_promoter["pos"] - promoter_info.start
    
    if promoter_info.strand == -1:
        qtls_promoter = qtls_promoter.iloc[::-1].copy()
        qtls_promoter["relpos"] = -qtls_promoter["relpos"] + (window[1] - window[0]) + 1
    
    n.append(len(qtls_promoter))
    
    position_ixs += (qtls_promoter["relpos"] + (gene_ix * (window[1] - window[0]))).astype(int).tolist()
    motif_ixs += (qtls_promoter[motif_col].cat.codes.values).astype(int).tolist()
    scores += ([1] * len(qtls_promoter))

# %%
promoter_info

# %%
-qtls_promoter["relpos"] + (window[1] - window[0]) + 1

# %% [markdown]
# Control with sequence

# %%
# onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb"))
# qtls_promoter.groupby("snp").first().head(20)
# onehot_promoters[gene_ix, 11000]

# %%
promoters["n"] = n

# %%
(promoters["n"] == 0).mean()

# %%
promoters.sort_values("n", ascending = False).head(30).assign(symbol = lambda x:transcriptome.symbol(x.index).values)

# %%
motifs_oi = qtl[[motif_col]].groupby([motif_col]).first()
motifs_oi["n"] = qtl.groupby(motif_col).size()

# %%
motifs_oi.sort_values("n", ascending = False)

# %%
import scipy.sparse

# convert to csr, but using coo as input
motifscores = scipy.sparse.csr_matrix((scores, (position_ixs, motif_ixs)), shape = (len(promoters) * (window[1] - window[0]), motifs_oi.shape[0]))

# %% [markdown]
# ### Save

# %%
import chromatinhd as chd

# %%
motifscan = chd.data.Motifscan(chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name)

# %%
motifscan.indices = motifscores.indices
motifscan.indptr = motifscores.indptr
motifscan.data = motifscores.data
motifscan.shape = motifscores.shape

# %%
motifscan

# %%
pickle.dump(motifs_oi, open(motifscan.path / "motifs.pkl", "wb"))

# %%
motifscan.n_motifs = len(motifs_oi)

# %%
# !ls -lh {motifscan.path}

# %%

# %%
