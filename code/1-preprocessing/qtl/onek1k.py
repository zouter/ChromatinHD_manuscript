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
import polyptich as pp
pp.setup_ipython()

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
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "onek1k"

# %%
remote_root = pathlib.Path("/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output")
assert remote_root.exists()
local_root = chd.get_output()

remote_folder = remote_root / "data" / "qtl" / "hs" / "onek1k"
remote_folder.mkdir(exist_ok = True, parents=True)

local_folder = folder_qtl

# symlink remote_folder to local_folder
if local_folder.is_dir() and not local_folder.is_symlink():
    local_folder.rmdir()
if not local_folder.is_symlink():
    local_folder.symlink_to(remote_folder)

# %% [markdown]
# ## Download & process

# %%
if not (remote_folder / "eqtl_table.tsv.gz").exists():
    # !wget https://onek1k.s3.ap-southeast-2.amazonaws.com/eqtl/eqtl_table.tsv.gz -O {folder_qtl / "eqtl_table.tsv.gz"}

# %%
import gzip
lines_significant = []
with gzip.open(folder_qtl / "eqtl_table.tsv.gz", "rt") as f:
    line = f.readline()
    columns = line.strip().split("\t")
    for line in tqdm.tqdm(f):
        line = line.strip().split("\t")
        if float(line[16]) < 0.05: # 14=p-value, 15=q-value, 16=fdr
            lines_significant.append(line)
            if (len(lines_significant) % 10000) == 0:
                print(len(lines_significant), lines_significant[-1][1])

# %%
data = pd.DataFrame(lines_significant, columns=pd.Series(columns).str.lower())

# %%
data["fdr"] = data["fdr"].astype(float)

# %%
data["cell_type"] = data["cell_type"].astype("category")
data["rsid"] = data["rsid"].astype("category")
data["gene_id"] = data["gene_id"].astype("category")
data["gene"] = data["gene"].astype("category")
data["genotyped"] = data["genotyped"].astype("category")
data["cell_id"] = data["cell_id"].astype("category")

# %%
pickle.dump(data, open(folder_qtl / "eqtl_table.pkl", "wb"))

# %% [markdown]
# ## Read all data of significant SNPs

# %%
# data = pickle.load(open(folder_qtl / "eqtl_table.pkl", "rb"))

# %%
folder_qtl_reference = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
qtl_name = "gwas_immune"
qtl_mapped_reference = pd.read_pickle(folder_qtl_reference / ("qtl_mapped_" + qtl_name + ".pkl"))

snps_oi = qtl_mapped_reference["snp_main"].unique()
snps_oi = set(qtl_mapped_reference["snp"].unique())
len(snps_oi)

# %%
import gzip
lines_significant = []
with gzip.open(folder_qtl / "eqtl_table.tsv.gz", "rt") as f:
    line = f.readline()
    columns = line.strip().split("\t")
    for line in tqdm.tqdm(f):
        line = line.strip().split("\t")
        if line[2] in snps_oi:
            lines_significant.append(line)
            if (len(lines_significant) % 1000000) == 0:
                print(len(lines_significant), lines_significant[-1][1])

# %%
data = pd.DataFrame(lines_significant, columns=pd.Series(columns).str.lower())

# %%
data["fdr"] = data["fdr"].astype(float)
data["rsquare"] = data["rsquare"].astype(float)

# %%
data["cell_type"] = data["cell_type"].astype("category")
data["rsid"] = data["rsid"].astype("category")
data["gene_id"] = data["gene_id"].astype("category")
data["gene"] = data["gene"].astype("category")
data["genotyped"] = data["genotyped"].astype("category")
data["cell_id"] = data["cell_id"].astype("category")

# %%
data["pos"] = data["pos"].astype(float).astype(int)
data["spearmans_rho"] = data["spearmans_rho"].astype(float)

# %%
data["rsquare"] = data["rsquare"].astype(float)
data["q_value"] = data["q_value"].astype(float)
data["pos"] = data["pos"].astype(float).astype(int)

# %%
data = data[["cell_type", "rsid", "gene", "gene_id", "chr", "pos", "q_value", "fdr", "rsquare", "genotyped", "spearmans_rho"]]

# %%
pickle.dump(data, open(folder_qtl / "eqtl_table_gwas_immune.pkl", "wb"))

# %% [markdown]
# ## Process further

# %%
data = pickle.load(open(folder_qtl / "eqtl_table.pkl", "rb")) # only significant
# data = pickle.load(open(folder_qtl / "eqtl_table_gwas_immune.pkl", "rb"))

# %% [markdown]
# ### Select SNPs of interest from GWAS

# %%
folder_qtl_reference = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
qtl_name = "gwas_immune"
qtl_mapped_reference = pd.read_pickle(folder_qtl_reference / ("qtl_mapped_" + qtl_name + ".pkl"))

snps_oi = qtl_mapped_reference["snp_main"].unique()
snps_oi = qtl_mapped_reference["snp"].unique()

# %% [markdown]
# ### Determine cell-type specific SNPs

# %%
snps_oi = celltype_snp_significant.columns[celltype_snp_significant.any()]

# %%
mapping = {
    "CD4 T":["CD4 Effector memory/TEMRA", "CD4 Naive/Central memory T cell", "CD4 SOX4 T cell"],
    "CD8 T":["CD8 Effector memory", "CD8 Naive/Central memory T cell", "CD8 S100B T cell"],
    "Monocyte":["Classic Monocyte", "Non-classic Monocyte"],
    "NK":["Natural Killer Cell", "Natural Killer Recruiting Cell"],
    "cDC":["Dendritic Cell"],
    "B":["Naïve/Immature B Cell", "Memory B Cell", "Plasma Cell"]
}

# %%
data_oi = data.loc[data["rsid"].isin(snps_oi)].copy()
data_oi["rsid"] = data_oi["rsid"].astype(str).astype("category")

celltype_snp_qvalues = data_oi.groupby(["cell_type", "rsid"])["fdr"].min().unstack()
celltype_snp_significant = celltype_snp_qvalues < 0.05
celltype_snp_r = data_oi.groupby(["cell_type", "rsid"])["spearmans_rho"].mean().unstack()

# %%
celltype_snp_significant

# %%
celltype_snp_r2

# %%
celltype_snp_r2.loc["Plasma Cell"].sort_values()

# %%
data_oi

# %%
data_oi.set_index(["cell_type", "rsid"])

# %%
celltype_snp_qvalues

# %%
group_snps = []
for group, cell_types in mapping.items():
    other_cell_types = [x for x in data_oi["cell_type"].unique() if x not in cell_types]
    
    group_snps.append(pd.DataFrame({
        "snp":celltype_snp_significant.columns[celltype_snp_significant.loc[cell_types].any(axis = 0) & ~celltype_snp_significant.loc[other_cell_types].any(axis = 0)],
        "group":group
    }))

    # celltype_snps
    # data_oi.loc[data_oi["cell_type"].isin(cell_types), "cell_type"] = group
group_snps = pd.concat(group_snps)

# %%
qtl_mapped = pd.DataFrame({"snp":group_snps["snp"], "disease/trait":group_snps["group"]})

# %%
pickle.dump(qtl_mapped, open(folder_qtl / "qtl_mapped_gwas_specific.pkl", "wb"))

# %% [markdown]
# Corresponds quite well to figure 2B (bottom right):
# ![](https://www.science.org/cms/10.1126/science.abf3041/asset/55db8185-d844-4f4d-90dd-e86b32693427/assets/images/large/science.abf3041-f2.jpg)

# %% [markdown]
# ## SPecific

# %%
# # !zcat {folder_qtl / "eqtl_table.tsv.gz"} | grep rs4795397 > {folder_qtl / "eqtl_table_rs4795397.tsv"}
# # !zcat {folder_qtl / "eqtl_table.tsv.gz"} | grep rs443623 > {folder_qtl / "eqtl_table_rs443623.tsv"}
# !zcat {folder_qtl / "eqtl_table.tsv.gz"} | grep rs59754851 > {folder_qtl / "eqtl_table_rs59754851.tsv"}

# %%
data = pd.read_table(folder_qtl / "eqtl_table_rs443623.tsv", nrows=1000, names = [x.lower() for x in gzip.open(folder_qtl / "eqtl_table.tsv.gz", "rt").readline().strip().split("\t")])

# %%
data.query("gene == 'HLA-DOA'")

# %%
data["significant"] = data["fdr"] < 0.05

# %%
data.query("significant")["gene"].value_counts()

# %%
data.iloc[0]

# %%
data.sort_values("fdr").query("round == 1").query("gene == 'ORMDL3'").set_index("cell_type")["q_value"].plot.bar()

# %%
data.sort_values("fdr").head(20).style.bar(subset = ["spearmans_rho"], vmin = -.4, vmax = .4)
