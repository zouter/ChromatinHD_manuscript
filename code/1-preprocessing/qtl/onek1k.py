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
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "onek1k"
folder_qtl.mkdir(exist_ok = True, parents=True)

# %% [markdown]
# ## Download & cleaning

# %%
# !wget https://onek1k.s3.ap-southeast-2.amazonaws.com/onek1k_eqtl_dataset.zip -O {folder_qtl}/onek1k_eqtl_dataset.zip

# %%
# !unzip {folder_qtl}/onek1k_eqtl_dataset.zip

# %%
# !ls -lh onek1k_eqtl_dataset.tsv

# %% [markdown]
# Something weird is going on with this tsv given that the first line contains both the header and the first line of data...

# %%
# !head -n 3 onek1k_eqtl_dataset.tsv

# %%
import pathlib

# %%
eqtl_file = pathlib.Path(".") / "onek1k_eqtl_dataset.tsv"

# %%
with eqtl_file.open("r") as original: data = original.read()
a, b = data.split("\n", 1)
a1, a2 = a.split("bin")
a2 = "bin" + a2
with eqtl_file.open("w") as modified: modified.write(a1 + "\n" + a2 + "\n" + b)

# %%
# !head -n 3 onek1k_eqtl_dataset.tsv

# %% [markdown]
# Something else is weird about the data: some genomics positions are not stored as full integers, but rather in scientific notation, e.g. `7.2e+07`. I hope this truly means that the position is 72,000,000 or otherwise we're a bit screwed. Just as a check, not all positions of this size are stored like this, so I think we're safe

# %%
# !head -n 100000 {eqtl_file} | tail -n 1

# %% [markdown]
# However, we still need to convert this somehow. Let's do it in python and read this position column for now as a float. Let's also load in all relevant str columns as categorical to massively reduce memory footprint.

# %%
eqtl_file = pathlib.Path(".") / "onek1k_eqtl_dataset.tsv"

# %%
import polars as pl

# %%
eqtl = pl.read_csv(
    eqtl_file,
    sep = "\t",
    has_header = True,
    dtypes = {
        "POS":pl.Float64,
        "CELL_ID":pl.Categorical,
        "CELL_TYPE":pl.Categorical,
        "GENE":pl.Categorical,
        "GENE_ID":pl.Categorical,
        "GENOTYPED":pl.Categorical
    }
)
eqtl = eqtl.with_column(pl.col("POS").cast(pl.Int64))
eqtl = eqtl.rename(dict(zip(eqtl.columns, [col.lower() for col in eqtl.columns])))

# %% [markdown]
# Let's store this in a more efficient format than a 13GB tsv 🙄

# %%
eqtl["cell_id"].value_counts()

# %%
eqtl.filter(pl.col("q_value") < 0.05)["cell_id"].value_counts().to_pandas().set_index("cell_id").plot(kind = "bar")

# %% [markdown]
# Corresponds quite well to figure 2B (bottom right):
# ![](https://www.science.org/cms/10.1126/science.abf3041/asset/55db8185-d844-4f4d-90dd-e86b32693427/assets/images/large/science.abf3041-f2.jpg)

# %% [markdown]
# It's important to note that the dataframe is sorted by celltype, chromosome and position. We will resort it here by chromosome and position, ignoring the celltype. This is useful for any binary search later

# %%
eqtl = eqtl.sort([pl.col("chr"), pl.col("pos")])

# %%
eqtl.write_parquet(eqtl_file2)

# %%
np.searchsorted(eqtl["pos"].to_numpy(), 10000000)

# %%
# # !cp onek1k_eqtl_dataset.parquet /home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output/data/eqtl/onek1k/raw/

# %% [markdown]
# ## Create 

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"; organism = "hs"; chromosomes = ["chr"+str(i) for i in range(23)] + ["chrX", "chrY"]
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
# dataset_name = "brain"

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

# %%
eqtl_file2 = pathlib.Path(".") / "onek1k_eqtl_dataset.parquet"
eqtl = pl.read_parquet(eqtl_file2).filter(pl.col("fdr") < 0.05).to_pandas()

# %%
eqtl["chrom"] = "chr" + eqtl["chr"].astype(str)

# %% [markdown]
# ### Get SNP info

# %%
import sqlalchemy as sql
import pymysql

# %%
n = 1000

# %%
chunks = [eqtl["rsid"][i:i+n] for i in range(0, len(eqtl["rsid"]), n)]

# %%
len(chunks)

# %%
import tqdm.auto as tqdm

# %%
snp_info = []
for snp_names in tqdm.tqdm(chunks):
    snp_names = ",".join("'" + snp_names + "'")
    query = f"select name,chrom,chromStart,chromENd from snp151 where name in ({snp_names})"
    result = pd.read_sql(query,"mysql+pymysql://genome@genome-mysql.cse.ucsc.edu/{organism}?charset=utf8mb4".format(organism='hg38')).set_index("name")
    snp_info.append(result)

# %%
snp_info = pd.concat(snp_info)

# %%
snp_info.to_pickle(folder_qtl / "snp_info.pkl")

# %%
promoters.loc[transcriptome.gene_id("ANXA4")]

# %%
eqtl_oi = eqtl.query("gene == 'FCAR'")

# %%
eqtl_oi

# %%
fig, ax = plt.subplots()
ax.scatter(eqtl_oi["pos"], np.log(eqtl_oi["q_value"]))


# %% [markdown]
# ### Link QTLs to SNP location

# %%
snp_info = pd.read_pickle(folder_qtl / "snp_info.pkl")[["chrom", "chromENd"]].rename(columns = {"chromENd":"pos"})
snp_info.index.name = "snp"
snp_info = snp_info.loc[snp_info["chrom"].isin(chromosomes)]
snp_info = snp_info.groupby(level = 0).first()

# %%
eqtl = eqtl.drop(columns = ["pos", "chrom"]).rename(columns = {"rsid":"snp"}).join(snp_info, on = "snp")

# %%
chromosome_mapping = pd.Series(np.arange(len(chromosomes)), chromosomes)
promoters["chr_int"] = chromosome_mapping[promoters["chr"]].values

# %%
eqtl = eqtl.loc[eqtl.chrom.isin(chromosomes)].copy()

# %%
eqtl["chr"] = chromosome_mapping[eqtl["chrom"]].values

# %%
assert np.all(np.diff(eqtl["chr"].to_numpy()) >= 0), "Should be sorted by chr"

# %%
n = []

position_ixs = []
motif_ixs = []
scores = []

for gene_ix, promoter_info in enumerate(promoters.itertuples()):
    chr_int = promoter_info.chr_int
    chr_start = np.searchsorted(eqtl["chr"].to_numpy(), chr_int)
    chr_end = np.searchsorted(eqtl["chr"].to_numpy(), chr_int + 1)
    
    pos_start = chr_start + np.searchsorted(eqtl["pos"].iloc[chr_start:chr_end].to_numpy(), promoter_info.start)
    pos_end = chr_start + np.searchsorted(eqtl["pos"].iloc[chr_start:chr_end].to_numpy(), promoter_info.end)
    
    eqtls_promoter = eqtl.iloc[pos_start:pos_end].copy()
    eqtls_promoter["relpos"] = eqtls_promoter["pos"] - promoter_info.start
    
    if promoter_info.strand == -1:
        eqtls_promoter = eqtls_promoter.iloc[::-1].copy()
        eqtls_promoter["relpos"] = -eqtls_promoter["relpos"] + (window[1] - window[0]) + 1
        
    # if promoter_info.chr == 'chr6':
    #     eqtls_promoter = eqtls_promoter.loc[[]]
    
    n.append(len(eqtls_promoter))
    
    position_ixs += (eqtls_promoter["relpos"] + (gene_ix * (window[1] - window[0]))).astype(int).tolist()
    motif_ixs += (eqtls_promoter["cell_id"].cat.codes.values).astype(int).tolist()
    scores += (eqtls_promoter["fdr"]).tolist()
    
    # if transcriptome.var.iloc[gene_ix]["symbol"] == "TYMP":
    #     break
    
    # if len(eqtls_promoter) > 10:
    #     break

# %% [markdown]
# Control with sequence

# %%
# onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb"))
# eqtls_promoter.groupby("snp").first().head(20)
# onehot_promoters[gene_ix, 11000]

# %%
promoters["n"] = n

# %%
(promoters["n"] == 0).mean()

# %%
promoters.sort_values("n", ascending = False).head(30).assign(symbol = lambda x:transcriptome.symbol(x.index).values)

# %% [markdown]
# Just as a check, ENSG00000196126, encoding for HLA-DR, is very polymorphic and should therefore have plenty of SNPs and eQTLs

# %%
assert promoters.loc["ENSG00000196126", "n"] > 100

# %%
motifs_oi = eqtl[["cell_id", "cell_type"]].groupby(["cell_id"]).first()
motifs_oi["n"] = eqtl.groupby("cell_id").size()

# %%
motifs_oi.sort_values("n", ascending = False)

# %%
import scipy.sparse

# convert to csr, but using coo as input
motifscores = scipy.sparse.csr_matrix((scores, (position_ixs, motif_ixs)), shape = (len(promoters) * (window[1] - window[0]), motifs_oi.shape[0]))

# %%
motifscan_name = "onek1k_0.2"

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
# motifscan_folder = chd.get_output() / "motifscans" / dataset_name / promoter_name
# motifscan_folder.mkdir(parents=True, exist_ok=True)

# %%
pickle.dump(motifs_oi, open(motifscan.path / "motifs.pkl", "wb"))

# %%
# !ls -lh {motifscan.path}

# %% [markdown]
# ## Explore enrichemnt (TEMP)

# %%
enrichment = pd.read_pickle(chd.get_output() / "prediction_likelihood/pbmc10k/10k10k/leiden_0.1/v4_128-64-32_30_rep/scoring/cellranger/onek1k_0.2/motifscores_all.pkl")

# %%
group_motif_matching = pd.DataFrame([
    [0, "cd4et"],
    [0, "cd4nc"],
    [0, "cd4sox4"],
], columns = ["group", "motif"]).set_index(["group", "motif"])

# %%
group_motif_matching.join(enrichment.reset_index().set_index(["group", "motif"]))[["logodds_peak", "logodds_region"]].style.bar(axis = 1)

# %%
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.scatter(enrichment["logodds_peak"], enrichment["logodds_region"], c = enrichment["group"])
ax.axline((0, 0), slope = 1)

# %%
