# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import tqdm.auto as tqdm

import pathlib

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %% [markdown]
# ## Create the SNP motifscan

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name, genome = ("liver", "mm10")

# dataset_name, genome = ("lymphoma", "GRCh38")
# dataset_name, genome = ("pbmc10k", "GRCh38")
dataset_name, genome = ("hspc", "GRCh38")

dataset_folder = chd.get_output() / "datasets" / dataset_name

transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")

# %%
# regions_name = "10k10k"
regions_name = "100k100k"
regions = chd.data.Regions(dataset_folder / "regions" / regions_name)

# %% [markdown]
# ### Link QTLs to SNP location

# %%
chromosomes = regions.coordinates["chrom"].unique()

# %%
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + qtl_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))

if "gtex" in qtl_name:
    association = qtl_mapped
    association["disease/trait"] = association["tissue"]
else:
    association = qtl_mapped.join(snp_info, on="snp")
    association = association.loc[~pd.isnull(association["start"])]
    association["pos"] = association["start"].astype(int)

if motifscan_name.endswith("main"):
    association = association.loc[association["snp_main"] == association["snp"]]

# %%
# pip install liftover
# import liftover

# %% [markdown]
# LiftOver if necessary

# %%
import liftover
if genome == "mm10":
    if "converter" not in globals():
        if not pathlib.Path("hg38ToMm10.over.chain.gz").exists():
            # !wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToMm10.over.chain.gz
        converter = liftover.ChainFile("hg38ToMm10.over.chain.gz", "hg38", "mm10")
    association_old = association
    association_new = []
    for _, row in tqdm.tqdm(association.iterrows(), total=len(association)):
        converted = converter.convert_coordinate(row["chr"], row["pos"])
        if len(converted) == 1:
            converted = converted[0]
            row["chr"] = converted[0]
            row["pos"] = converted[1]
            association_new.append(row)
    association_new = pd.DataFrame(association_new)
    association = association_new

    print(len(association_old), len(association_new), len(association))

# %%
import pybedtools

# %%
association_bed = pybedtools.BedTool.from_dataframe(
    association.reset_index()[["chr", "pos", "pos", "index"]]
)

coordinates = regions.coordinates
coordinates["start"] = np.clip(coordinates["start"], 0, np.inf)
coordinates_bed = pybedtools.BedTool.from_dataframe(coordinates[["chrom", "start", "end"]])
intersection = association_bed.intersect(coordinates_bed)
association = association.loc[intersection.to_dataframe()["name"].unique()]

chromosome_mapping = pd.Series(np.arange(len(chromosomes)), chromosomes)
coordinates["chr_int"] = chromosome_mapping[coordinates["chrom"]].values

association = association.loc[association.chr.isin(chromosomes)].copy()
association["chr_int"] = chromosome_mapping[association["chr"]].values
association = association.sort_values(["chr_int", "pos"])

assert np.all(
    np.diff(association["chr_int"].to_numpy()) >= 0
), "Should be sorted by chr"

motif_col = "disease/trait"
association[motif_col] = association[motif_col].astype("category")

# %%
# if differential, we only select eQTLs that affect differentially expressed genes
if "differential" in motifscan_name:
    transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")
    gene_ids = transcriptome.var.sort_values("dispersions_norm", ascending = False).query("dispersions_norm > 1").index

    if genome == "mm10":
        # convert grch38 to mm10 gene ids
        association["gene"] = chd.biomart.get_orthologs(chd.biomart.Dataset.from_genome("GRCh38"), association["gene"])
        
    association = association.loc[association["gene"].isin(gene_ids)].copy()

# %%
len(association["snp"].unique())

# %%
assert association[motif_col].dtype.name == "category"

# %%
n = []

coordinate_ixs = []
region_ixs = []
position_ixs = []
motif_ixs = []
scores = []

for gene_ix, region_info in enumerate(coordinates.itertuples()):
    chr_int = region_info.chr_int
    chr_start = np.searchsorted(association["chr_int"].to_numpy(), chr_int)
    chr_end = np.searchsorted(association["chr_int"].to_numpy(), chr_int + 1)

    pos_start = chr_start + np.searchsorted(
        association["pos"].iloc[chr_start:chr_end].to_numpy(), region_info.start
    )
    pos_end = chr_start + np.searchsorted(
        association["pos"].iloc[chr_start:chr_end].to_numpy(), region_info.end
    )

    qtls_promoter = association.iloc[pos_start:pos_end].copy()
    qtls_promoter["relpos"] = qtls_promoter["pos"] - region_info.tss

    if region_info.strand == -1:
        qtls_promoter = qtls_promoter.iloc[::-1].copy()
        qtls_promoter["relpos"] = -qtls_promoter["relpos"]# + regions.width + 1


    # if "rs10065637" in qtls_promoter["snp"].values.tolist():
        # print(gene_ix)
        # raise ValueError

    n.append(len(qtls_promoter))

    coordinate_ixs += (
        (qtls_promoter["relpos"])
        .astype(int)
        .tolist()
    )
    region_ixs += [gene_ix] * len(qtls_promoter)
    position_ixs += (
        (qtls_promoter["relpos"] + (gene_ix * regions.width))
        .astype(int)
        .tolist()
    )
    motif_ixs += (qtls_promoter[motif_col].cat.codes.values).astype(int).tolist()
    scores += [1] * len(qtls_promoter)

# %% [markdown]
# Control with sequence

# %%
# onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb"))
# qtls_promoter.groupby("snp").first().head(20)
# onehot_promoters[gene_ix, 11000]

# %%
coordinates["n"] = n
(coordinates["n"] == 0).mean()

# %%
coordinates.sort_values("n", ascending=False).head(30).assign(
    symbol=lambda x: transcriptome.symbol(x.index).values
)

# %%
# coordinates.sort_values("n", ascending=False).assign(
#     symbol=lambda x: transcriptome.symbol(x.index).values
# ).set_index("symbol").loc["POU2AF1"]

# %%
motifs_oi = association[[motif_col]].groupby([motif_col]).first()
motifs_oi["n"] = association.groupby(motif_col).size()

# %%
motifs_oi.sort_values("n", ascending=False)

# %% [markdown]
# ### Save

# %%
if "immune" in motifscan_name:
    association = association.loc[association["chr"] != "chr6"]

# %%
import chromatinhd as chd

# %%
motifscan = chd.data.Motifscan(
    chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name, reset = True
)

# %%
motifscan.regions = regions
motifscan.coordinates = np.array(coordinate_ixs)
motifscan.region_indices = np.array(region_ixs)
motifscan.indices = np.array(motif_ixs)
motifscan.scores = np.array(scores)
motifscan.strands = np.array([1] * len(motifscan.indices))
motifscan.motifs = motifs_oi

# %%
motifscan.create_region_indptr(overwrite = True)

# %%
association["snp_main"] = association["snp"]
association["rsid"] = association["snp"]

# %%
association.to_pickle(motifscan.path / "association.pkl")

# %%

# %%
