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

# %% [markdown]
# # Preprocess

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import torch


import pickle

import scanpy as sc

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

original_dataset_name = "pbmc10k"
dataset_name = "pbmc10k_leiden_0.1"
organism = "hs"

original_folder_data_preproc = folder_data / original_dataset_name
original_folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

if organism == "hs":
    chromosome_names = ["chr" + str(i) for i in range(1, 23)]

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# %% [markdown]
# ### Load latent

# %%
latent_name = "leiden_0.1"
latent_folder = original_folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))
cluster_info = pickle.load((latent_folder / (latent_name + "_info.pkl")).open("rb"))

# %%
cluster_info["ix"] = range(len(cluster_info))

# %%
cell_to_cluster_ix = pd.Series(
    cluster_info["ix"][latent.idxmax(1)].values, latent.index
).to_dict()

# %% [markdown]
# ### Creating chromosomes

# %%
list((folder_data_preproc / "genome").iterdir())

# %%
chromosomes = (
    pd.read_table(
        folder_data_preproc / "genome" / "chromosome.sizes", names=["chr", "size"]
    )
    .set_index("chr")
    .loc[chromosome_names]
)
chromosomes.to_csv(folder_data_preproc / ("chromosomes.csv"))

chromosomes["position_start"] = np.hstack([[0], np.cumsum(chromosomes["size"])[:-1]])
chromosomes["position_end"] = np.cumsum(chromosomes["size"])

# %%
chunk_size = 100

# %%
cluster_info["n_cells"] = latent.idxmax(1).value_counts()

# %%
pickle.dump(latent, (folder_data_preproc / "latent.pkl").open("wb"))
cluster_info.to_pickle(folder_data_preproc / "cluster_info.pkl")

# %% [markdown]
# ### Create fragments

# %%
chunk_size = 100

# %%
import tabix

# %%
fragments_file = folder_data_preproc / "fragments.tsv.gz"
if not fragments_file.exists():
    (folder_data_preproc / "fragments.tsv.gz").symlink_to(
        original_folder_data_preproc / "atac_fragments.tsv.gz"
    )
    (folder_data_preproc / "fragments.tsv.gz.tbi").symlink_to(
        original_folder_data_preproc / "atac_fragments.tsv.gz.tbi"
    )

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "fragments.tsv.gz"))

# %%
coordinates_raw = []
indices_cluster_raw = []

i = 0

for chromosome, chromosome_info in tqdm.tqdm(
    chromosomes.iterrows(), total=len(chromosomes)
):
    chromosome_position_start = chromosome_info["position_start"]
    fragments_chromosome = fragments_tabix.query(chromosome, 0, chromosome_info["size"])
    for fragment in fragments_chromosome:
        i += 1
        cell = fragment[3]
        if cell in cell_to_cluster_ix:
            coordinates_raw.append(
                [
                    int(fragment[1]) + chromosome_position_start,
                    int(fragment[2]) + chromosome_position_start,
                ]
            )
            indices_cluster_raw.append(cell_to_cluster_ix[cell])
        if (i % 10000000) == 0:
            print(len(coordinates_raw))

coordinates = torch.tensor(
    np.array(coordinates_raw, dtype=np.int64)
)  # int64 is absolutely needed here to keep within int32_max
coordinates = coordinates.flatten()

chunkcoords = torch.div(coordinates, chunk_size).to(
    torch.int32
)  # int32 can be added here since the chunks are much smaller

clusters = torch.tensor(np.array(indices_cluster_raw), dtype=torch.int32)
clusters = torch.repeat_interleave(clusters, 2)

relcoords = coordinates % chunk_size

# %%
import chromatinhd.data

# %%
import pathlib

fragments = chd.data.fragments.ChunkedFragments(folder_data_preproc / "fragments")

# %%
assert coordinates.shape == clusters.shape
assert chunkcoords.shape == relcoords.shape

# %%
fragments.chromosomes = chromosomes
fragments.clusters_info = cluster_info
fragments.chunk_size = chunk_size

# %%
sorted_idx = torch.argsort(coordinates)

# %%
fragments.chunkcoords = chunkcoords[sorted_idx]
fragments.clusters = clusters[sorted_idx]
fragments.relcoords = relcoords[sorted_idx]
fragments.coordinates = coordinates[sorted_idx]

# %% [markdown]
# Check size

# %%
np.product(chunkcoords.size()) * 32 / 8 / 1024 / 1024

# %%
np.product(clusters.size()) * 32 / 8 / 1024 / 1024

# %% [markdown]
# Create chunkcoord index pointers

# %%
n_chunks = chromosomes["position_end"][-1] // chunk_size

# %%
fragments.chunkcoords_indptr = chd.utils.torch.indices_to_indptr(chunkcoords, n_chunks)

# %% [markdown]
# ## Create genes
#

# %%
biomart_dataset_name = "hsapiens_gene_ensembl"
query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >

    <Dataset name = "{biomart_dataset_name}" interface = "default" >
        <Filter name = "transcript_is_canonical" excluded = "0"/>
        <Filter name = "transcript_biotype" value = "protein_coding"/>
        <Attribute name = "ensembl_gene_id" />
        <Attribute name = "transcript_start" />
        <Attribute name = "transcript_end" />
        <Attribute name = "end_position" />
        <Attribute name = "start_position" />
        <Attribute name = "ensembl_transcript_id" />
        <Attribute name = "chromosome_name" />
        <Attribute name = "strand" />
        <Attribute name = "external_gene_name" />
    </Dataset>
</Query>"""
url = "http://www.ensembl.org/biomart/martservice?query=" + query.replace(
    "\t", ""
).replace("\n", "")
from io import StringIO
import requests

session = requests.Session()
session.headers.update({"User-Agent": "Custom user agent"})
r = session.get(url)
result = pd.read_table(StringIO(r.content.decode("utf-8")))

# %%
genes = result.rename(
    columns={
        "Gene stable ID": "gene",
        "Transcript start (bp)": "start",
        "Transcript end (bp)": "end",
        "Chromosome/scaffold name": "chr",
        "Gene name": "symbol",
        "Strand": "strand",
    }
)
genes["chr"] = "chr" + genes["chr"].astype(str)
genes = genes.groupby("gene").first()

# %%
genes = genes.loc[genes["chr"].isin(chromosomes.index)]

# %%
assert (
    genes.groupby(level=0).size().mean() == 1
), "For each gene, there should only be one transcript"

# %%
genes.to_csv(folder_data_preproc / "genes.csv")

# %% [markdown]
# ## Subset BAM and create cluster BigWigs

# %% [markdown]
# https://github.com/10XGenomics/subset-bam

# %% [markdown]
# ### Create subset lists

# %% [markdown]
# Ran on GPU server

# %%
subset_list_dir = folder_data_preproc / "subset"
subset_list_dir.mkdir(exist_ok=True, parents=True)

# %%
cluster_to_cells = {}
for cluster, cells in (
    latent.idxmax(1).to_frame("cluster").reset_index().groupby("cluster")["cell"]
):
    cluster_to_cells["cluster"] = cells
    cells.to_csv(subset_list_dir / (cluster + ".csv"), index=False, header=False)

# %%
import subprocess

# %%
source_file

# %%
target_file

# %%
source_output = chd.get_output()
target_output = (
    "/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output"
)
source_files = [subset_list_dir]
target_files = [
    target_output / file.relative_to(source_output) for file in source_files
]

for source_file, target_file in zip(source_files, target_files):
    target_file.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "rsync",
            "-av",
            "--progress",
            # "--dry-run",
            str(source_file),
            str(target_file.parent),
        ]
    )
    proc.communicate()

# %% [markdown]
# ### Subset

# %%
# !wget https://github.com/10XGenomics/subset-bam/releases/download/v1.1.0/subset-bam_linux -O $HOME/bin/subset_bam
# !chmod +x $HOME/bin/subset_bam

# %% tags=[]
bam_location = original_folder_data_preproc / "bam/atac_possorted_bam.bam"

# %% tags=[]
subset_list_dir = folder_data_preproc / "subset"
subset_list_dir.mkdir(exist_ok=True, parents=True)

# %% tags=[]
source_output = (
    "/home/wsaelens/NAS2/wsaelens/projects/chromatinhd/chromatinhd_manuscript/output"
)
target_folder = subset_list_dir
target_output = chd.get_output()
source_folder = source_output / target_folder.relative_to(target_output)
if not target_folder.exists():
    target_folder.symlink_to(source_folder)

# %% tags=[]
# !samtools index -@ 20 {bam_location}

# %% tags=[]
for cluster in cluster_info.index:
    subset_file = subset_list_dir / (cluster + ".csv")
    out_bam = subset_list_dir / (cluster + ".bam")
    # !subset_bam --bam {bam_location} --cell-barcodes "{subset_file}" --out-bam {out_bam}
