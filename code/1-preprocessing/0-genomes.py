# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
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

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm

import chromatinhd as chd
import gzip


# %%
def digitize_genome(folder_genome, chromosomes):
    import gzip
    genome = {}
    chromosome = None
    translate_table = {"A":0, "C":1, "G":2, "T":3, "N":4} # alphabetic order
    for i, line in enumerate(gzip.GzipFile(folder_genome / "dna.fa.gz")):
        line = str(line,'utf-8')
        if line.startswith(">"):
            if chromosome is not None:
                genome[chromosome] = np.array(genome_chromosome, dtype = np.int8)
            chromosome = "chr" + line[1:line.find(" ")]
            genome_chromosome = []

            print(chromosome)

            if chromosome not in chromosomes:
                break
        else:
            genome_chromosome += [translate_table[x] for x in line.strip("\n").upper()]
    return genome


# %% [markdown]
# ## GRCh38

# %%
genome = "GRCh38.107"
folder_genome = chd.get_output() / "data" / "genomes" / genome
folder_genome.mkdir(exist_ok = True, parents=True)

chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %%
# !wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes -O  {folder_genome}/chromosome.sizes

# %%
# !wget http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz -O {folder_genome}/dna.fa.gz

# %% [markdown]
# Store the genome as integers, which is much easier to work with. Bases are stored in alphabetic order ACGT(N) = 01234

# %%
genome = digitize_genome(folder_genome, chromosomes)
pickle.dump(genome, gzip.GzipFile((folder_genome / "genome.pkl.gz"), "wb", compresslevel = 3))

# %% [markdown]
# ### Genes

# %% [markdown]
# Get the gene annotaton from ensembl biomart

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
url = "http://www.ensembl.org/biomart/martservice?query=" + query.replace("\t", "").replace("\n", "")
from io import StringIO
import requests
session = requests.Session()
session.headers.update({'User-Agent': 'Custom user agent'})
r = session.get(url)
result = pd.read_table(StringIO(r.content.decode("utf-8")))

# %%
genes = result.rename(columns = {
    "Gene stable ID":"gene",
    "Transcript start (bp)":"start",
    "Transcript end (bp)":"end",
    "Chromosome/scaffold name":"chr",
    "Gene name":"symbol",
    "Strand":"strand"
})
genes["chr"] = "chr" + genes["chr"].astype(str)
genes = genes.groupby("gene").first()
genes = genes.loc[genes["chr"].isin(chromosomes)]
assert genes.groupby(level = 0).size().mean() == 1, "For each gene, there should only be one transcript"

genes.to_csv(folder_genome / "genes.csv")

# %% [markdown]
# ## mm10

# %%
genome = "mm10"
chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]

folder_genome = chd.get_output() / "data" / "genomes" / genome
folder_genome.mkdir(exist_ok = True, parents=True)

# %%
# !wget http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes -O  {folder_genome}/chromosome.sizes

# %%
# !wget http://ftp.ensembl.org/pub/release-98/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna_sm.toplevel.fa.gz -O {folder_genome}/dna.fa.gz

# %%
genome = digitize_genome(folder_genome, chromosomes)
pickle.dump(genome, gzip.GzipFile((folder_genome / "genome.pkl.gz"), "wb", compresslevel = 3))

# %%
(folder_genome / "dna.fa.gz").unlink()

# %% [markdown]
# ### Genes

# %%
biomart_dataset_name = "mmusculus_gene_ensembl"
query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >

    <Dataset name = "{biomart_dataset_name}" interface = "default" >
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
url = "https://nov2020.archive.ensembl.org:443/biomart/martservice?query=" + query.replace("\t", "").replace("\n", "")
from io import StringIO
import requests
session = requests.Session()
session.headers.update({'User-Agent': 'Custom user agent'})
r = session.get(url)
result = pd.read_table(StringIO(r.content.decode("utf-8")))

# %%
genes = result.rename(columns = {
    "Gene stable ID":"gene",
    "Transcript start (bp)":"start",
    "Transcript end (bp)":"end",
    "Chromosome/scaffold name":"chr",
    "Gene name":"symbol",
    "Strand":"strand"
})
genes["chr"] = "chr" + genes["chr"].astype(str)
genes = genes.groupby("gene").first()
genes = genes.loc[genes["chr"].isin(chromosomes)]
assert genes.groupby(level = 0).size().mean() == 1, "For each gene, there should only be one transcript"

# %%
genes.to_csv(folder_genome / "genes.csv")
