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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import math

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import chromatinhd as chd

# %% [markdown]
# ## Download and process JASPAR

# %%
# # #!/bin/bash


# # Download TRANSFAC format of JAPSAR data (CORE, Vertebrate, non-redudant)
# wget https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_transfac.txt
# wget https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_transfac.zip
# unzip JASPAR2024_CORE_non-redundant_pfms_transfac.zip -d JASPAR2024_CORE_non-redundant_pfms_transfac

# # Config
# in_dir=JASPAR2024_CORE_non-redundant_pfms_transfac
# ofile=out/table_JASPAR2024_CORE_vertebrates_non-redundant_pfms_transfac.tsv
# ofile2=out/JASPAR2024_CORE_vertebrates_non-redundant_pfms_transfac_uniprot_id.txt

# # mkdir -p out

# # Make JASPAR accession - uniprot_id table
# # echo -e "jaspar_matrix\tjaspar_id\tuniprot_id" > $ofile

# for transfac_file in `ls $in_dir`
# do
#     AC=`awk '/^AC/{print $2}' ${in_dir}/${transfac_file}`
#     ID=`awk '/^ID/{print $2}' ${in_dir}/${transfac_file}`
#     UNI=`awk '/uniprot_ids/{gsub("CC uniprot_ids:", "", $0); print $0}' ${in_dir}/${transfac_file}`

#     UNI_array=($(echo $UNI | tr "; " "\n"))

#     for UNIPROT in "${UNI_array[@]}"
# 	do
# 	    echo -e "$AC\t$ID\t$UNIPROT" >> $ofile
# 	    echo "$UNIPROT" >> $ofile2
# 	done    
# done

# %%
organism = "hs"
motifs_folder = chd.get_output() / "data" / "motifs" / organism / "jaspar2024"
motifs_folder.mkdir(parents=True, exist_ok=True)

# %%
import gimmemotifs
pwms_raw = gimmemotifs.motif.read_motifs(
    str(motifs_folder / "JASPAR2024_CORE_non-redundant_pfms_transfac.txt"), fmt = "transfac", as_dict=True
)

# %%
motifs = pd.DataFrame({"full_id":pwms_raw.keys()})
motifs["motif"] = motifs["full_id"].str.split("_").str[0]
motifs = motifs.set_index("motif")
motifs2 = pd.read_table(motifs_folder / "out" / "table_JASPAR2024_CORE_vertebrates_non-redundant_pfms_transfac.tsv").rename(columns={"jaspar_matrix":"motif"}).groupby("motif").agg({"uniprot_id":lambda x: x}).reset_index().set_index("motif")
motifs = motifs.join(motifs2)

# %%
biomart_dataset = chd.biomart.Dataset()
mapping_human = biomart_dataset.get_batched(
    [
        biomart_dataset.attribute("ensembl_gene_id"),
        biomart_dataset.attribute("external_gene_name"),
        biomart_dataset.attribute("uniprotswissprot"),
    ],
    filters=[
        biomart_dataset.filter("uniprotswissprot", value=motifs.dropna()["uniprot_id"].explode().unique()),
    ],
    batch_size = 500,
)
mapping_human = mapping_human.groupby(["uniprotswissprot"]).first().reset_index().rename(columns = {"ensembl_gene_id":"human_ensembl_gene_id", "external_gene_name":"human_gene_name"})
mapping_human["mouse_ensembl_gene_id"] = chd.biomart.get_orthologs(biomart_dataset, mapping_human["human_ensembl_gene_id"], organism = "mmusculus")

# %%
biomart_dataset = chd.biomart.Dataset("mmusculus_gene_ensembl")
mapping_mouse = biomart_dataset.get_batched(
    [
        biomart_dataset.attribute("ensembl_gene_id"),
        biomart_dataset.attribute("external_gene_name"),
        biomart_dataset.attribute("uniprotswissprot"),
    ],
    filters=[
        biomart_dataset.filter("uniprotswissprot", value=motifs.dropna()["uniprot_id"].explode().unique()),
    ],
    batch_size = 500,
)
mapping_mouse = mapping_mouse.groupby(["uniprotswissprot"]).first().reset_index().rename(columns = {"ensembl_gene_id":"mouse_ensembl_gene_id", "external_gene_name":"mouse_gene_name"})
mapping_mouse["human_ensembl_gene_id"] = chd.biomart.get_orthologs(biomart_dataset, mapping_mouse["mouse_ensembl_gene_id"], organism = "hsapiens")

# %%
mapping = pd.concat([mapping_human, mapping_mouse]).set_index("uniprotswissprot")

assert not mapping.index.duplicated().any()


# %%
def convert(x):
    x = x.unique()
    x = [y for y in x if not pd.isnull(y)]
    if len(x) == 0:
        return None
    elif len(x) == 1:
        return x[0]
    return x


# %%
motifs = motifs.explode("uniprot_id").join(mapping, on = "uniprot_id").groupby("motif").agg({"human_ensembl_gene_id":convert, "mouse_ensembl_gene_id":convert, "uniprot_id":"first", "human_gene_name":convert, "mouse_gene_name":convert})
motifs = motifs.loc[motifs["human_ensembl_gene_id"].notnull() | motifs["mouse_ensembl_gene_id"].notnull()]

# %%
motifs.to_csv(motifs_folder / "motifs.tsv", sep = "\t")

# %%
for motif in motifs.index:
    pass

# %%
pwms_raw2 = {k.split("_")[0]:v for k,v in pwms_raw.items()}

# %%
pwms = {}
for motif in motifs.index:
    pwms[motif] = pwms_raw2[motif].logodds

# %%
motifs["cutoff"] = 4.

# %%
pickle.dump(pwms, open(motifs_folder / "pwms.pkl", "wb"))

# %%
organism = "mm"
motifs_folder2 = chd.get_output() / "data" / "motifs" / organism / "jaspar2024"
motifs_folder2.mkdir(parents=True, exist_ok=True)

pickle.dump(pwms, open(motifs_folder2 / "pwms.pkl", "wb"))
motifs.to_csv(motifs_folder2 / "motifs.tsv", sep = "\t")

# %% [markdown]
# ## Full genome hocomoco

# %%
# genome = "GRCh38"
# organism = "hs"

genome = "mm10"
organism = "mm"

# genome = "GRCm39"
# organism = "mm"

# %%
genome_folder = pathlib.Path(f"/srv/data/genomes/{genome}")

# %%
fasta_file = f"/srv/data/genomes/{genome}/{genome}.fa"
chromosomes_file = f"/srv/data/genomes/{genome}/{genome}.fa.sizes"

regions = chd.data.Regions.from_chromosomes_file(chromosomes_file, path = genome_folder / "regions")
# regions = regions.filter(["chr22"])

# %%
region_onehots = chd.data.motifscan.motifscan.create_region_onehots(regions, fasta_file)

# %%
motifs_folder = chd.get_output() / "data" / "motifs" / organism / "hocomocov12"
motifs_folder.mkdir(parents=True, exist_ok=True)

pwms, motifs = chd.data.motifscan.download.get_hocomoco(motifs_folder, organism = organism)

# %%
# # !ls -la {chd.data.Motifscan(genome_folder / "motifscans" / motifscan_name).path}

# %%
motifscan_name = "hocomocov12_1e-4";cutoff_col="cutoff_0.0001"
# motifscan_name = "hocomocov12_5e-4";cutoff_col="cutoff_0.0005"
motifscan_name = "hocomocov12_5";cutoff_col="cutoff_5"
motifs["cutoff_5"] = 5
motifscan = chd.data.Motifscan.from_pwms(
    pwms,
    regions,
    motifs=motifs,
    region_onehots=region_onehots,
    cutoff_col=cutoff_col,
    min_cutoff = 3,
    fasta_file=fasta_file,
    path=genome_folder / "motifscans" / motifscan_name,
    device = "cuda:0",
    overwrite = True,
)

# %%
motifscan.create_region_indptr()

# %%
motifscan.create_indptr()

# %%
counts = np.bincount(motifscan.indices[:])

# %%
pd.DataFrame({
    "motif": motifs.index,
    "count": counts,
    "cutoff": motifs[cutoff_col],
}).sort_values("count", ascending = False).tail(50)

# %%
np.diff(motifscan.region_indptr[:])

# %%
chd.data.motifscan.Motifscan(motifscan.path)

# %%


# %% [markdown]
# ## Full genome jaspar

# %%
genome = "GRCh38"
organism = "hs"

genome = "mm10"
organism = "mm"

# %%
genome_folder = chd.get_output() / "genomes" / genome

# %%
fasta_file = f"/data/genome/{genome}/{genome}.fa"
chromosomes_file = f"/data/genome/{genome}/{genome}.fa.sizes"

regions = chd.data.Regions.from_chromosomes_file(chromosomes_file, path = genome_folder / "regions")
# regions = regions.filter(["chr22"])

# %%
region_onehots = chd.data.motifscan.motifscan.create_region_onehots(regions, fasta_file)

# %%
import gzip
motifs_folder = chd.get_output() / "data" / "motifs" / organism / "jaspar2024"
motifs_folder.mkdir(parents=True, exist_ok=True)

motifs = pd.read_table(motifs_folder / "motifs.tsv").set_index("motif")
motifs["cutoff"] = 4.5
pwms = pickle.load(open(motifs_folder / "pwms.pkl", "rb"))
# pwms, motifs = chd.data.motifscan.download.get_hocomoco(motifs_folder, organism = organism)
# motifs = motifs.iloc[:5]
# pwms = {k: v for k, v in pwms.items() if k in motifs.index}

# %%
# # !ls -la {chd.data.Motifscan(genome_folder / "motifscans" / motifscan_name).path}

# %%
motifscan_name = "jaspar2024_4.5"
motifscan = chd.data.Motifscan.from_pwms(
    pwms,
    regions,
    motifs=motifs,
    region_onehots=region_onehots,
    cutoff_col="cutoff",
    min_cutoff = 3,
    fasta_file=fasta_file,
    path=genome_folder / "motifscans" / motifscan_name,
    device = "cuda:0",
    overwrite = True,
)

# motifscan.create_region_indptr()
# motifscan.create_indptr()

# %%
motifscan.create_region_indptr()

# %%
counts = np.bincount(motifscan.indices[:])

# %%
pd.DataFrame({
    "motif": motifs.index,
    "count": counts,
    "cutoff": motifs["cutoff_0.0001"],
}).sort_values("count", ascending = False).tail(50)

# %%
np.diff(motifscan.region_indptr[:])

# %%
motifscan = chd.data.motifscan.Motifscan(genome_folder / "motifscans" / motifscan_name)

# %%
motifscan.motifs = motifs

# %%
motifscan.create_indptr()

# %% [markdown]
# ## View of dataset

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "pbmc10k_gran"
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "pbmc10kx"
# dataset_name = "hspc_cycling"
dataset_name = "hspc_meg_cycling"
dataset_name = "hspc_gmp_cycling"
genome = "GRCh38"
organism = "hs"

# dataset_name = "liverphx"
# genome = "mm10"
# organism = "mm"

folder_data_preproc = folder_data / dataset_name

# %%
motifscan_name = "hocomocov12_1e-4"
# motifscan_name = "jaspar2024_4.5"

# %%
regions_name = "100k100k"
# regions_name = "10k10k"
regions = chd.data.Regions(chd.get_output() / "datasets" / dataset_name / "regions" / regions_name)

# %%
genome_folder = chd.get_output() / "genomes" / genome
parent = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)

# %%
motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
    parent, 
    regions,
    path = chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
)

# %% [markdown]
# ## Full dataset

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

# dataset_name = "pbmc10k"
dataset_name = "pbmc10k/subsets/top250"
genome = "GRCh38"
organism = "hs"

folder_data_preproc = folder_data / dataset_name

# %%
fasta_file = "/data/genome/GRCh38/GRCh38.fa"
chromosomes_file = "/data/genome/GRCh38/GRCh38.fa.sizes"

regions = chd.data.Regions.from_chromosomes_file(chromosomes_file, path = genome_folder / "regions")
# regions = regions.filter(["chr22"])

# %%
regions_name = "100k100k"
# regions_name = "10k10k"
regions = chd.data.Regions(chd.get_output() / "datasets" / dataset_name / "regions" / regions_name)

# %%
import gzip
motifs_folder = chd.get_output() / "data" / "motifs" / organism / "hocomocov12"
motifs_folder.mkdir(parents=True, exist_ok=True)

pwms, motifs = chd.data.motifscan.download.get_hocomoco(motifs_folder, organism = organism)
# motifs = motifs.iloc[:5]
# pwms = {k: v for k, v in pwms.items() if k in motifs.index}

# %%
motifscan_name = "hocomocov12_1e-4_full"
motifscan = chd.data.Motifscan.from_pwms(
    pwms,
    regions,
    motifs=motifs,
    cutoff_col="cutoff_0.0001",
    min_cutoff = 3,
    fasta_file=fasta_file,
    path = chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name,
    device = "cuda:0",
    overwrite = True,
)

# motifscan.create_region_indptr()
# motifscan.create_indptr()

# %%
motifscan.create_region_indptr()

# %%
motifscan.create_indptr()

# %%
motifscan
