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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import tempfile

# %%
import os

# %% [markdown]
# ## Data

# %% [markdown]
# ### Dataset

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.window = window

# %%
fragments.create_cut_data()

# %%
import pyranges as pr

# %%
all_regions = pr.PyRanges(pd.DataFrame({"Start":[0, 1000], "End":[1000, 2000], "Chromosome":["chr1", "chr2"]}))

# %%
region_sets = {
    "A":pr.PyRanges(pd.DataFrame({"Start":[0, 100000], "End":[100000, 200000], "Chromosome":["chr1", "chr2"]})), 
    "B":pr.PyRanges(pd.DataFrame({"Start":[1000000, 1100000], "End":[1100000, 1200000], "Chromosome":["chr1", "chr2"]}))
}

# %%
import pycistarget.motif_enrichment_dem

# %%
# # !wget https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/screen/mc_v10_clust/region_based/hg38_screen_v10_clust.regions_vs_motifs.scores.feather

# %%
import pycistarget.motif_enrichment_cistarget

# %%
results = pycistarget.motif_enrichment_cistarget.run_cistarget(
    "hg38_screen_v10_clust.regions_vs_motifs.scores.feather",
    region_sets = region_sets,
    specie = "homo_sapiens",
    auc_threshold = 0.005,
    nes_threshold = 3.0,
    rank_threshold = 0.05
)

# %%
results["A"].motif_hits

# %%
# Specify database:
feather_database_url='https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg19/refseq_r45/mc9nr/gene_based/hg19-500bp-upstream-7species.mc9nr.genes_vs_motifs.rankings.feather'
# feather_database_url='https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg19/refseq_r45/mc9nr/gene_based/hg19-500bp-upstream-10species.mc9nr.genes_vs_motifs.rankings.feather'
# feather_database_url='https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg19/refseq_r45/mc9nr/gene_based/hg19-tss-centered-5kb-7species.mc9nr.genes_vs_motifs.rankings.feather'
# feather_database_url='https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg19/refseq_r45/mc9nr/gene_based/hg19-tss-centered-5kb-10species.mc9nr.genes_vs_motifs.rankings.feather'
# feather_database_url='https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg19/refseq_r45/mc9nr/gene_based/hg19-tss-centered-10kb-7species.mc9nr.genes_vs_motifs.rankings.feather'
# feather_database_url='https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg19/refseq_r45/mc9nr/gene_based/hg19-tss-centered-10kb-10species.mc9nr.genes_vs_motifs.rankings.feather'
# Download database with zsync_curl:
!"${ZSYNC_CURL}" "${feather_database_url}.zsync"

# %%
# !wget https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg19/refseq_r45/mc9nr/gene_based/hg19-500bp-upstream-7species.mc9nr.genes_vs_motifs.rankings.feather -P {folder}

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
region_sets = {
    "A":pr.PyRanges(promoters.iloc[:100].rename(columns = {"start":"Start", "end":"End", "chr":"Chromosome"})),
    "B":pr.PyRanges(promoters.iloc[10:20].rename(columns = {"start":"Start", "end":"End", "chr":"Chromosome"}))
}

# %%
results = pycistarget.motif_enrichment_cistarget.run_cistarget(
    "hg38_screen_v10_clust.regions_vs_motifs.scores.feather",
    # str(folder / "hg19-500bp-upstream-7species.mc9nr.genes_vs_motifs.rankings.feather"),
    region_sets = region_sets,
    specie = "homo_sapiens",
    auc_threshold = 0.005,
    nes_threshold = 3.0,
    rank_threshold = 0.05
)

# %%
pd.read_feather(folder / "hg19-500bp-upstream-7species.mc9nr.genes_vs_motifs.rankings.feather")

# %%
results

# %% [markdown]
# ## Create motif DB

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
# !ls {folder_data_preproc}

# %%
promoters[["chr", "start", "end"]].sort_values(["chr", "start"]).to_csv(folder / "promoters.bed", index=False, header = False, sep = "\t")

# %%
!({folder}/create_cisTarget_databases/create_fasta_with_padded_bg_from_bed.sh \
{folder_data_preproc}/dna.fa.gz \
{folder_data_preproc}/chromosome.sizes \
{folder}/promoters.bed \
{folder}/output.bed \
10 \
)

# %%
# import tempfile
# tempdir = pathlib.Path(tempfile.mkdtemp())

# %%
# !pip install flatbuffers

# %%
folder = pfa.get_git_root() / "tmp"

# %%
# # !wget https://resources.aertslab.org/cistarget/motif_collections/v10nr_clust_public/v10nr_clust_public.zip -P {folder}/
# # !unzip {folder}/v10nr_clust_public.zip -d {folder}
# # !git clone https://github.com/aertslab/create_cisTarget_databases.git {folder}/create_cisTarget_databases

# %%
onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb"))

# %%
regions = pd.DataFrame({"gene_ix":np.random.randint(fragments.n_genes, size = 100), "start":0, "end":10000})

# %%
fasta_file = folder / "fasta.fa"
with open(fasta_file, "w") as fasta:
    for region_ix, region in regions.iterrows():
        fasta.write(f">{region_ix}\n")
        fasta.write("".join(np.array(["A", "C", "G", "T"])[np.where(onehot_promoters[region["gene_ix"], region["start"]:region["end"]])[1]]) + "\n")

# %%
# motif_ids = pd.read_feather(folder/"hg19-500bp-upstream-7species.mc9nr.genes_vs_motifs.rankings.feather")["motifs"]
motif_ids = list([f.name[:-3] for f in (folder / "v10nr_clust_public/singletons/").iterdir() if f.name[-2:] == "cb"])[:200]
motifs_file = folder / "motifs.txt"
with open(motifs_file, "w") as motifs:
    motifs.write("\n".join(motif_ids))

# %%
!(python {folder.resolve()}/create_cisTarget_databases/create_cistarget_motif_databases.py \
-f {folder.resolve()}/fasta.fa \
-M {folder.resolve()}/v10nr_clust_public/singletons/ \
-m  {folder.resolve()}/motifs.txt \
-o  {folder.resolve()}/out.feather \
-c /home/wsaelens/projects/peak_free_atac/software/cluster-buster/cbust \
) 2> out.log

# %%
# pd.read_feather("/home/wsaelens/projects/peak_free_atac/tmp/out.feather.regions_vs_motifs.rankings.feather")

# %%
region_sets = {
    "A":pr.PyRanges(regions.iloc[[0, 1]].rename(columns = {"start":"Start", "end":"End", "gene_ix":"Chromosome"})),
    "B":pr.PyRanges(regions.iloc[[2, 3]].rename(columns = {"start":"Start", "end":"End", "gene_ix":"Chromosome"}))
}
# region_sets = {
#     "A":pr.PyRanges(pd.DataFrame({"Start":[0, 100000], "End":[100000, 200000], "Chromosome":["chr1", "chr2"]})), 
#     "B":pr.PyRanges(pd.DataFrame({"Start":[1000000, 1100000], "End":[1100000, 1200000], "Chromosome":["chr1", "chr2"]}))
# }

# %%
db = pycistarget.motif_enrichment_cistarget.cisTargetDatabase(
    str(folder / "out.feather.regions_vs_motifs.rankings.feather"),
    # pr.PyRanges(regions.rename(columns = {"start":"Start", "end":"End", "gene_ix":"Chromosome"}))
)

# %%
# regions_to_db = {region_set_name:pd.DataFrame({"Query":np.arange(len(regions)).astype(str), "Target":np.arange(len(regions)).astype(str)}) for region_set_name in region_sets.keys()}
regions_to_db = {"A":pd.DataFrame({"Query":["0", "1"], "Target":["0", "1"]}), "B":pd.DataFrame({"Query":["2", "3"], "Target":["2", "3"]})}
db.regions_to_db = regions_to_db

# %%
results = pycistarget.motif_enrichment_cistarget.run_cistarget(
    db,
    
    region_sets = region_sets,
    name = "hi",
    
    specie = "homo_sapiens",
    auc_threshold = 0.05,
    nes_threshold = 3.0,
    rank_threshold = 0.05,
    n_cpu = 1
)

# %%
results["B"].motif_enrichment

# %%
# !ls {folder.resolve()}

# %%
str(folder/"out.feather")

# %%
regions_all = pr.concat(region_sets.values())

# %%
regions_all.to_bed(folder / "regions.bed")

# %%
# !ls {folder_data_preproc}/dna.fa.gz

# %%
# !{folder}/create_cisTarget_databases/create_fasta_with_padded_bg_from_bed.sh  

# %%
results = pycistarget.motif_enrichment_cistarget.run_cistarget(
    # "hg38_screen_v10_clust.regions_vs_motifs.scores.feather",
    str(folder/"out.feather.regions_vs_motifs.scores.feather"),
    
    region_sets = region_sets,
    
    specie = "homo_sapiens",
    auc_threshold = 0.005,
    nes_threshold = 3.0,
    rank_threshold = 0.05
)

# %% [markdown]
# ## Get regions

# %%
import peakfreeatac.peakcounts

# %%
peaks_name = "cellranger"

# %%
latent_name = "leiden_0.1"

latent_folder = folder_data_preproc / "latent"
latent = pd.read_pickle(latent_folder / (latent_name + ".pkl"))

# %%
peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)
adata_atac = sc.AnnData(peakcounts.counts.astype(np.float32), obs = fragments.obs, var = peakcounts.var)
sc.pp.normalize_total(adata_atac)
sc.pp.log1p(adata_atac)

adata_atac.obs["cluster"] = pd.Categorical(
    latent.columns[np.where(latent.values)[1]],
    categories = latent.columns
)
sc.tl.rank_genes_groups(adata_atac, "cluster")

# %%
regions = []
region_cluster_assignment = {}
for cluster_ix, cluster in enumerate(latent.columns):
    peakscores = sc.get.rank_genes_groups_df(adata_atac, group = cluster).rename(columns = {"names":"peak", "scores":"score"}).set_index("peak")
    peakscores_joined = peakcounts.peaks.join(peakscores, on = "peak").sort_values("score", ascending = False)

    positive = (peakscores_joined["logfoldchanges"] > 1.0) & (peakscores_joined["pvals_adj"] < 0.05)
    if positive.sum() < 20:
        positive = peakscores_joined.index.isin(peakscores_joined.index[:20])
    negative = ~positive

    position_slices = peakscores_joined.loc[positive, ["relative_start", "relative_end"]].values
    position_slices = position_slices - window[0]
    gene_ixs_slices = peakscores_joined.loc[positive, "gene_ix"].values
    
    df = pd.DataFrame({"gene_ix":gene_ixs_slices, "start":position_slices[:, 0], "end":position_slices[:, 1],})
    df["region"] = df["gene_ix"].astype(str) + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    
    regions.append(df)
    
    region_cluster_assignment[cluster] = df["region"].values
regions = pd.concat(regions).groupby("region").first()

# %% [markdown]
# ## Get db

# %%
onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb"))

# %%
fasta_file = folder / "fasta.fa"
with open(fasta_file, "w") as fasta:
    for region_ix, region in regions.iterrows():
        fasta.write(f">{region_ix}\n")
        fasta.write("".join(np.array(["A", "C", "G", "T"])[np.where(onehot_promoters[region["gene_ix"], region["start"]:region["end"]])[1]]) + "\n")

# %%
# motif_ids = pd.read_feather(folder/"hg19-500bp-upstream-7species.mc9nr.genes_vs_motifs.rankings.feather")["motifs"]
motif_ids = list([f.name[:-3] for f in (folder / "v10nr_clust_public/singletons/").iterdir() if f.name[-2:] == "cb"])[:1000]
motifs_file = folder / "motifs.txt"
with open(motifs_file, "w") as motifs:
    motifs.write("\n".join(motif_ids))

# %%
!(python {folder.resolve()}/create_cisTarget_databases/create_cistarget_motif_databases.py \
-f {folder.resolve()}/fasta.fa \
-M {folder.resolve()}/v10nr_clust_public/singletons/ \
-m  {folder.resolve()}/motifs.txt \
-o  {folder.resolve()}/out.feather \
-c /home/wsaelens/projects/peak_free_atac/software/cluster-buster/cbust \
) 2> out.log

# %%
# pd.read_feather("/home/wsaelens/projects/peak_free_atac/tmp/out.feather.regions_vs_motifs.rankings.feather")

# %%
region_sets = {cluster:pr.PyRanges(regions.loc[region_cluster_assignment[cluster]].rename(columns = {"start":"Start", "end":"End", "gene_ix":"Chromosome"})) for cluster in region_cluster_assignment.keys()}

# %%
db = pycistarget.motif_enrichment_cistarget.cisTargetDatabase(
    str(folder / "out.feather.regions_vs_motifs.rankings.feather"),
    # pr.PyRanges(regions.rename(columns = {"start":"Start", "end":"End", "gene_ix":"Chromosome"}))
)

# %%
regions_to_db = {cluster:pd.DataFrame({"Query":region_ids, "Target":region_ids}) for cluster, region_ids in region_cluster_assignment.items()}
db.regions_to_db = regions_to_db

# %%
results = pycistarget.motif_enrichment_cistarget.run_cistarget(
    db,
    
    region_sets = region_sets,
    name = "hi",
    
    specie = "homo_sapiens",
    auc_threshold = 0.005,
    nes_threshold = 3.0,
    rank_threshold = 0.05,
    n_cpu = 1
)

# %%
results["Monocytes"].motif_enrichment
