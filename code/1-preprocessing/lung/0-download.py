# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import polyptich as pp

import pysam

pp.setup_ipython()

# %%
pp.paths.results().mkdir(parents=True, exist_ok=True)

# %%
bam_folder = pathlib.Path(
    "/home/wouters/fs4/u_mgu/private/JeanFrancois/For_Wouter/Alveolar_Mac/ATAC/BAM/"
)

# %%
diffexp_folder = pathlib.Path(
    "/home/wouters/fs4/u_mgu/private/JeanFrancois/For_Wouter/Alveolar_Mac/RNA/"
)

# %%
diffexp = pd.read_excel(diffexp_folder / "DEG.xlsx")
diffexp

# %%
obs = pd.DataFrame(
    [
        ["Ctrl", "AM", 1, "Ctrl_AM1.bam"],
        ["Ctrl", "AM", 2, "Ctrl_AM2.bam"],
        ["Ctrl", "AM", 3, "Ctrl_AM3.bam"],
        ["IL33", "AM", 1, "IL33_AM1.bam"],
        ["IL33", "AM", 2, "IL33_AM2.bam"],
        ["IL33", "AM", 3, "IL33_AM3.bam"],
    ],
    columns=["treatment", "celltype", "replicate", "path"],
)

obs["path"] = (
    "/home/wouters/fs4/u_mgu/private/JeanFrancois/For_Wouter/Alveolar_Mac/ATAC/BAM/"
    + obs["path"]
)

obs["alignment"] = obs["path"].apply(lambda x: pysam.AlignmentFile(x, "rb"))

# %%
dataset_folder = chd.get_output() / "datasets" / "lung"

# %%
# import genomepy
# genomepy.install_genome("GRCm39", genomes_dir="/srv/data/genomes/")

fasta_file = "/srv/data/genomes/GRCm39/GRCm39.fa"
chromosomes_file = "/srv/data/genomes/GRCm39/GRCm39.fa.sizes"

regions_all = chd.data.Regions.from_chromosomes_file(
    chromosomes_file, path=dataset_folder / "regions" / "all"
)

fragments_all = chd.data.Fragments.from_alignments(
    obs,
    regions=regions_all,
    alignment_column="alignment",
    path=dataset_folder / "fragments" / "all",
    # overwrite=True,
    batch_size=10e7,
)

# %%
# get the TSS 
symbols = diffexp["gene"].tolist()
regions_name = "100k100k"
transcripts = chd.biomart.get_canonical_transcripts(
    chd.biomart.Dataset.from_genome("GRCm39"),
    filter_canonical=True,
    symbols=symbols,
)
transcripts.head()

# %%
# transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(gene_ids)]

window = [-100000, 100000]
regions = chd.data.Regions.from_transcripts(
    transcripts, window, path=dataset_folder / "regions" / regions_name, overwrite=True
)
regions.coordinates.head()

# %%
fragments = chd.data.fragments.FragmentsView.from_fragments(
    fragments_all,
    regions=regions,
    path=dataset_folder / "fragments" / regions_name,
    # overwrite=True,
)
fragments.create_regionxcell_indptr2(
    overwrite=False,
)
fragments.var["symbol"] = transcripts["symbol"]
fragments.var = fragments.var

# %% [markdown]
# ## Motifscan

# %%
dataset_name = "lung"

# %%
motifscan_name = "hocomocov12" + "_" + "1e-4"
# motifscan_name = "hocomocov12" + "_" + "5"
genome_folder = chd.get_output() / "genomes" / "GRCm39"
parent = chd.data.Motifscan(genome_folder / "motifscans" / motifscan_name)

motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
    parent,
    regions,
    path=chd.get_output()
    / "datasets"
    / dataset_name
    / "motifscans"
    / regions_name
    / motifscan_name,
)

# %%
parent.get_slice(region_id = "chr1", start = 5000000, end = 6000000)

# %%
parent.motifs

# %% [markdown]
# ## Clustering

# %%
obs["cluster"] = obs["treatment"] + "-" + obs["celltype"]
clustering = chd.data.Clustering.from_labels(
    obs["cluster"],
    var=obs.groupby("cluster").first()[["treatment", "celltype"]],
    path=dataset_folder / "clusterings" / "cluster",
    overwrite=True,
)

# %% [markdown]
# ## Training

# %%
fold = {
    "cells_train": np.arange(len(fragments.obs)),
    "cells_test": np.arange(len(fragments.obs)),
    "cells_validation": np.arange(len(fragments.obs)),
}

# %%
clustering = chd.data.Clustering(dataset_folder / "clusterings" / "cluster")
model_folder = (
    chd.get_output() / "diff" / "lung" / "binary" / "split" / regions_name / "cluster"
)


# %%
import chromatinhd.models.diff.model.binary

model = chd.models.diff.model.binary.Model.create(
    fragments,
    clustering,
    fold=fold,
    encoder="shared",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale=0.5,
        bias_regularization=True,
        bias_p_scale=0.5,
        binwidths=(5000, 1000, 500, 100, 50),
    ),
    path=model_folder / "model",
    # overwrite = True,
)

# %%
model.train_model(
    n_epochs=40, n_regions_step=50, early_stopping=False, do_validation=True, lr=1e-2
)

# %%
model.trace.plot()

# %%
model.save_state()

# %% [markdown]
# ## Inference

# %%
model = chd.models.diff.model.binary.Model(model_folder / "model")

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    # reset=True,
)

# %%
regionpositional.score(
    [model],
    fragments=fragments,
    clustering=clustering,
    device="cpu",
)
regionpositional

# %%
regionpositional.probs["ENSMUSG00000000001"].sel(cluster = "IL33-AM").plot()
regionpositional.probs["ENSMUSG00000000001"].sel(cluster = "Ctrl-AM").plot()
