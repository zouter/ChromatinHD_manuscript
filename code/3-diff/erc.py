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
import crispyKC

crispyKC.setup_ipython()

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

import torch

import tqdm.auto as tqdm

import chromatinhd as chd
import polyptich

# %%
import chromatinhd as chd
import tempfile
# %%
fs4_folder = pathlib.Path("/home/wouters/fs4").resolve()
folder = fs4_folder / "exp/units/u_mgu/private/JeanFrancois/Epigenetic_data/Stijn_CUT_and_RUN_IRF4_AM_IL33/Processed_data/Bedgraph"

# %%
!ls {folder}

# %%
import pyBigWig
import chromatinhd as chd

# %%
# pip install pyBigWig pyBedGraph PyRanges

# %%
genome_folder = pathlib.Path("/srv/data/genomes/GRCm39")

# %%
biomart_dataset = chd.biomart.Dataset.from_genome("GRCm39")
genes = chd.biomart.get_canonical_transcripts(biomart_dataset, symbols = ["Ccl17", "Ccl24", "Arg1", "Mmp12", "Ocstamp", "Dcstamp", "Cdh5"])

# %%
!wget https://raw.githubusercontent.com/ENCODE-DCC/kentUtils/v302.1.0/bin/linux.x86_64/bedGraphToBigWig -O bedGraphToBigWig
!chmod +x bedGraphToBigWig

# %%
design = pd.DataFrame({
    "sample": ["Clean_AM_IRF4_IL33_rep1", "Clean_AM_IRF4_IL33_rep2", "Clean_AM_IRF4_PBS_rep1", "Clean_AM_IRF4_PBS_rep2"],
    "condition": ["IL33", "IL33", "PBS", "PBS"],
    "replicate": [1, 2, 1, 2],
})

# %%
# convert to bigwig
# although pyBedGraph exists, it doesn't have the same functionality as pyBigWig, so we convert to bw
for sample in design["sample"]:
    bedgraph = folder / f"{sample}.bedgraph"
    if not bedgraph.exists():
        raise FileNotFoundError(bedgraph)
    if not pathlib.Path(f"{sample}.bw").exists():
        os.system(f"bedGraphToBigWig {bedgraph} {genome_folder / 'GRCm39.fa.sizes'} ./{sample}.bw")

# %%
design["bw"] = [pyBigWig.open(f"{sample}.bw") for sample in design["sample"]]

# %%
# symbol = "Cdh5"
symbol = "Ccl24"
symbol = "Mmp12"
symbol = "Arg1"
gene_oi = genes.index[genes["symbol"] == symbol][0]
gene = genes.loc[gene_oi]
region = gene.copy()
region["start"] = region["tss"] - 30000
region["end"] = region["tss"] + 30000

# %%
motifscan = chd.data.motifscan.Motifscan(chd.get_output() / "genomes/GRCm39/motifscans/hocomocov12_1e-4")

# %%
motifs_oi = motifscan.motifs.loc[[
    *motifscan.motifs.index[motifscan.motifs.index.str.contains("IRF2")][:1],
    # *motifscan.motifs.index[motifscan.motifs.HUMAN_gene_symbol.str.contains("SPI1")],
    # *motifscan.motifs.index[motifscan.motifs.HUMAN_gene_symbol.str.contains("RBPJ")],
    # *motifscan.motifs.index[motifscan.motifs.HUMAN_gene_symbol.str.contains("NR1H3")],
]]
motifs_oi["group"] = motifs_oi["HUMAN_gene_symbol"]

# %%
width = 10
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

panel_genes = chd.plot.genome.genes.Genes.from_region(
    region, genome="GRCm39", width = width, only_canonical=True
)
fig.main.add_under(panel_genes)

step = 10
x = np.arange(region["start"], region["end"], step = step)
ymax = 0
panels = []
for _, sample_info in design.iterrows():
    bw = sample_info["bw"]

    panel, ax = fig.main.add_under(
        polyptich.grid.Panel((width, 0.5)),
    )
    values = np.array(bw.values(region["chrom"], region["start"], region["end"]))[::step]
    ax.fill_between(x, values, color = "black", alpha = 0.9, lw = 0)
    ax.set_xlim(region["start"], region["end"])
    ax.set_ylim(0)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ymax = max(ymax, values.max())
    panels.append(panel)

    label = sample_info["condition"]

    ax.annotate(label, (0., 1.), xytext = (4, -4), xycoords = "axes fraction", textcoords = "offset points", ha = "left", va = "top", )

for panel, ax in panels:
    ax.set_ylim(0, ymax)

panel_motifs = chd.data.motifscan.plot.GroupedMotifs(
    motifscan, region["chrom"], motifs_oi = motifs_oi, window = region[["start", "end"]], width = width
)
fig.main.add_under(panel_motifs)

fig.display()


# %% [markdown]
# # Test motif scan

# %%
# motif_ixs = [motifscan.motifs.index.get_loc("SUH.H12CORE.0.P.B")]
motif_ixs = [motifscan.motifs.index.get_loc("IRF1.H12CORE.0.P.B")]
# motif_ixs = [motifscan.motifs.index.get_loc("IRF1.H12CORE.0.P.B")]
# motif_ixs = [motifscan.motifs.index.get_loc("SPI1.H12CORE.0.P.B")]
print(motifscan.motifs.iloc[motif_ixs]["consensus"])
coords, _, scores, strands = motifscan.get_slice(
    region_id = region["chrom"], start = region["start"], end = region["end"], motif_ixs = motif_ixs
)

coord_ix = 2
print(f"{region["chrom"]}:{coords[coord_ix]-10}-{coords[coord_ix]+10}")
print(scores[coord_ix], strands[coord_ix])

# %%
import pysam
fasta = pysam.FastaFile(genome_folder / "GRCm39.fa")

# %%
seq = fasta.fetch(region["chrom"], coords[coord_ix]-10, coords[coord_ix]+10)
seq
# %%
