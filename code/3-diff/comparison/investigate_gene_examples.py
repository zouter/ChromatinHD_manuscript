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
import IPython

if IPython.get_ipython():
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
    IPython.get_ipython().run_line_magic(
        "config", "InlineBackend.figure_format='retina'"
    )

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
from example import Example

chd.models.diff.plot.get_cmap_atac_diff()

# %%
# information on chip seq files
bed_folder = chd.get_output() / "bed" / "gm1282_tf_chipseq"
files_bed = pd.read_csv(bed_folder / "files.csv", index_col=0)
bw_folder = chd.get_output() / "bed" / "gm1282_tf_chipseq_bw"
files = pd.read_csv(bw_folder / "files.csv", index_col=0)

# %%
window = np.array([-10000, 10000])

# %%
motifscan_name = "gwas_immune"

# load qtl data
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on="snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

def get_plotdata_snps(association, promoter):
    association["pos"] = association["start"].astype(int)
    association_oi = association.loc[
        (association["chr"] == promoter.chr)
        & (association["pos"] >= promoter.start)
        & (association["pos"] <= promoter.end)
    ].copy()

    association_oi["location"] = (
        promoter.tss - association_oi["pos"]
    ) * promoter.strand
    plotdata_snps = association_oi
    plotdata_snps["position"] = -plotdata_snps["location"]
    plotdata_snps["rsid"] = plotdata_snps["snp"].astype(str)
    return plotdata_snps


# %%
fig_colorbar = plt.figure(figsize=(3.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=chd.models.diff.plot.get_norm_atac_diff(),
    cmap=chd.models.diff.plot.get_cmap_atac_diff(),
)
colorbar = plt.colorbar(
    mappable, cax=ax_colorbar, orientation="horizontal", extend="both"
)
colorbar.set_label("Differential accessibility")
colorbar.set_ticks(np.log([0.25, 0.5, 1, 2, 4]))
colorbar.set_ticklabels(["¼", "½", "1", "2", "4"])
manuscript.save_figure(fig_colorbar, "3", "colorbar_atac_diff")

# %%
fig_colorbar = plt.figure(figsize=(0.1, 1.0))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=chd.models.diff.plot.get_norm_atac_diff(),
    cmap=chd.models.diff.plot.get_cmap_atac_diff(),
)
colorbar = plt.colorbar(
    mappable,
    cax=ax_colorbar,
    extend="both",
    orientation="vertical",
)
ax_colorbar.set_title("Differential\naccessibility", va="bottom", rotation=0, ha="center", fontsize = 10)
colorbar.set_ticks(np.log([0.25, 0.5, 1, 2, 4]))
colorbar.set_ticklabels(["¼", "½", "1", "2", "4"])
manuscript.save_figure(fig_colorbar, "3", "colorbar_atac_diff_vertical")

# %%
fig_colorbar = plt.figure(figsize=(0.1, 1.0))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=0, vmax=1),
    cmap=chd.models.diff.plot.get_cmap_rna_diff(),
)
colorbar = plt.colorbar(
    mappable,
    cax=ax_colorbar,
    orientation="vertical",
    ticks=[0, 1],
)
ax_colorbar.set_title("Relative\nexpression", va="bottom", rotation=0, ha="center", fontsize = 10)
# colorbar.set_ticks(np.log([0, 0.5]))
colorbar.set_ticklabels(["0", "max"])
manuscript.save_figure(fig_colorbar, "3", "colorbar_rna_diff_vertical")

# %%
fig_colorbar = plt.figure(figsize=(3.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=0, vmax=1),
    cmap=chd.models.diff.plot.get_cmap_rna_diff(),
)
colorbar = plt.colorbar(
    mappable,
    cax=ax_colorbar,
    orientation="horizontal",
    ticks=[0, 1],
)
colorbar.set_label("Relative expression")
# colorbar.set_ticks(np.log([0, 0.5]))
colorbar.set_ticklabels(["0", "max"])
manuscript.save_figure(fig_colorbar, "3", "colorbar_rna_diff")

# %% [markdown]
# ## HLADQA Lymphoma

# %%
motifs_to_merge = []
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "HLA-DQA1",
    motifs_to_merge,
    subset_clusters=["cDCs"],
    show_motifs = False,
)


# plotdata_snps = get_plotdata_snps(association, example.promoter)
# if len(plotdata_snps):
#     example.fig.main.add_under(
#         chdm.plotting.gwas.SNPs(
#             plotdata_snps,
#             example.panel_width,
#             window = window,
#         ),
#         column = example.wrap_differential
#     )


design = [
    {
        # "file":bw_folder / files.loc[files["experiment_target"].str.contains("IRF1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        "tf":"IRF1",
        "motif_identifiers":["IRF1"],
        # "gene":example.transcriptome.gene_id("IRF1")
    },
]
design = pd.DataFrame(design)
example.add_bigwig(design)
example.fig.plot()
example.fig.show()

# %%
print(
    f"{example.promoter['chr']}:{example.promoter['start']}-{example.promoter['end']}"
)

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %% [markdown]
# ## BCL2 Lymphoma

# %%
motifs_to_merge = []
example = Example(
    "lymphoma",
    "10k10k",
    "celltype",
    "v9_128-64-32",
    "cutoff_0001",
    "BCL2",
    motifs_to_merge,
    subset_clusters=["B", "Lymphoma cycling"],
    show_motifs = False,
)


plotdata_snps = get_plotdata_snps(association, example.promoter)
if len(plotdata_snps):
    example.fig.main.add_under(
        chdm.plotting.gwas.SNPs(
            plotdata_snps,
            example.panel_width,
            window = window,
        ),
        column = example.wrap_differential
    )


design = [
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        "tf":"POU2F2",
        "motif_identifiers":["PO2F2"],
        # "gene":example.transcriptome.gene_id("POU2F2")
    },
]
design = pd.DataFrame(design)
example.add_bigwig(design)
example.fig.plot()
example.fig.show()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %%
motifdata = example.motifdata.query("motif == 'PO2F2_HUMAN.H11MO.0.A'").copy()
motifdata["start"] = motifdata["position"] - 100
motifdata["end"] = motifdata["position"] + 100
motifdata["start_genome"] = (
    example.promoter["tss"]
    + motifdata["start"] * (example.promoter["strand"] == 1)
    + motifdata["end"] * (example.promoter["strand"] == -1) * -1
)
motifdata["end_genome"] = (
    example.promoter["tss"]
    + motifdata["end"] * (example.promoter["strand"] == 1)
    + motifdata["start"] * (example.promoter["strand"] == -1) * -1
)
motifdata["chr"] = example.promoter["chr"]

# %%
import pyBigWig

bw = pyBigWig.open(
    str(chd.get_output() / "data" / "lymphoma" / "atac_cut_sites.bigwig")
)

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

for _, row in motifdata.iterrows():
    panel, ax = fig.main.add(chd.grid.Panel((2, 1)))

    ax.plot(
        bw.values(row["chr"], row["start_genome"], row["end_genome"]),
        color="k",
        linewidth=0.5,
    )
    ax.set_title(row["position"])
    print(f"{row['chr']}:{row['start_genome']}-{row['end_genome']}")
    ax.set_ylim(0)
fig.plot()

# %%
# -----------
motifs_to_merge = []
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "IRF7",
    motifs_to_merge,
    subset_clusters=["pDCs", "Monocytes"],
)

import pyBigWig
import pybedtools

file_oi = files.loc[files["experiment_target"].str.contains("BCL11A")].iloc[0]
# file_oi = files.loc[files["experiment_target"].str.contains("SPI1")].iloc[0]
# file_oi = files.loc[files["experiment_target"].str.contains("POU2F2")].iloc[0]
# file_oi = files.loc[files["experiment_target"].str.contains("TCF3")].iloc[0]
filename = bed_folder / file_oi["filename"]

# !wget https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE128834.SPI1.THP-1.bed.gz
# filename = "GSE128834.SPI1.THP-1.bed.gz"
# !wget https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/ENCSR145TSJ.ATF4.K-562.bed.gz
# filename = "ENCSR145TSJ.ATF4.K-562.bed.gz"
# !wget https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE104676.BCL11A.CD34_Day3_30min.bed.gz
# filename = "GSE104676.BCL11A.CD34_Day3_30min.bed.gz"
# !wget https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE76147.TCF4.GEN2-2.bed.gz
filename = "GSE76147.TCF4.GEN2-2.bed.gz"

# bw = pyBigWig.open(str(bed_folder / file_oi["filename"]))
bed = pybedtools.BedTool(str(filename))

example.add_bed(bed)
example.fig.plot()
example.fig.show()

# %% [markdown]
# ## QKI

# %%
motifs_to_merge = []
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "QKI",
    subset_clusters=["Monocytes"],
)

plotdata_snps = get_plotdata_snps(association, example.promoter)
if len(plotdata_snps):
    example.fig.main.add_under(
        chdm.plotting.gwas.SNPs(
            plotdata_snps,
            example.panel_width,
            window = window,
        ),
        column = example.wrap_differential
    )

file = "ENCFF891VJC.bigWig"
if not (bw_folder / file).exists():
    # !wget https://www.encodeproject.org/files/ENCFF891VJC/@@download/ENCFF891VJC.bigWig -O {bw_folder / file}

file = "ENCFF248TTI.bigWig"
if not (bw_folder / file).exists():
    # !wget https://www.encodeproject.org/files/ENCFF248TTI/@@download/ENCFF248TTI.bigWig -O {bw_folder / file}

design = [
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        "tf":"SPI1",
        "motif_identifiers":["SPI1"]
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"ATF4",
        "motif_identifiers":["ATF4"],
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"CEBPB",
        "motif_identifiers":["CEBPB"],
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"NFE2L2",
        "motif_identifiers":["NF2L2"],
    },
]
design = pd.DataFrame(design)

example.add_bigwig(design, show_peaks=True)
example.fig.plot()
example.fig.show()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %%
print(
    f"{example.promoter['chr']}:{example.promoter['start']}-{example.promoter['end']}"
)

# %%
motifdata = example.motifdata.query("motif == 'PO2F2_HUMAN.H11MO.0.A'").copy()
motifdata = example.motifdata.query("motif == 'SPI1_HUMAN.H11MO.0.A'").copy()
motifdata["start"] = motifdata["position"] - 100
motifdata["end"] = motifdata["position"] + 100
motifdata["start_genome"] = (
    example.promoter["tss"]
    + motifdata["start"] * (example.promoter["strand"] == 1)
    + motifdata["end"] * (example.promoter["strand"] == -1) * -1
)
motifdata["end_genome"] = (
    example.promoter["tss"]
    + motifdata["end"] * (example.promoter["strand"] == 1)
    + motifdata["start"] * (example.promoter["strand"] == -1) * -1
)
motifdata["chr"] = example.promoter["chr"]

# %%
import pyBigWig

bw = pyBigWig.open(
    # str(chd.get_output() / "data" / "lymphoma" / "atac_cut_sites.bigwig")
    str(chd.get_output() / "data" / "pbmc10k" / "atac_cut_sites.bigwig")
)

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

for _, row in motifdata.iterrows():
    panel, ax = fig.main.add(chd.grid.Panel((2, 1)))

    ax.plot(
        bw.values(row["chr"], row["start_genome"], row["end_genome"]),
        color="k",
        linewidth=0.5,
    )
    ax.set_title(row["position"])
    print(f"{row['chr']}:{row['start_genome']}-{row['end_genome']}")
    ax.set_ylim(0)
fig.plot()


# %% [markdown]
# ## NKG7

# %%
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "NKG7",
    subset_clusters=["NK"],
)

plotdata_snps = get_plotdata_snps(association, example.promoter)
if len(plotdata_snps):
    example.fig.main.add_under(
        chdm.plotting.gwas.SNPs(
            plotdata_snps,
            example.panel_width,
            window = window,
        ),
        column = example.wrap_differential
    )

design = [
     {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("RUNX3")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        "tf":"RUNX3",
        "motif_identifiers":["RUNX3", "PEBB", "RUNX2"]
    },
     {
        # "file":bw_folder / files.loc[files["experiment_target"].str.contains("TBX21")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        "tf":"TBX21",
        "motif_identifiers":["TBX21"]
    },
]
design = pd.DataFrame(design)

example.add_bigwig(design, show_peaks=True)
example.fig.plot()
example.fig.show()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %%
print(
    f"{example.promoter['chr']}:{example.promoter['start']}-{example.promoter['end']}"
)

# %% [markdown]
# ## PAX5 Pmbcs

# %%
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "PAX5",
    subset_clusters=["B"],
)

plotdata_snps = get_plotdata_snps(association, example.promoter)
if len(plotdata_snps):
    example.fig.main.add_under(
        chdm.plotting.gwas.SNPs(
            plotdata_snps,
            example.panel_width,
            window = window,
        ),
        column = example.wrap_differential
    )

design = [
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        "tf":"POU2F2",
        "motif_identifiers":["PO2F2"],
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("TCF3")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("TCF3")].iloc[0]["filename"],
        "tf":"TCF3",
        "motif_identifiers":["TFE2"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        "tf":"SPI1",
        "motif_identifiers":["SPI1"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("EBF1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("EBF1")].iloc[0]["filename"],
        "tf":"EBF1",
        "motif_identifiers":["COE1"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("IRF4")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("IRF4")].iloc[0]["filename"],
        "tf":"IRF4",
        "motif_identifiers":["IRF4"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("PAX5")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("PAX5")].iloc[0]["filename"],
        "tf":"PAX5",
        "motif_identifiers":["PAX5"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("ZEB1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("ZEB1")].iloc[0]["filename"],
        "tf":"ZEB1",    
        "motif_identifiers":["ZEB1"]
    },
    {
        "tf":"SNAI1",
        "motif_identifiers":["SNAI1"] 
    },
]
design = pd.DataFrame(design)
example.add_bigwig(design)
example.fig.plot()
example.fig.show()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %% [markdown]
# ## AAK1 Pmbcs

# %%
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "AAK1",
    subset_clusters=["CD4 T"],
)

plotdata_snps = get_plotdata_snps(association, example.promoter)
if len(plotdata_snps):
    example.fig.main.add_under(
        chdm.plotting.gwas.SNPs(
            plotdata_snps,
            example.panel_width,
            window = window,
        ),
        column = example.wrap_differential
    )

design = [
    {
        # "file":bw_folder / files.loc[files["experiment_target"].str.contains("TCF7")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        "tf":"TCF7",
        "motif_identifiers":["TF7L1", "TCF7", "TF7L2"],
    },
    {
        # "file":bw_folder / files.loc[files["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        "tf":"GATA3",
        "motif_identifiers":["GATA6"],
    },
    {
        # "file":bw_folder / files.loc[files["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        "tf":"ZEB1",
        "motif_identifiers":["ZEB1"],
    },
]
design = pd.DataFrame(design)
example.add_bigwig(design)
example.fig.plot()
example.fig.show()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %% [markdown]
# ## CD74 Pmbcs

# %%
motifs_to_merge = []
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    "CD74",
    motifs_to_merge,
    subset_clusters=["B", "Monocytes"],
    show_motifs = False,
)


plotdata_snps = get_plotdata_snps(association, example.promoter)
if len(plotdata_snps):
    example.fig.main.add_under(
        chdm.plotting.gwas.SNPs(
            plotdata_snps,
            example.panel_width,
            window = window,
        ),
        column = example.wrap_differential
    )

design = [
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("POU2F2")].iloc[0]["filename"],
        "tf":"POU2F2",
        "motif_identifiers":["PO2F2"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("TCF3")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("TCF3")].iloc[0]["filename"],
        "tf":"TCF3",
        "motif_identifiers":["TFE2"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        "tf":"SPI1",
        "motif_identifiers":["SPI1"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("EBF1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("EBF1")].iloc[0]["filename"],
        "tf":"EBF1",
        "motif_identifiers":["COE1"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("IRF4")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("IRF4")].iloc[0]["filename"],
        "tf":"IRF4",
        "motif_identifiers":["IRF4"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("PAX5")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("PAX5")].iloc[0]["filename"],
        "tf":"PAX5",
        "motif_identifiers":["PAX5"]
    },
    {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("ZEB1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("ZEB1")].iloc[0]["filename"],
        "tf":"ZEB1",    
        "motif_identifiers":["ZEB1"]
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"ATF4",
        "motif_identifiers":["ATF4"],
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"CEBPB",
        "motif_identifiers":["CEBPB"],
    },
]
design = pd.DataFrame(design)
example.add_bigwig(design)
example.fig.plot()
example.fig.show()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %%
motifdata = example.motifdata.query("motif == 'PO2F2_HUMAN.H11MO.0.A'").copy()
motifdata = example.motifdata.query("motif == 'SPI1_HUMAN.H11MO.0.A'").copy()
motifdata["start"] = motifdata["position"] - 100
motifdata["end"] = motifdata["position"] + 100
motifdata["start_genome"] = (
    example.promoter["tss"]
    + motifdata["start"] * (example.promoter["strand"] == 1)
    + motifdata["end"] * (example.promoter["strand"] == -1) * -1
)
motifdata["end_genome"] = (
    example.promoter["tss"]
    + motifdata["end"] * (example.promoter["strand"] == 1)
    + motifdata["start"] * (example.promoter["strand"] == -1) * -1
)
motifdata["chr"] = example.promoter["chr"]

# %%
import pyBigWig

bw = pyBigWig.open(
    # str(chd.get_output() / "data" / "lymphoma" / "atac_cut_sites.bigwig")
    str(chd.get_output() / "data" / "pbmc10k" / "atac_cut_sites.bigwig")
)

# %%
fig = chd.grid.Figure(chd.grid.Wrap())

for _, row in motifdata.iterrows():
    panel, ax = fig.main.add(chd.grid.Panel((2, 1)))

    ax.plot(
        bw.values(row["chr"], row["start_genome"], row["end_genome"]),
        color="k",
        linewidth=0.5,
    )
    ax.set_title(row["position"])
    print(f"{row['chr']}:{row['start_genome']}-{row['end_genome']}")
    ax.set_ylim(0)
fig.plot()

# %% [markdown]
# ## ZAP70

# %%
symbol = "ZAP70"
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    symbol,
    subset_clusters=["NK"],
    show_motifs = False,
)

example.fig.main.add_under(
    chdm.plotting.gwas.SNPs(
        get_plotdata_snps(association, example.promoter),
        example.panel_width,
        window = window,
    ),
    column = example.wrap_differential
)

design = [
     {
        "file":bw_folder / files.loc[files["experiment_target"].str.contains("RUNX3")].iloc[0]["filename"],
        "tf":"RUNX3",
        "motif_identifiers":["RUNX3", "PEBB", "RUNX2", "RUNX1"]
    },
     {
        "tf":"TBX21",
        "motif_identifiers":["TBX21"]
    },
]
design = pd.DataFrame(design)
example.add_bigwig(design)

example.fig.plot()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %% [markdown]
# ## IGF2BP2

# %%
symbol = "IGF2BP2"
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    symbol,
    subset_clusters=["Monocytes"],
    show_motifs = False,
)

plotdata_snps = get_plotdata_snps(association, example.promoter)
example.fig.main.add_under(
    chdm.plotting.gwas.SNPs(
        plotdata_snps,
        example.panel_width,
        window = window,
    ),
    column = example.wrap_differential
)

design = [
    {
        "file":chd.get_output() / "bed" / "pu_mcp_chipseq" / "GSM3686967_THP-1_PU.1-ChIP.bigwig",
        # "file":chd.get_output() / "bed" / "pu_mcp_chipseq" / "GSM3686948_DC7d_PU.1-ChIP_donorK.bigwig",
        # "file":bw_folder / files.loc[files["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        # "file_bed":bed_folder / files_bed.loc[files_bed["experiment_target"].str.contains("SPI1")].iloc[0]["filename"],
        "tf":"SPI1",
        "motif_identifiers":["SPI1"]
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"ATF4",
        "motif_identifiers":["ATF4"],
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"CEBPB",
        "motif_identifiers":["CEBPB", "CEBPA", "CEBPD", "CEBPE"],
    },
    {
        # "file":bw_folder / "ENCFF891VJC.bigWig",
        # "file":bw_folder / "ENCFF248TTI.bigWig",
        # "file_bed":bw_folder/"ENCSR145TSJ.ATF4.K-562.bed.gz",
        "tf":"NFE2L2",
        "motif_identifiers":["COE1"],
    },
]
design = pd.DataFrame(design)
example.add_bigwig(design)
example.fig.plot()
example.fig.show()

# %%
manuscript.save_figure(example.fig, "3", f"examples/{example.symbol}_{example.dataset_name}")

# %% [markdown]
# ## CD22

# %%
symbol = "CD22"
example = Example(
    "pbmc10k",
    "10k10k",
    "leiden_0.1",
    "v9_128-64-32",
    "cutoff_0001",
    symbol,
    subset_clusters=["B"],
    show_motifs = False,
)

plotdata_snps = associations_genes[transcriptome.gene_id(symbol)]
plotdata_snps["position"] = -plotdata_snps["location"]
example.fig.main.add_under(
    chdm.plotting.gwas.SNPs(
        plotdata_snps,
        example.panel_width,
        window = window,
    ),
    column = example.wrap_differential
)
example.fig.plot()

# %%
