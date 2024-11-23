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

pp.setup_ipython()

# %%
pp.paths.results().mkdir(parents=True, exist_ok=True)

# %%
dataset_name = "glas_2019"
dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
regions_name = "100k100k"
regions = chd.data.Regions(dataset_folder / "regions" / regions_name)

# %%
bed_folder = chd.get_output() / "data" / "glas_2019" / "peaks"
peakcallers = pd.DataFrame(
    [
        [
            "Monocyte",
            bed_folder / "BloodLy6cHi_mouse3_ATAC_notx.bed",
        ],
        [
            "24h",
            bed_folder / "RLM_mouse106_ATAC_DT24h.bed",
        ],
        [
            "48h",
            bed_folder / "RLM_mouse195_ATAC_DT48h.bed",
        ],
        [
            "KC",
            bed_folder / "KC_mouse216_ATAC_PBS.bed",
        ],
    ],
    columns=["label", "path"],
)
peakcallers["label"] = peakcallers["label"].str.split("-").str[0]

# %%
motifscan_name = "hocomocov12" + "_" + "1e-4"
# motifscan_name = "hocomocov12" + "_" + "5"
genome_folder = pathlib.Path("/srv/data/genomes/mm10")
motifscan_genome = chd.flow.Flow.from_path(
    genome_folder / "motifscans" / motifscan_name
)

motifscan = chd.data.motifscan.MotifscanView(
    path=chd.get_output()
    / "datasets"
    / dataset_name
    / "motifscans"
    / regions_name
    / motifscan_name,
)
motifscan.parent = motifscan_genome

# %%
latent_name = "cluster"
# latent_name = "file"
clustering = chd.data.Clustering(dataset_folder / "clusterings" / latent_name)

# %%
fragments = chd.data.fragments.FragmentsView(
    dataset_folder / "fragments" / regions_name,
    # overwrite=True,
)
regions.var = fragments.var

# %%
model_folder = (
    chd.get_output()
    / "diff"
    / "glas_2019"
    / "binary"
    / "split"
    / regions_name
    / latent_name
)
model = chd.models.diff.model.binary.Model(
    model_folder / "model",
    # overwrite = True,
)

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    # reset=True,
)
regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %% [markdown]
# ## Differential slices and their enrichment

# %%
# this cutoff depends on your dataset
# typically 5% of the genome
slices = regionpositional.calculate_slices(5.5, step=25)
# check if its about 5%
((slices.end_position_ixs - slices.start_position_ixs) * slices.step).sum() / (
    regions.coordinates["end"] - regions.coordinates["start"]
).sum()

# %%
differential_slices = regionpositional.calculate_differential_slices(
    slices, fc_cutoff=1.5, expand=1
)

# %%
differential_slices.get_slice_scores(regions=regions, clustering=clustering)


# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
pd.DataFrame(
    {
        "chd": slicescores.groupby("cluster")["length"]
        .sum()
        .sort_values(ascending=False),
    }
)

# %% [markdown]
# ## Plot regions

# %%
# %%
motifs_oi = pd.concat(
    [
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["SPI1"])],
            "group":"SPI1",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["SUH"])],
            "group":"RBPJ",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["HEY1", "HES1"])],
            "group":"HEY1/HES1",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["NR1H3"])],
            "group":"NR1H3",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["RXRA"])],
            "group":"RXRA",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["PPARG"])],
            "group":"PPARG",
        }),
        pd.DataFrame(
            {
                "motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["SMAD4"])],
                "group": "SMAD4",
            }
        ),
        pd.DataFrame(
            {
                "motif": motifscan.motifs.index[
                    motifscan.motifs.tf.str.contains("SMAD")
                ],
                "group": "Any SMAD",
            }
        ),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["RREB1"])],
            "group":"RREB1",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.MOUSE_gene_symbol.isin(["Tcf7l1", "Tcf3"])],
            "group":"TCF7L1/TCF3",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.MOUSE_gene_symbol.isin(["Meis3"])],
            "group":"MEIS3",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.MOUSE_gene_symbol.isin(["Klf7"])],
            "group":"KLF7",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.MOUSE_gene_symbol.isin(["Spic"])],
            "group":"SPIC",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.MOUSE_gene_symbol.isin(["Mef2a"])],
            "group":"MEF2A",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["FOS"])],
            "group":"FOS",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["CTCF"])],
            "group":"CTCF",
        }),
        pd.DataFrame({
            "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["RELB", "NFKB1"])],
            "group":"RELB/NFKB",
        }),
    ]
).set_index("motif")
motif_groups = pd.DataFrame({"group": motifs_oi["group"].unique()}).set_index("group")
motif_groups["label"] = motif_groups.index
motif_groups["color"] = [
    mpl.colors.to_hex(mpl.colormaps["tab10"](i % 10)) for i in range(len(motif_groups))
]

# %%
cluster_info1 = clustering.cluster_info.loc[["blood_monocyte_0_WT_Ly6cHi", "liver_RLM_24_WT", "liver_RLM_48_WT", "liver_KC_final_WT"]]
cluster_info1["label"] = ["Monocyte", "24h", "48h", "KC"]

cluster_info2 = clustering.cluster_info.loc[["liver_KC_final_WT", "liver_KC_final_Smad4-KO"]]
cluster_info2["label"] = ["WT KC", "Smad4-KO KC"]

cluster_info3 = clustering.cluster_info.loc[["liver_KC_final_WT", "liver_KC_final_Nr1h3-KO_Clec4f+Tim4-", "liver_KC_final_Nr1h3-KO_Clec4f+Tim4+"]]
cluster_info3["label"] = ["WT KC", "Tim4- Nr1h3 KO", "Tim4+ Nr1h3 KO"]

# %%
dataset_name = "liver"
folder_dataset2 = chd.get_output() / "datasets" / dataset_name
model_folder2 = (
    chd.get_output() / "diff" / "liver" / "100k100k" / "cluster"
)
regionpositional2 = chd.models.diff.interpret.RegionPositional(
    model_folder2 / "scoring" / "regionpositional",
    # reset=True,
)
clustering2 = chd.data.Clustering(
    path=folder_dataset2 / "clusterings" / "cluster"
)
fragments2 = chd.data.Fragments(folder_dataset2 / "fragments" / "100k100k")
regionpositional2.clustering = clustering2
regionpositional2.regions = fragments2.regions

# %%
# symbol = "Clec1b"
symbol = "Cd40"
symbol = "Cxcl9"
symbol = "Cdh5"
# symbol = "Id3"
# symbol = "Itga9"
# symbol = "Clec4f"

if symbol not in regions.var["symbol"].tolist():
    raise ValueError(f"Symbol {symbol} not found in regions")
gene_id = regions.var.query("symbol == @symbol").index[0]

windows = regionpositional.select_windows(
    gene_id,
    prob_cutoff=6.5,
    differential_prob_cutoff=3.0,
    keep_tss=True,
)
breaking = pp.grid.Breaking(windows, resolution=4000, gap=0.03)

fig = pp.grid.Figure(pp.grid.Grid(padding_height=0.05, padding_width=0.05))

region = regions.coordinates.loc[gene_id]

panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    xticks=[-100000, 0, 100000],
    gene_overlap_padding=10000 if windows["length"].sum() > 20000 else 100000,
    show_others=False,
    only_canonical=False,
)
fig.main.add_under(panel_genes, padding=0.0)

panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    show_others=False,
    only_canonical=False,
    label_positions = False,
)
fig.main.add_under(panel_genes, padding=0.0)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional,
    cluster_info=cluster_info1,
    breaking=breaking,
    ylintresh=2000,
    ymax=10000,
    # relative_to=cluster_info.index[0],
    relative_to="previous",
    label_cluster="front",
    label_accessibility=False,
)
# panel_differential.add_differential_slices(differential_slices)
fig.main.add_under(panel_differential)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional,
    cluster_info=cluster_info2,
    breaking=breaking,
    ylintresh=2000,
    ymax=10000,
    # relative_to=cluster_info.index[0],
    relative_to="previous",
    label_cluster="front",
    label_accessibility=False,
)
# panel_differential.add_differential_slices(differential_slices)
fig.main.add_under(panel_differential)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional,
    cluster_info=cluster_info3,
    breaking=breaking,
    ylintresh=2000,
    ymax=10000,
    # relative_to=cluster_info.index[0],
    relative_to="previous",
    label_cluster="front",
    label_accessibility=False,
)
# panel_differential.add_differential_slices(differential_slices)
fig.main.add_under(panel_differential)

panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
    motifscan,
    gene_id,
    motifs_oi,
    breaking,
    group_info=motif_groups,
    slices_oi=differential_slices.get_slice_scores(
        clustering=clustering, regions=regions
    ),
    show_triangle=False,
)
fig.main.add_under(panel_motifs)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional2,
    cluster_info=regionpositional2.clustering.cluster_info.loc[["Mid Hepatocyte", "KC", "LSEC"]],
    breaking=breaking,
    # ylintresh=2000,
    ymax=10,
    ylintresh = 5,
    relative_to="Mid Hepatocyte",
    label_cluster="front",
    label_accessibility=False,
)
fig.main.add_under(panel_differential)

import chromatinhd.data.peakcounts

panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed(
    regions.coordinates.loc[gene_id],
    peakcallers=peakcallers,
    breaking=breaking,
    label_methods_side="left",
    row_height=0.4,
    label_rows="Peaks",
)
fig.main.add_under(panel_peaks)

fig.display()

# %%
motifdata = chd.data.motifscan.plot._process_grouped_motifs(
    gene_id, motifs_oi.query("group == 'Any SMAD'"), motifscan
)

stds = regionpositional.get_plotdata(gene_id)[0]["prob"].unstack().loc[["blood_monocyte_0_WT_Ly6cHi", "liver_RLM_24_WT"]].diff().iloc[1]
maxs = regionpositional.get_plotdata(gene_id)[0]["prob"].unstack().max()
motifdata["std"] = chd.utils.numpy.interpolate_1d(
    motifdata["position"].values,
    stds.index.values,
    stds.values,
)
motifdata["max"] = chd.utils.numpy.interpolate_1d(
    motifdata["position"].values,
    maxs.index.values,
    maxs.values,
)
motifslices_oi = motifdata.query("max > 7.").sort_values("std", ascending = False).head(8).sort_values("position", ascending=True)
motifslices_oi["start"] = motifslices_oi["position"] - 8
motifslices_oi["end"] = motifslices_oi["position"] + 20

# %%
# regions2 = pd.DataFrame([
#     [-2980, -2938],
#     # [-680, -550],
#     [-675, -650],
#     [-572, -558],
#     [-170, -155],
#     [-90, -70],
# ], columns=["start", "end"])
regions2 = motifslices_oi[["start", "end"]].copy()
regions2["chrom"] = regions.coordinates.loc[gene_id]["chrom"]
regions2["strand"] = regions.coordinates.loc[gene_id]["strand"]
regions2["start_chrom"] = regions.coordinates.loc[gene_id]["tss"] + regions2["start"]
regions2["end_chrom"] = regions.coordinates.loc[gene_id]["tss"] + regions2["end"]
breaking2 = pp.grid.Breaking(regions2, resolution=40, gap=0.1)

import pysam
genome = pysam.FastaFile(genome_folder / "mm10.fa")

motifs_folder = chd.get_output() / "data" / "motifs" / "mm" / "hocomocov12"
motifs_folder.mkdir(parents=True, exist_ok=True)

pwms, motifs = chd.data.motifscan.download.get_hocomoco(motifs_folder, organism = "mm")

panel_expanding = pp.grid.broken.Expanding(
    breaking, breaking2
)
fig.main.add_under(panel_expanding, padding_up = 0.)

# panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
#     region.name,
#     regionpositional,
#     cluster_info=cluster_info,
#     breaking=breaking2,
#     ylintresh=2000,
#     ymax=10000,
#     relative_to="previous",
#     label_cluster="front",
#     label_accessibility=False,
# )
# fig.main.add_under(panel_differential)

panel_genome = chd.data.motifscan.plot_genome.GroupedMotifsGenomeBroken(motifscan, gene_id, motifs_oi, breaking2, genome, pwms = pwms)
fig.main.add_under(panel_genome, padding_up = 0.)

fig.display()
# %%



# %% [markdown]
# ## Focus on Cdh5

# %%
symbol = "Cdh5"

if symbol not in regions.var["symbol"].tolist():
    raise ValueError(f"Symbol {symbol} not found in regions")
gene_id = regions.var.query("symbol == @symbol").index[0]

windows = regionpositional.select_windows(
    gene_id,
    prob_cutoff=6.5,
    differential_prob_cutoff=3.0,
    keep_tss=True,
    padding = 2000,
)
breaking = pp.grid.Breaking(windows, resolution=4000, gap=0.03)

fig = pp.grid.Figure(pp.grid.Grid(padding_height=0.05, padding_width=0.05))

region = regions.coordinates.loc[gene_id]

panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    xticks=[-100000, 0, 100000],
    gene_overlap_padding=10000 if windows["length"].sum() > 20000 else 100000,
    show_others=False,
    only_canonical=False,
)
fig.main.add_under(panel_genes, padding=0.0)

panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    show_others=False,
    only_canonical=False,
    label_positions = False,
)
fig.main.add_under(panel_genes, padding=0.0)

# exon usage
folder = pathlib.Path("/home/wouters/projects/crispyKC/output") / "rnaseq" / "cdh5_differential_exon"
region = regions.coordinates.loc[gene_id]
exon_usage_data = pd.read_pickle(folder / "exon_usage.pkl")
exon_usage_data.columns = (exon_usage_data.columns.astype(int) - region.tss) * region.strand
exon_usage_data = exon_usage_data.groupby(level = 0).mean()

# fig = pp.grid.Figure(pp.grid.Grid(padding_height=0.05, padding_width=0.05))
panel_exon = pp.grid.broken.Broken(breaking)

for ax, (_, subregion) in zip(panel_exon, breaking.regions.iterrows()):
    oi = (exon_usage_data.columns >= subregion["start"]) & (exon_usage_data.columns <= subregion["end"])
    plotdata = exon_usage_data.loc[:, oi]
    for i, (celltype, row) in enumerate(plotdata.iterrows()):
        ax.plot(row.index, row.values, label = celltype)
    ax.set_ylim(0, exon_usage_data.max().max())
fig.main.add_under(panel_exon, padding_up = 0.)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional,
    cluster_info=cluster_info1,
    breaking=breaking,
    ylintresh=2000,
    ymax=10000,
    relative_to="previous",
    label_cluster="front",
    label_accessibility=False,
)
fig.main.add_under(panel_differential)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional2,
    cluster_info=regionpositional2.clustering.cluster_info.loc[["Mid Hepatocyte", "KC", "LSEC"]],
    breaking=breaking,
    ymax=10,
    ylintresh = 5,
    relative_to="Mid Hepatocyte",
    label_cluster="front",
    label_accessibility=False,
)
fig.main.add_under(panel_differential)

panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
    motifscan,
    gene_id,
    motifs_oi,
    breaking,
    group_info=motif_groups,
    slices_oi=differential_slices.get_slice_scores(
        clustering=clustering, regions=regions
    ),
    show_triangle=False,
)
fig.main.add_under(panel_motifs)

motifdata = chd.data.motifscan.plot._process_grouped_motifs(
    gene_id, motifs_oi.query("group == 'Any SMAD'"), motifscan
)
motifdata["length"] = motifscan.motifs.loc[motifdata["motif"]]["length"].values

stds = regionpositional.get_plotdata(gene_id)[0]["prob"].unstack().loc[["blood_monocyte_0_WT_Ly6cHi", "liver_RLM_24_WT"]].diff().iloc[1]
maxs = regionpositional.get_plotdata(gene_id)[0]["prob"].unstack().max()
motifdata["std"] = chd.utils.numpy.interpolate_1d(
    motifdata["position"].values,
    stds.index.values,
    stds.values,
)
motifdata["max"] = chd.utils.numpy.interpolate_1d(
    motifdata["position"].values,
    maxs.index.values,
    maxs.values,
)
motifslices_oi = motifdata.query("max > 7.").sort_values("std", ascending = False).head(8).sort_values("position", ascending=True)
pad = 8
motifslices_oi["start"] = motifslices_oi["position"] - pad
motifslices_oi["end"] = motifslices_oi["position"] + motifslices_oi["length"] + pad

# merge adjacent motifslices
max_distance = 20
motifslices_oi = motifslices_oi.sort_values("position")
motifslices_oi["distance_to_next"] = (motifslices_oi["start"].shift(-1) - motifslices_oi["end"]).shift(1).fillna(0)
motifslices_oi["group"] = (motifslices_oi["distance_to_next"] > max_distance).cumsum()
motifslices_oi = motifslices_oi.groupby("group").agg(
    start = ("start", "min"),
    end = ("end", "max"),
    motif = ("motif", lambda x: x.tolist()),
    group = ("group", "first"),
)


# regions2 = pd.DataFrame([
#     [-2980, -2938],
#     # [-680, -550],
#     [-675, -650],
#     [-572, -558],
#     [-170, -155],
#     [-90, -70],
# ], columns=["start", "end"])
regions2 = motifslices_oi[["start", "end"]].copy()
regions2["chrom"] = regions.coordinates.loc[gene_id]["chrom"]
regions2["strand"] = regions.coordinates.loc[gene_id]["strand"]
regions2["start_chrom"] = regions.coordinates.loc[gene_id]["tss"] + regions2["start"]
regions2["end_chrom"] = regions.coordinates.loc[gene_id]["tss"] + regions2["end"]
breaking2 = pp.grid.Breaking(regions2, resolution=40, gap=0.1)

import pysam
genome = pysam.FastaFile(genome_folder / "mm10.fa")

motifs_folder = chd.get_output() / "data" / "motifs" / "mm" / "hocomocov12"
motifs_folder.mkdir(parents=True, exist_ok=True)

pwms, motifs = chd.data.motifscan.download.get_hocomoco(motifs_folder, organism = "mm")

panel_expanding = pp.grid.broken.Expanding(
    breaking, breaking2
)
fig.main.add_under(panel_expanding, padding_up = 0.)

# panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
#     region.name,
#     regionpositional,
#     cluster_info=cluster_info,
#     breaking=breaking2,
#     ylintresh=2000,
#     ymax=10000,
#     relative_to="previous",
#     label_cluster="front",
#     label_accessibility=False,
# )
# fig.main.add_under(panel_differential)

panel_genome = chd.data.motifscan.plot_genome.GroupedMotifsGenomeBroken(motifscan, gene_id, motifs_oi, breaking2, genome, pwms = pwms)
fig.main.add_under(panel_genome, padding_up = 0.)

for broken in panel_differential:
    for element in broken:
        # for 
        pass

fig.display()

# %%
# %%
symbol = "Cdh5"

if symbol not in regions.var["symbol"].tolist():
    raise ValueError(f"Symbol {symbol} not found in regions")
gene_id = regions.var.query("symbol == @symbol").index[0]

windows = regionpositional.select_windows(
    gene_id,
    prob_cutoff=6.5,
    differential_prob_cutoff=3.0,
    keep_tss=True,
    padding = 2000,
)
windows = pd.DataFrame([{'start': 7500,
 'end': 12500,}])
windows["length"] = windows["end"] - windows["start"]
breaking = pp.grid.Breaking(windows, resolution=500, gap=0.03)

fig = pp.grid.Figure(pp.grid.Grid(padding_height=0.05, padding_width=0.05))

region = regions.coordinates.loc[gene_id]

panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    xticks=[-100000, 0, 100000],
    gene_overlap_padding=10000 if windows["length"].sum() > 20000 else 100000,
    show_others=False,
    only_canonical=False,
)
fig.main.add_under(panel_genes, padding=0.0)

panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    show_others=False,
    only_canonical=False,
    label_positions = False,
)
fig.main.add_under(panel_genes, padding=0.0)

# exon usage
folder = pathlib.Path("/home/wouters/projects/crispyKC/output") / "rnaseq" / "cdh5_differential_exon"
region = regions.coordinates.loc[gene_id]
exon_usage_data = pd.read_pickle(folder / "exon_usage.pkl")
exon_usage_data.columns = (exon_usage_data.columns.astype(int) - region.tss) * region.strand
exon_usage_data = exon_usage_data.groupby(level = 0).mean()

# fig = pp.grid.Figure(pp.grid.Grid(padding_height=0.05, padding_width=0.05))
panel_exon = pp.grid.broken.Broken(breaking)

for ax, (_, subregion) in zip(panel_exon, breaking.regions.iterrows()):
    oi = (exon_usage_data.columns >= subregion["start"]) & (exon_usage_data.columns <= subregion["end"])
    plotdata = exon_usage_data.loc[:, oi]
    for i, (celltype, row) in enumerate(plotdata.iterrows()):
        ax.plot(row.index, row.values, label = celltype)
    ax.set_ylim(0, exon_usage_data.max().max())
fig.main.add_under(panel_exon, padding_up = 0.)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional,
    cluster_info=cluster_info1,
    breaking=breaking,
    ylintresh=2000,
    ymax=10000,
    relative_to="previous",
    label_cluster="front",
    label_accessibility=False,
)
fig.main.add_under(panel_differential)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region.name,
    regionpositional2,
    cluster_info=regionpositional2.clustering.cluster_info.loc[["Mid Hepatocyte", "KC", "LSEC"]],
    breaking=breaking,
    ymax=10,
    ylintresh = 5,
    relative_to="Mid Hepatocyte",
    label_cluster="front",
    label_accessibility=False,
)
fig.main.add_under(panel_differential)

panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
    motifscan,
    gene_id,
    motifs_oi,
    breaking,
    group_info=motif_groups,
    slices_oi=differential_slices.get_slice_scores(
        clustering=clustering, regions=regions
    ),
    show_triangle=False,
)
fig.main.add_under(panel_motifs)

motifslices_oi = pd.DataFrame([
    {"start":9000, "end":9500},
    {"start":9500, "end":10000}
])
regions2 = motifslices_oi[["start", "end"]].copy()
regions2["chrom"] = regions.coordinates.loc[gene_id]["chrom"]
regions2["strand"] = regions.coordinates.loc[gene_id]["strand"]
regions2["start_chrom"] = regions.coordinates.loc[gene_id]["tss"] + regions2["start"]
regions2["end_chrom"] = regions.coordinates.loc[gene_id]["tss"] + regions2["end"]
breaking2 = pp.grid.Breaking(regions2, resolution=100, gap=0.1)

import pysam
genome = pysam.FastaFile(genome_folder / "mm10.fa")

motifs_folder = chd.get_output() / "data" / "motifs" / "mm" / "hocomocov12"
motifs_folder.mkdir(parents=True, exist_ok=True)

pwms, motifs = chd.data.motifscan.download.get_hocomoco(motifs_folder, organism = "mm")

panel_expanding = pp.grid.broken.Expanding(
    breaking, breaking2
)
fig.main.add_under(panel_expanding, padding_up = 0.)

panel_genome = chd.data.motifscan.plot_genome.GroupedMotifsGenomeBroken(motifscan, gene_id, motifs_oi, breaking2, genome, pwms = pwms)
fig.main.add_under(panel_genome, padding_up = 0.)

# add cuts
ax = panel_exon.elements[0][0]
# for i in range(windows["start"][0], windows["end"][0], 50):
for i in [8000, 9500, 10550]:
    ax.axvline(i, color = "k", lw = 0.5)
    text = ax.text(i, 0.5, i, rotation = 90, va = "bottom", ha = "center", fontsize = 6)
    text.set_path_effects([mpl.patheffects.withStroke(linewidth=1, foreground='w')])
ax.axvline()

fig.display()
# %%
region1 = {
    "start":8000,
    "end":9500,
}
region1["chrom"] = region["chrom"]
region1["start_genome"] = int(region["tss"] + region1["start"] if region["strand"] == 1 else region["tss"] - region1["start"])
region1["end_genome"] = int(region["tss"] + region1["end"] if region["strand"] == 1 else region["tss"] - region1["end"])
region1["strand"] = int(region["strand"])

sequence1 = genome.fetch(region1["chrom"], region1["start_genome"], region1["end_genome"])

with open("/home/wouters/projects/crispyKC/results/cdh5_regionA_PrimedEnhancer.fa", "w") as f:
    f.write(f">{region1['chrom']}:{region1['start_genome']}-{region1['end_genome']}\n{sequence1}")

# %%
region2 = {
    "start":9500,
    "end":10550,
}
region2["chrom"] = region["chrom"]
region2["start_genome"] = int(region["tss"] + region2["start"] if region["strand"] == 1 else region["tss"] - region2["start"])
region2["end_genome"] = int(region["tss"] + region2["end"] if region["strand"] == 1 else region["tss"] - region2["end"])
region2["strand"] = int(region["strand"])

sequence2 = genome.fetch(region2["chrom"], region2["start_genome"], region2["end_genome"])
with open("/home/wouters/projects/crispyKC/results/cdh5_regionB_TSS.fa", "w") as f:
    f.write(f">{region2['chrom']}:{region2['start_genome']}-{region2['end_genome']}\n{sequence2}")

# %%
