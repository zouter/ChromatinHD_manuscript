# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import polyptich as pp

import pysam
import eyck

chd.set_default_device("cpu")


pp.setup_ipython()

# %%
pp.paths.results().mkdir(parents=True, exist_ok=True)

# %%
dataset_name = "liver"
folder_dataset = chd.get_output() / "datasets" / dataset_name

# %%
folds = chd.data.folds.Folds(
    folder_dataset / "folds"
)
transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings" / "cluster"
)
fragments = chd.data.Fragments(folder_dataset / "fragments" / "100k100k")

# %%
eyck.m.t.plot_umap(transcriptome, ["Ly6c2", "Clec4f", "Cdh5", "Spi1", "Zeb2", "Hdac9", "Mki67"], datashader = True).display()

# %% [markdown]
# ## Training

# %%
model_folder = (
    chd.get_output() / "diff" / "liver" / "100k100k" / "cluster"
)

# %%
import chromatinhd.models.diff.model.binary

models = chd.models.diff.model.binary.Models.create(
    fragments = fragments,
    clustering=clustering,
    folds=folds,
    model_params = dict(
        encoder="shared",
        encoder_params=dict(
            delta_regularization=True,
            delta_p_scale=1.0,
            # bias_regularization=True,
            # bias_p_scale=0.5,
            binwidths=(5000, 1000, 500, 100, 50, 25),
        ),
    ),
    train_params = dict(
        early_stopping=False,
        n_epochs = 5, # <----- originally I used 40
    ),
    path=model_folder,
    reset = True,
)

# %%
models.train_models()

# %%
models.models["0"].trace.plot()

# %% [markdown]
# ## Inference

# %%
models = chd.models.diff.model.binary.Models(model_folder)

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    reset=True,
)

# %%
regionpositional.score(
    models,
    fragments=fragments,
    clustering=clustering,
    device="cpu",
)
regionpositional

# %%
# gene_id = transcriptome.gene_id("Slc40a1")
# gene_id = transcriptome.gene_id("Cdh5")
# gene_id = transcriptome.gene_id("Id3")
# gene_id = transcriptome.gene_id("Lhx2")
gene_id = transcriptome.gene_id("Bmp10")
# gene_id = transcriptome.gene_id("Lyve1")
windows = regionpositional.select_windows(
    gene_id,
    prob_cutoff=0.5,
    differential_prob_cutoff=3.,
    keep_tss=True,
    padding = 1000,
)
breaking = pp.grid.Breaking(windows, resolution=4000, gap=0.03)

# %%
motifscan_name = "hocomocov12_1e-4"
genome_folder = pathlib.Path("/srv/data/genomes/mm10")
motifscan_genome = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)
motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
    path=chd.get_output()
    / "datasets"
    / dataset_name
    / "motifscans"
    / "100k100k"
    / motifscan_name,
    parent = motifscan_genome,
    regions = fragments.regions,
)

# %%
motifs_oi = pd.concat([
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
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["SMAD4"])],
        "group":"SMAD4",
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["RREB1"])],
        "group":"RREB1",
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.MOUSE_gene_symbol.isin(["Tcf7l1", "Tcf3", "Tcf7l2"])],
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
        "motif":motifscan.motifs.index[motifscan.motifs.MOUSE_gene_symbol.isin(["Lhx2"])],
        "group":"LHX2",
    }),
]).set_index("motif")
motif_groups = pd.DataFrame({"group": motifs_oi["group"].unique()}).set_index("group")
motif_groups["label"] = motif_groups.index
motif_groups["color"] = [mpl.colors.to_hex(mpl.colormaps["tab10"](i)) for i in range(len(motif_groups))]

# %%
cluster_info = clustering.cluster_info.loc[["KC", "LSEC", "Stellate", "Portal Hepatocyte"]]

# %%
slices = regionpositional.calculate_slices(-0.5, step=25)
((slices.end_position_ixs - slices.start_position_ixs) * slices.step).sum() / (fragments.regions.coordinates["end"] - fragments.regions.coordinates["start"]).sum()

# %%
differential_slices = regionpositional.calculate_differential_slices(
    slices, fc_cutoff=1.5, a = "KC", b = "LSEC"
)

# %%
fig = pp.grid.Figure(pp.grid.Grid(padding_height=0.01, padding_width=0.05))

region = fragments.regions.coordinates.loc[gene_id]

panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    xticks=[-100000, 0, 100000],
    gene_overlap_padding=10000 if windows["length"].sum() > 20000 else 100000,
    show_others=True,
    only_canonical=False,
)
fig.main.add_under(panel_genes, padding=0.)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    gene_id,
    regionpositional,
    cluster_info=cluster_info,
    breaking=breaking,
    ylintresh=5,
    ymax=20,
    label_accessibility=False,
    norm_atac_diff=mpl.colors.Normalize(np.log(1 / 8), np.log(8.0), clip=True),
)
fig.main.add_under(panel_differential)

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome,
    clustering,
    gene_id,
    cluster_info=cluster_info,
    show_n_cells=False,
)
fig.main.add_right(panel_expression, panel_differential)

panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
    motifscan,
    gene_id,
    motifs_oi = motifs_oi,
    breaking = breaking,
    group_info=motif_groups,
    show_triangle=False,
    slices_oi=differential_slices.get_slice_scores(
        clustering=clustering, regions=fragments.regions
    ),
)
fig.main.add_under(panel_motifs)

panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking=breaking,
    genome="GRCm38",
    show_others = False,
    only_canonical=False,
)
fig.main.add_under(panel_genes, padding=0.0, padding_up = 0.2)

fig.display()

# %%
eyck.m.t.plot_umap(transcriptome, ["Vwf"], datashader = False, panel_size = 3).display()