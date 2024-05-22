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
# %config InlineBackend.figure_format = 'retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %%
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "pbmc20k"
# dataset_name = "hspc"
dataset_name = "lymphoma"
# dataset_name = "liver"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")

regions_name = "100k100k"
# regions_name = "10k10k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
fragments.coordinates[:].shape[0] / fragments.obs.shape[0]

# %% [markdown]
# ## Motifs

# %% [markdown]
# ### Get slice scores chromatinhd

# %%
# clusters_oi = ["B", "Lymphoma"]
clusters_oi = ["Lymphoma", "Lymphoma cycling"]

# clusters_oi = ["Erythroblast", "MEP"]

# clusters_oi = ["Erythroblast", "Erythrocyte precursors"]
# clusters_oi = ["Erythroblast", "Megakaryocyte"]
# clusters_oi = ["Granulocyte 1", "Granulocyte 2"]

# clusters_oi = ["Erythroblast", "Myeloid"]
# clusters_oi = ["CD4 naive T", "CD4 memory T"]
# clusters_oi = ["naive B", "memory B"]

celltype_a, celltype_b = clusters_oi

# %%
# selected_slices = regionpositional.calculate_slices(-1., step = 5, clusters_oi = clusters_oi)
# differential_slices = regionpositional.calculate_differential_slices(selected_slices, fc_cutoff = 1.5, score = "diff")

# %%
folder = chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional" / f"differential_{celltype_a}_{celltype_b}" / "-1-1.5"
differential_slices = pickle.load(open(folder / "differential_slices.pkl", "rb"))

# %%
slicescores = differential_slices.get_slice_scores(regions = fragments.regions, clustering = clustering, cluster_info = clustering.cluster_info.loc[clusters_oi])

slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

# %%
n_desired_positions = slicescores.groupby("cluster")["length"].sum()
n_desired_positions

# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
slicecounts = motifscan.count_slices(slices)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores, slicecounts)
enrichment["log_odds"] = np.log(enrichment["odds"])

# %%
sc.pl.umap(transcriptome.adata, color = "celltype", legend_loc = "on data", frameon = False, show = False)

# %%
motifscan.motifs["gene"] = transcriptome.var.reset_index().set_index("symbol").reindex(motifscan.motifs["HUMAN_gene_symbol"])["gene"].values

# %%
# enrichment.loc[clusters_oi[0]].sort_values("odds", ascending = False).query("q_value < 0.05").head(20)

# %% [markdown]
# ### Diffexp

# %%
# # t-test
# import scanpy as sc
# adata_raw = transcriptome.adata.raw.to_adata()
# adata_raw.obs["cluster"] = clustering.labels
# sc.pp.normalize_total(adata_raw, target_sum=1e4)
# sc.pp.log1p(adata_raw)

# sc.tl.rank_genes_groups(adata_raw, groupby="cluster", groups = ["Lymphoma"], reference = "B", method="t-test", key_added = "lymphoma_vs_b")
# lymphoma_vs_b = pd.DataFrame({
#     "gene":adata_raw.uns["lymphoma_vs_b"]["names"].tolist(),
#     "scores":adata_raw.uns["lymphoma_vs_b"]["scores"].tolist(),
#     "pvals_adj":adata_raw.uns["lymphoma_vs_b"]["pvals_adj"].tolist(),
#     "logfoldchanges":adata_raw.uns["lymphoma_vs_b"]["logfoldchanges"].tolist(),
# }).apply(lambda x:x.str[0]).set_index("gene")

# %%
# t-test
import scanpy as sc
adata_raw = transcriptome.adata.raw.to_adata()
adata_raw = adata_raw[adata_raw.obs["celltype"].isin(clusters_oi), transcriptome.var.index]
adata_raw.obs["cluster"] = clustering.labels
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)

sc.tl.rank_genes_groups(adata_raw, groupby="cluster", groups = [clusters_oi[0]], reference = clusters_oi[1], method="t-test", key_added = "lymphoma_vs_cycling")
lymphoma_vs_cycling = pd.DataFrame({
    "gene":adata_raw.uns["lymphoma_vs_cycling"]["names"].tolist(),
    "scores":adata_raw.uns["lymphoma_vs_cycling"]["scores"].tolist(),
    "pvals_adj":adata_raw.uns["lymphoma_vs_cycling"]["pvals_adj"].tolist(),
    "logfoldchanges":adata_raw.uns["lymphoma_vs_cycling"]["logfoldchanges"].tolist(),
}).apply(lambda x:x.str[0]).set_index("gene")

# %%
enrichment_joined = enrichment.loc[clusters_oi[0]].loc[motifscan.motifs.loc[enrichment.loc[clusters_oi[0]].index, "gene"].dropna().index]
enrichment_joined = enrichment_joined.join(motifscan.motifs.loc[enrichment_joined.index, "gene"])
enrichment_joined[["scores", "pvals_adj", "logfoldchanges"]] = lymphoma_vs_cycling[["scores", "pvals_adj", "logfoldchanges"]].loc[enrichment_joined["gene"]].values

# %%
# score_cutoff = np.log(20.)
score_cutoff = 1.
(
    np.exp(enrichment_joined.query("pvals_adj < 0.05").query("scores > @score_cutoff")["log_odds"].mean() - enrichment_joined.query("pvals_adj < 0.05").query("scores < -@score_cutoff")["log_odds"].mean()),
    # np.exp(enrichment_peak_joined.query("pvals_adj < 0.05").query("scores > @score_cutoff")["log_odds"].mean() - enrichment_peak_joined.query("pvals_adj < 0.05").query("scores < -@score_cutoff")["log_odds"].mean())
)

# %%
enrichment_joined.query("pvals_adj < 0.05").query("scores > @score_cutoff")["log_odds"].mean()

# %% [markdown]
# ### GRN capture

# %%
import chromatinhd_manuscript as chdm

# %%
motifclustermapping = chdm.motifclustermapping.get_motifclustermapping(dataset_name, motifscan, clustering)

# enrichment_q_value_cutoff = 0.01
# enrichment_odds_cutoff = 1.5
# n_top_motifs = 20
# motifclustermapping = []
# for ct in tqdm.tqdm(clustering.cluster_info.index):
#     motifs_oi = (
#         enrichment.loc[ct]
#         # enrichment_peak.loc[ct]
#         .query("q_value < @enrichment_q_value_cutoff")
#         .query("odds > @enrichment_odds_cutoff")
#         .sort_values("odds", ascending=False)
#         .head(n_top_motifs)
#         .index
#     )
#     motifclustermapping.append(pd.DataFrame({"cluster":ct, "motif":motifs_oi}))
# motifclustermapping = pd.concat(motifclustermapping)
# motifclustermapping = motifclustermapping.query("cluster != 'Stromal'")

# motifclustermapping["motif"] = np.random.choice(motifscan.motifs.index, size = len(motifclustermapping))

# motifclustermapping = motifclustermapping.loc[motifclustermapping["motif"].str.contains("CUX")]
# motifclustermapping = motifclustermapping.loc[motifclustermapping["motif"].str.contains("SPI1")]
# motifclustermapping = motifclustermapping.loc[motifclustermapping["motif"].str.contains("HNF4")]
# motifclustermapping = motifclustermapping.loc[motifclustermapping["motif"].str.contains("GATA1")]

motifclustermapping = pd.DataFrame([
    ["Lymphoma", motifscan.select_motif("PO5F1"), "POU2F2"],
    ["Lymphoma", motifscan.select_motif("P5F1B"), "POU2F2"],
    # ["Lymphoma", motifscan.select_motif("IRF7"), "IRF"],
    # ["Lymphoma", motifscan.select_motif("IRF4"), "IRF"],
    ["Lymphoma", motifscan.select_motif("ETV3"), "ETS/SPI1"],
    ["Lymphoma", motifscan.select_motif("ETV6"), "ETS/SPI1"],
    # ["Lymphoma", motifscan.select_motif("SPI1"), "ETS/SPI1"],
    # ["Lymphoma cycling", motifscan.select_motif("CREM"), "CREM"],
], columns = ["cluster", "motif", "group"])

# %%
# expression_lfc_cutoff = 0.
# expression_lfc_cutoff = -np.inf
# expression_lfc_cutoff = np.log(1.2)
# expression_lfc_cutoff = np.log(1.1)

score_cutoff = -np.inf
# score_cutoff = 0.
# score_cutoff = 10
# score_cutoff = np.log2(2)

# %%
enrichment["expected"] = enrichment["contingency"].str[0].str[1] / enrichment["contingency"].str[0].str[0]


# %%
def enrich_per_gene(motifclustermapping, enrichment, slicescores, slicecounts):
    founds = []
    for ct in tqdm.tqdm(motifclustermapping["cluster"].unique()):
        motifs_oi = motifclustermapping.query("cluster == @ct")["motif"]

        slicescores_foreground = slicescores.query("cluster == @ct")

        slicecounts_oi = slicecounts.loc[slicescores_foreground["slice"], motifs_oi]

        genes_oi = transcriptome.var.index

        found = slicecounts_oi.groupby(slicescores_foreground["region"].values.astype(str)).sum()
        found.index.name = "gene"
        found = found.reindex(genes_oi, fill_value = 0)
        expected = (slicescores_foreground.groupby("region")["length"].sum().reindex(genes_oi).values[:, None] * enrichment.loc[ct].loc[motifs_oi]["expected"].values[None, :]) + 1e-4
        ratio = found / expected
        
        found = found.unstack().to_frame(name = "found")
        ratio = ratio.unstack().to_frame(name = "ratio")
        ratio["expected"] = expected.flatten()

        found = found.join(ratio).assign(cluster = ct)

        if len(found):
            founds.append(found)
    founds = pd.concat(founds).reset_index().set_index(["cluster", "gene", "motif"])
    return founds


# %%
founds = enrich_per_gene(motifclustermapping, enrichment, slicescores, slicecounts)

# %%
2**(0.67)

# %%
# lymphoma_vs_b.loc[transcriptome.var.index[transcriptome.var.symbol.str.startswith("BCL")]].join(transcriptome.var[["symbol"]])
lymphoma_vs_cycling.loc[transcriptome.var.index[transcriptome.var.symbol.str.startswith("ETV6")]].join(transcriptome.var[["symbol"]]).sort_values("scores", ascending = False)

# %%
enrichment.loc["Lymphoma"].loc[motifscan.motifs.loc[motifscan.motifs["symbol"].str.contains("PAX")].index].sort_values("odds", ascending = False)

# %%
motifclustermapping

# %%
founds.sort_values("found", ascending = False).join(transcriptome.var[["symbol"]]).loc["Lymphoma"].join(lymphoma_vs_cycling).query("pvals_adj < 0.2").query("scores > 0").xs("ETV3.H12CORE.0.SM.B", level = "motif").head(20)

# %%
lymphoma_vs_cycling.loc[transcriptome.var.loc[transcriptome.var.symbol.str.contains("PAX5")].index].join(transcriptome.var[["symbol"]]).sort_values("pvals_adj")

# %%
# select which genes to use
# regions_oi = fragments.regions.coordinates.loc[
#     diffexp.loc[selection_cluster_ids]
#     .groupby("gene")
#     .mean(numeric_only=True)
#     .sort_values("logfoldchanges", ascending=False)
#     .head(500)
#     .index
# ]
regions_oi = fragments.regions.coordinates.loc[
    transcriptome.gene_id(
        [
            "PAX5",
            "CD74",
            "FCRL1",
            "FCRL2",
            "FCRL3",
            "FCRL6",
            "BLK",
            "FCHSD2",
            "IL2RA",
            "SAP25",
            "CKLF",
            "GAB2",
            "TGFBI",
            "TNFSF9",
            "SINHCAF",
            "IRAK2",
            "UTRN",
            "POU2F2",
            "EBF1",
            "RAD51B",
            "BCL2",

            "ETV6",
            "IRF4",
        ]
    )
]


# regions_oi = fragments.regions.coordinates.loc[
#     transcriptome.gene_id(
#         [
#             "LMF2",
#             "IL4I1",
#             "BAHD1",
#             "TYMP",
#             "HIVEP3",
#             "MKI67",
#         ]
#     )
# ]

# %%
motif_ids = motifclustermapping["motif"]

# %%
selection_cluster_ids = clusters_oi 

# %%
# determine slices of interest, i.e. those with a motif
slicescores_selected = slicescores.query("cluster in @selection_cluster_ids").query("region in @regions_oi.index")
slicescores_oi = slicescores_selected.loc[(slicecounts.loc[slicescores_selected["slice"], motif_ids] > 0).any(axis = 1).values].copy()
slicescores_oi["region"] = pd.Categorical(slicescores_oi["region"], categories = regions_oi.index.unique())
slicescores_oi = slicescores_oi.sort_values("region")
slicescores_oi["start"] = slicescores_oi["start"] - 400
slicescores_oi["end"] = slicescores_oi["end"] + 400

# merge regions that are close to each other
slicescores_oi = slicescores_oi.sort_values(["region", "start"])
max_merge_distance = 10
slicescores_oi["distance_to_next"] = slicescores_oi["start"].shift(-1) - slicescores_oi["end"]
slicescores_oi["same_region"] = slicescores_oi["region"].shift(-1) == slicescores_oi["region"]

slicescores_oi["merge"] = (slicescores_oi["same_region"] & (slicescores_oi["distance_to_next"] < max_merge_distance)).fillna(False)
slicescores_oi["group"] = (~slicescores_oi["merge"]).cumsum().shift(1).fillna(0).astype(int)
slicescores_oi = (
    slicescores_oi.groupby("group")
    .agg({"start": "min", "end": "max", "distance_to_next": "last", "region":"first"})
    .reset_index(drop=True)
)
slicescores_oi = slicescores_oi#.iloc[:3]

# %%
breaking = chd.grid.broken.Breaking(slicescores_oi, 0.05, resolution = 3000)

# %%
# preload peakcallers
peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name)
# peakcallers = peakcallers.loc[["macs2_summits"]]
peakcallers = peakcallers.loc[["macs2_leiden_0.1_merged", "macs2_summits", "encode_screen"]]

peakcaller_region_data = {}
for peakcaller, peakcaller_info in peakcallers.iterrows():
    data = pd.read_table(peakcaller_info["path"], header = None)
    data.columns = ["chrom", "start", "end"] + list(data.columns[3:])
    for region_id in slicescores_oi["region"].unique():
        region = fragments.regions.coordinates.loc[region_id]
        peakcaller_region_data[(peakcaller, region_id)] = data.loc[
                (data["chrom"] == region["chrom"]) &
                (data["start"] >= region["start"]) &
                (data["end"] <= region["end"])
            ]

# %%
# load chipseq data and determine chip-seq+motif design
import pyBigWig
files_all = []
for folder in ["gm1282_tf_chipseq_bw", "gm12891_tf_chipseq_bw"]:
    bed_folder = chd.get_output() / "bed" / folder
    files = pd.read_csv(bed_folder / "files.csv", index_col=0)
    files["path"] = bed_folder / files["filename"]

    files["n_complaints"] = files["audit_not_compliant"].map(lambda x: x.count(",") if isinstance(x, str) else 1)
    files["n_replicated"] = files["technical_replicate(s)"].map(lambda x: x.count(",") + 1 if isinstance(x, str) else 1)
    files["sorter"] = files["n_replicated"] / files["n_complaints"]
    files["subset"] = folder
    files = files.sort_values("sorter", ascending=False)
    files_all.append(files)
files = pd.concat(files_all)

design = pd.DataFrame(
    [
        [
            "OCT2\nGM12878",
            [motifscan.select_motif(symbol = "POU2F2"), motifscan.select_motif("P5F1B"), motifscan.select_motif("PO5F1")],
            files.query("(experiment_target == 'POU2F2-human') & (subset == 'gm1282_tf_chipseq_bw')")["path"].iloc[0],
        ],
        [
            "ETV6\nGM12878",
            [*motifscan.select_motifs(symbol = "ETV6"), *motifscan.select_motifs(symbol = "ETV3")],
            files.query("(experiment_target == 'ETV6-human')")["path"].iloc[0],
        ],
        # [
        #     "IRF4\nGM12878",
        #     [*motifscan.select_motifs(symbol = "IRF4"), *motifscan.select_motifs(symbol = "IRF7")],
        #     files.query("(experiment_target == 'IRF4-human')")["path"].iloc[0],
        # ],
    ],
    columns=["label", "motifs", "bigwig"],
)
assert all([path.exists() for path in design["bigwig"]])

# %%
# preload setting and motif data per region
setting_region_motifdata = {}
for setting_name, setting in design.iterrows():
    for region_id in slicescores_oi["region"].unique():
        motifs_oi = pd.DataFrame({"motif":setting["motifs"]}).set_index("motif")
        motifs_oi, group_info, motifdata = chd.data.motifscan.plot._process_grouped_motifs(region_id, motifs_oi, motifscan)

        motifdata["captured"] = ((slicescores_selected["start"].values[None, :] < motifdata["position"].values[:, None]) & (slicescores_selected["end"].values[None, :] > motifdata["position"].values[:, None])).any(1)
        motifdata = motifdata.loc[motifdata["captured"]]

        setting_region_motifdata[(setting_name, region_id)] = motifs_oi, group_info, motifdata

# %%
cluster_ids = ["Lymphoma", "Lymphoma cycling"]

# %%
motifs_oi = motifclustermapping.set_index("motif")

# %%
n = 0
for subregion_ix, (_, subregion_info) in zip(
    range(len(slicescores_oi)), slicescores_oi.iloc[:99999].iterrows()
):
    region = fragments.regions.coordinates.loc[subregion_info["region"]]
    window = subregion_info[["start", "end"]]
    for setting_id, setting in design.iterrows():
        subregion_info_uncentered = chd.data.peakcounts.plot.uncenter_peaks(subregion_info.copy(), region)

        # chip-seq
        bw = pyBigWig.open(str(setting["bigwig"]))

        plotdata_chip = pd.DataFrame(
            {
                "position": np.arange(subregion_info["start"], subregion_info["end"], 1),
                "value": bw.values(
                    subregion_info_uncentered["chrom"],
                    int(subregion_info_uncentered["start"]),
                    int(subregion_info_uncentered["end"]),
                )[:: int(region["strand"])],
            }
        )

        # motifs
        _, group_info, motifdata = setting_region_motifdata[(setting_id, subregion_info["region"])]
        plotdata_motifs = motifdata.loc[
            (motifdata["position"] >= subregion_info["start"]) & (motifdata["position"] <= subregion_info["end"])
        ].copy()
        plotdata_motifs["significant"] = (
            plotdata_chip.set_index("position").reindex(plotdata_motifs["position"])["value"] > 2.0
        ).values

        n += (plotdata_motifs["significant"].sum())
print(n)

# %%
from chromatinhd.plot import format_distance
import textwrap

fig = chd.grid.Figure(chd.grid.BrokenGrid(breaking, padding_height = 0.03))

# cluster_info = clustering.cluster_info
# cluster_info = clustering.cluster_info.loc[cluster_ids]


for subregion_ix, (_, subregion_info), grid, width in zip(
    range(len(slicescores_oi)), slicescores_oi.iloc[:99999].iterrows(), fig.main, fig.main.panel_widths
):
    # add upper labelling panel
    panel_labeling = chd.grid.Panel((width, 0.1))
    panel_labeling.ax.set_xlim(subregion_info["start"], subregion_info["end"])
    panel_labeling.ax.axis("off")
    grid.add_under(panel_labeling)
    if subregion_ix == 0:  # we only use the left-most panel for labelling
        ax_labeling = panel_labeling.ax

    cluster_info = clustering.cluster_info.loc[["B", "Lymphoma"]]
    region = fragments.regions.coordinates.loc[subregion_info["region"]]
    window = subregion_info[["start", "end"]]
    panel_differential = chd.models.diff.plot.Differential.from_regionpositional(
        subregion_info["region"],
        regionpositional,
        width=width,
        window=window,
        cluster_info=cluster_info,
        panel_height=0.4,
        relative_to="B",
        label_accessibility=False,
        label_cluster=False,
        show_tss=False,
        ymax = 2,
    )
    grid.add_under(panel_differential)
    for _, ax in panel_differential.elements:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        sns.despine(ax=ax, left=True)
    if subregion_ix == 0:
        for (cluster_id, cluster_info_), (panel, ax) in zip(cluster_info.iterrows(), panel_differential.elements):
            ax.set_ylabel(cluster_info_["label"].replace(" ", "\n"), rotation=0, ha="right", va="center")

    cluster_info = clustering.cluster_info.loc[["Lymphoma cycling"]]
    region = fragments.regions.coordinates.loc[subregion_info["region"]]
    window = subregion_info[["start", "end"]]
    panel_differential = chd.models.diff.plot.Differential.from_regionpositional(
        subregion_info["region"],
        regionpositional,
        width=width,
        window=window,
        cluster_info=cluster_info,
        panel_height=0.4,
        relative_to="Lymphoma",
        label_accessibility=False,
        label_cluster=False,
        show_tss=False,
        ymax = 5,
    )
    grid.add_under(panel_differential)
    for _, ax in panel_differential.elements:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        sns.despine(ax=ax, left=True)
    if subregion_ix == 0:
        for (cluster_id, cluster_info_), (panel, ax) in zip(cluster_info.iterrows(), panel_differential.elements):
            ax.set_ylabel(cluster_info_["label"].replace(" ", "\n"), rotation=0, ha="right", va="center")

    # motifs
    panel_motifs = chd.data.motifscan.plot.GroupedMotifs(motifscan, subregion_info["region"], motifs_oi = motifs_oi, width = width, window = window, label_motifs=subregion_ix == len(slicescores_oi)-1)
    grid.add_under(panel_motifs)

    # peaks
    panel_peaks = chd.data.peakcounts.plot.Peaks.from_preloaded(
        region,
        peakcallers,
        peakcaller_data={
            peakcaller: peakcaller_region_data[(peakcaller, subregion_info["region"])]
            for peakcaller in peakcallers.index
        },
        window=subregion_info[["start", "end"]],
        width=width,
        label_rows=False,
        label_methods=subregion_ix == 0,
        label_methods_side = "left",
    )
    grid.add_under(panel_peaks)
    sns.despine(ax=panel_peaks.ax, left=True)


    # ChIP-seq + motifs
    if True:
        for setting_id, setting in design.iterrows():
            subregion_info_uncentered = chd.data.peakcounts.plot.uncenter_peaks(subregion_info.copy(), region)

            panel, ax = grid.add_under(chd.grid.Panel((width, 0.35)))
            # ax.set_ylabel(setting["label"], rotation = 0, ha = "right", va = "center")

            # chip-seq
            bw = pyBigWig.open(str(setting["bigwig"]))

            plotdata_chip = pd.DataFrame(
                {
                    "position": np.arange(subregion_info["start"], subregion_info["end"], 1),
                    "value": bw.values(
                        subregion_info_uncentered["chrom"],
                        int(subregion_info_uncentered["start"]),
                        int(subregion_info_uncentered["end"]),
                    )[:: int(region["strand"])],
                }
            )
            ax.plot(plotdata_chip["position"], plotdata_chip["value"], color="#333", lw=0.5)
            ax.fill_between(
                plotdata_chip["position"],
                plotdata_chip["value"],
                color="#333",
                alpha=0.3,
                lw=0.,
            )
            ax.set_ylim(0, 100)
            ax.set_yscale("symlog", linthresh=10)
            ax.set_yticks([])
            ax.set_yticks([], minor = True)
            ax.set_xticks([])
            ax.set_xlim(subregion_info["start"], subregion_info["end"])
            sns.despine(ax=ax, left=True)

            # motifs
            _, group_info, motifdata = setting_region_motifdata[(setting_id, subregion_info["region"])]
            plotdata_motifs = motifdata.loc[
                (motifdata["position"] >= subregion_info["start"]) & (motifdata["position"] <= subregion_info["end"])
            ].copy()
            plotdata_motifs["significant"] = (
                plotdata_chip.set_index("position").reindex(plotdata_motifs["position"])["value"] > 2.0
            ).values

            ax.scatter(
                plotdata_motifs["position"],
                [1] * len(plotdata_motifs),
                transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                marker="v",
                color=["orange" if x else "grey" for x in plotdata_motifs["significant"]],
                alpha=1,
                s=200,
                zorder=20,
                lw=1,
                edgecolors="white",
            )

            if subregion_ix == 0:
                ax.set_ylabel(setting["label"], rotation=0, ha="right", va="center")


# gene labels
last_x = breaking.regions["start"].iloc[0]
gap_data = breaking.gap * breaking.resolution
ax_labeling.set_ylim(-0.2, 0.2)
y = 0.2
for region_id, subregions in breaking.regions.groupby("region"):
    if len(subregions) == 0:
        continue
    x = [last_x, last_x + subregions["width"].sum() + (len(subregions) - 1) * gap_data]
    ax_labeling.plot(
        x,
        [y]*2,
        color="#888888",
        lw=1,
        clip_on=False,
    )
    ax_labeling.annotate(
        transcriptome.symbol(region_id),
        (np.mean(x), y+0.1),
        xycoords="data",
        fontsize=8,
        ha="center",
        va="bottom",
        clip_on=False,
        annotation_clip=False,
        # bbox=dict(facecolor="white", edgecolor="white", alpha=1.0, boxstyle='square,pad=0.1'),
        fontstyle="italic",
    )

    last_x = x[1] + gap_data

# distance labels
last_x = breaking.regions["start"].iloc[0]
y = -0.2
for subregion_id, subregion in breaking.regions.iterrows():
    x = [last_x, last_x + subregion["width"]]

    ax_labeling.annotate(
        format_distance(
            round(
                int(subregion["start"] + (subregion["end"] - subregion["start"]) / 2), -3
            ),
            None,
        ),
        (np.mean(x), y),
        ha="center",
        va="center",
        fontsize=6,
        color="#999999",
        bbox=dict(facecolor="#FFFFFF", boxstyle="square,pad=0.1", lw=0),
        clip_on=False,
        annotation_clip=False,
        # rotation=90,
    )

    last_x = x[1] + gap_data

fig.plot()

# %%
# Targets for therapies: BLK, IL2RA, TNFSF9 (CD137L)

# %%
axes = sc.pl.heatmap(transcriptome.adata[transcriptome.adata.obs["celltype"].isin(["B", "Lymphoma", "Lymphoma cycling"])], regions_oi.index, "celltype", use_raw = False, log = True, show = False)
axes["heatmap_ax"].set_xticklabels(transcriptome.symbol(regions_oi.index), rotation = 90, fontsize = 6)

# %% [markdown]
# ## Control: are these truly cycling lymphoma cells?

# %%
adata_oi = transcriptome.adata[transcriptome.adata.obs["celltype"].isin(["Lymphoma", "Lymphoma cycling", "B"])].copy()
sc.pp.pca(adata_oi)
sc.pp.neighbors(adata_oi)
sc.tl.umap(adata_oi)


# %%
# %%
def get_gene_id(symbols):
    return adata_oi.var.reset_index().set_index("symbol").loc[symbols, "gene"].values

# %%
if not pathlib.Path("./regev_lab_cell_cycle_genes.txt").exists():
    # !wget https://raw.githubusercontent.com/scverse/scanpy_usage/master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt

# %%
cell_cycle_genes = [x.strip() for x in open('./regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
s_genes = get_gene_id([g for g in s_genes if g in adata_oi.var["symbol"].tolist()])
g2m_genes = cell_cycle_genes[43:]
g2m_genes = get_gene_id([g for g in g2m_genes if g in adata_oi.var["symbol"].tolist()])

sc.tl.score_genes_cell_cycle(adata_oi, s_genes=s_genes, g2m_genes=g2m_genes, use_raw = False)

# %% [markdown]
# The "cycling" cluster are clearly tumor cells. I believe the one tiny "extension" in the non-cycling population might not be true lymphoma

# %%
# find genes that are conserved in cycling and non-cycling cells vs B-cells
lymphoma_vs_b.join(transcriptome.var[["symbol"]]).loc[lymphoma_vs_cycling.query("(scores > -1) & (scores < 1)").index].sort_values("scores", ascending = False).head(20)

# %%
sc.pl.umap(adata_oi, color = ["celltype", "S_score", "G2M_score", "phase"])
genes_to_plot = transcriptome.gene_id(["IGF2BP3", "NDFIP2", "AHR", "IRF8", "NFIA","MS4A1", "BCL2", "IRF4", "PAX5"])
sc.pl.umap(adata_oi, color = genes_to_plot, title = transcriptome.symbol(genes_to_plot), use_raw = False)

# %% [markdown]
# They are also clearly cycling cells, evidenced by MKI67

# %%
sc.pl.umap(adata_oi, color = [*transcriptome.gene_id(["MKI67", "CDK1", "PCNA", "BIRC5"])], use_raw = False)

# %% [markdown]
# The key TFs are clearly down

# %%
sc.pl.umap(adata_oi, color = ["celltype", "phase", "log_n_counts", *transcriptome.gene_id(["POU2F2", "ETV6", "IL2RA", "BLK", "TNFSF9"])], use_raw = False, layer = "normalized")

# %%
sc.pl.heatmap(adata_oi, [*transcriptome.gene_id(["POU2F2", "ETV6", "PAX5", "IL2RA"])], "celltype", use_raw = False, layer = "normalized")

# %%
fig, ax = plt.subplots()
ax.scatter(adata_oi.obs["S_score"], adata_oi.obs["G2M_score"], c = adata_oi.obs["cluster"])

# %% [markdown]
# ## Get differential slices peaks

# %%
# peakcaller = "macs2_leiden_0.1_merged"
peakcaller = "macs2_summits"
# peakcaller = "encode_screen"

# %%
scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / "t-test" / f"{clusters_oi[0]}_{clusters_oi[1]}scoring" / "regionpositional"
scoring_folder.mkdir(parents=True, exist_ok=True)

# %%
peakcounts = chd.flow.Flow.from_path(
    chd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / regions_name
)

# %%
# peakscores = []
# obs = pd.DataFrame({"cluster": pd.Categorical(clustering.labels)}, index=fragments.obs.index)

# # diffexp = "t-test-foldchange"
# diffexp = "t-test"


# def rank_genes_groups_df(
#     adata,
#     group,
#     *,
#     key: str = "rank_genes_groups",
#     colnames=("names", "scores", "logfoldchanges", "pvals", "pvals_adj"),
# ) -> pd.DataFrame:
#     """\
#     :func:`scanpy.tl.rank_genes_groups` results in the form of a
#     :class:`~pandas.DataFrame`.

#     Params
#     ------
#     adata
#         Object to get results from.
#     group
#         Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
#         argument) to return results from. Can be a list. All groups are
#         returned if groups is `None`.
#     key
#         Key differential expression groups were stored under.
        

#     Example
#     -------
#     >>> import scanpy as sc
#     >>> pbmc = sc.datasets.pbmc68k_reduced()
#     >>> sc.tl.rank_genes_groups(pbmc, groupby="louvain", use_raw=True)
#     >>> dedf = sc.get.rank_genes_groups_df(pbmc, group="0")
#     """
#     if isinstance(group, str):
#         group = [group]
#     if group is None:
#         group = list(adata.uns[key]["names"].dtype.names)

#     d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
#     d = pd.concat(d, axis=1, names=[None, "group"], keys=colnames)
#     d = d.stack(level=1).reset_index()
#     d["group"] = pd.Categorical(d["group"], categories=group)
#     d = d.sort_values(["group", "level_0"]).drop(columns="level_0")

#     # remove group column for backward compat if len(group) == 1
#     if len(group) == 1:
#         d.drop(columns="group", inplace=True)

#     return d.reset_index(drop=True)

# for region, _ in tqdm.tqdm(fragments.var.iterrows(), total=fragments.var.shape[0], leave=False):
#     var, counts = peakcounts.get_peak_counts(region)

#     if counts.sum() == 0:
#         print("no counts", dataset_name, regions_name, peakcaller, region)
#         continue

#     adata_atac = sc.AnnData(
#         counts.astype(np.float32),
#         obs=obs,
#         var=pd.DataFrame(index=var.index),
#     )
#     adata_atac = adata_atac[obs["cluster"].isin(clusters_oi), :]
#     # adata_atac = adata_atac[(counts > 0).any(1), :].copy()
#     import warnings

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         sc.pp.normalize_total(adata_atac)
#     sc.pp.log1p(adata_atac)

#     if diffexp in ["t-test", "t-test-foldchange"]:
#         sc.tl.rank_genes_groups(
#             adata_atac,
#             "cluster",
#             method="t-test",
#             max_iter=500,
#         )
#         columns = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
#     elif diffexp == "wilcoxon":
#         sc.tl.rank_genes_groups(
#             adata_atac,
#             "cluster",
#             method="wilcoxon",
#             max_iter=500,
#         )
#         columns = ["names", "scores"]
#     elif diffexp == "logreg":
#         sc.tl.rank_genes_groups(
#             adata_atac,
#             "cluster",
#             method="logreg",
#             max_iter=500,
#         )
#         columns = ["names", "scores"]
#     else:
#         print("Not supported diffexp ", diffexp)
#         continue

#     for cluster_oi in clusters_oi:
#         peakscores_cluster = (
#             rank_genes_groups_df(adata_atac, group=cluster_oi, colnames=columns)
#             .rename(columns={"names": "peak", "scores": "score"})
#             .set_index("peak")
#             .assign(cluster=cluster_oi)
#         )
#         peakscores_cluster = var.join(peakscores_cluster).sort_values("score", ascending=False)
#         # peakscores_cluster = peakscores_cluster.query("score > @min_score")
#         peakscores.append(peakscores_cluster)
# peakscores = pd.concat(peakscores)
# peakscores["cluster"] = pd.Categorical(peakscores["cluster"], categories=clustering.var.index)
# peakscores["length"] = peakscores["end"] - peakscores["start"]
# if "region" not in peakscores.columns:
#     peakscores["region"] = peakscores["gene"]

# peakscores["region_ix"] = fragments.var.index.get_indexer(peakscores["region"])
# peakscores["cluster_ix"] = clustering.var.index.get_indexer(peakscores["cluster"])

# differential_slices_peak = chd.models.diff.interpret.regionpositional.DifferentialPeaks(
#     peakscores["region_ix"].values,
#     peakscores["cluster_ix"].values,
#     peakscores["relative_start"],
#     peakscores["relative_end"],
#     data=peakscores["logfoldchanges"] if diffexp.endswith("foldchange") else peakscores["score"],
#     n_regions=fragments.regions.n_regions,
# )

# pickle.dump(differential_slices_peak, open(scoring_folder / "differential_slices.pkl", "wb"))

# differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
# differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
# differential_slices_peak.window = fragments.regions.window

# %%
differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))
differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
differential_slices_peak.window = fragments.regions.window

# %%
# match # of differential within each cluster
slicescores_peak_full = differential_slices_peak.get_slice_scores(regions = fragments.regions, clustering = clustering)
slicescores_peak_full = slicescores_peak_full.loc[slicescores_peak_full["cluster"].isin(clusters_oi)]
slicescores_peak_full["cluster"] = pd.Categorical(slicescores_peak_full["cluster"], categories = clusters_oi)
slicescores_peak = []
for cluster in n_desired_positions.index:
    peakscores_cluster = slicescores_peak_full.query("cluster == @cluster")
    peakscores_cluster = peakscores_cluster.sort_values("score", ascending=False)
    # peakscores_cluster = peakscores_cluster.sort_values("logfoldchanges", ascending=False)
    n_desired_positions_cluster = n_desired_positions[cluster]

    # peakscores_cluster["cumulative_length"] = peakscores_cluster["length"].cumsum() # at the latest as large
    peakscores_cluster["cumulative_length"] = np.pad(peakscores_cluster["length"].cumsum()[:-1], (1, 0)) # at least as large

    peakscores_cluster = peakscores_cluster.query("cumulative_length <= @n_desired_positions_cluster")
    slicescores_peak.append(peakscores_cluster)
slicescores_peak = pd.concat(slicescores_peak)
slicescores_peak["slice"] = pd.Categorical(slicescores_peak["region"].astype(str) + ":" + slicescores_peak["start"].astype(str) + "-" + slicescores_peak["end"].astype(str))
slices_peak = slicescores_peak.groupby("slice")[["region", "start", "end"]].first()

# %%
pd.DataFrame({
    "chd":slicescores.groupby("cluster")["length"].sum().sort_values(ascending = False),
    "peak":slicescores_peak.groupby("cluster")["length"].sum().sort_values(ascending = False),
})

# %%
slicecounts_peak = motifscan.count_slices(slices_peak)
enrichment_peak = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores_peak, slicecounts_peak)
enrichment_peak["log_odds"] = np.log(enrichment_peak["odds"])

# %%
enrichment_joined = enrichment.loc[clusters_oi[0]].loc[motifscan.motifs.loc[enrichment.loc[clusters_oi[0]].index, "gene"].dropna().index]
enrichment_joined = enrichment_joined.join(motifscan.motifs.loc[enrichment_joined.index, "gene"])
enrichment_joined[["scores", "pvals_adj", "logfoldchanges"]] = lymphoma_vs_cycling[["scores", "pvals_adj", "logfoldchanges"]].loc[enrichment_joined["gene"]].values

enrichment_peak_joined = enrichment_peak.loc[clusters_oi[0]].loc[motifscan.motifs.loc[enrichment_peak.loc[clusters_oi[0]].index, "gene"].dropna().index]
enrichment_peak_joined = enrichment_peak_joined.join(motifscan.motifs.loc[enrichment_peak_joined.index, "gene"])
enrichment_peak_joined[["scores", "pvals_adj", "logfoldchanges"]] = lymphoma_vs_cycling[["scores", "pvals_adj", "logfoldchanges"]].loc[enrichment_peak_joined["gene"]].values

# %%
enrichment_joined.loc[enrichment_joined["q_value"] > 0.05, "log_odds"] = 0.
enrichment_peak_joined.loc[enrichment_peak_joined["q_value"] > 0.05, "log_odds"] = 0.

# %%
# score_cutoff = 3
# score_cutoff = np.log(5.)
score_cutoff = 0
(
    np.exp(enrichment_joined.query("pvals_adj < 0.05").query("scores > @score_cutoff")["log_odds"].mean() - enrichment_joined.query("pvals_adj < 0.05").query("scores < -@score_cutoff")["log_odds"].mean()),
    np.exp(enrichment_peak_joined.query("pvals_adj < 0.05").query("scores > @score_cutoff")["log_odds"].mean() - enrichment_peak_joined.query("pvals_adj < 0.05").query("scores < -@score_cutoff")["log_odds"].mean())
)

# lfc_cutoff = np.log2(2.)
# (
#     np.exp(enrichment_joined.query("pvals_adj < 0.05").query("logfoldchanges > @lfc_cutoff")["log_odds"].mean() - enrichment_joined.query("pvals_adj < 0.05").query("logfoldchanges < -@lfc_cutoff")["log_odds"].mean()),
#     np.exp(enrichment_peak_joined.query("pvals_adj < 0.05").query("logfoldchanges > @lfc_cutoff")["log_odds"].mean() - enrichment_peak_joined.query("pvals_adj < 0.05").query("logfoldchanges < -@lfc_cutoff")["log_odds"].mean())
# )

# %%
enrichment_peak_joined.query("pvals_adj < 0.05").sort_values("logfoldchanges").iloc[:20]["log_odds"].mean() + enrichment_peak_joined.query("pvals_adj < 0.05").sort_values("logfoldchanges", ascending = False).iloc[:20]["log_odds"].mean()

# %%
enrichment_joined.query("pvals_adj < 0.05").sort_values("logfoldchanges").iloc[:20]

# %%
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"].str.contains("HMG")]
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"].str.contains("IRF4")]
motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"].str.contains("GATA1")]
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"].str.contains("POU2F2")]
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"].str.contains("ETV6")]
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"].str.contains("TAL1")]

# %%
enrichment_peak.loc[enrichment_peak.index.get_level_values("motif").isin(motifs_oi)].join(enrichment.loc[enrichment.index.get_level_values("motif").isin(motifs_oi)], lsuffix = "_peak", rsuffix = "_chd")[["log_odds_peak", "log_odds_chd", "q_value_chd", "q_value_peak"]].sort_values("log_odds_chd").style.background_gradient(vmin = -2, vmax = 2, cmap = "RdBu_r")

# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
fig, ax = plt.subplots()
plt.scatter(enrichment["log_odds"], enrichment_peak["log_odds"])
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axvline(0, color = "grey")
ax.axhline(0, color = "grey")
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
ax.set_aspect(1)

# %%
if dataset_name == "liver":
    motifscan.motifs["symbol"] = motifscan.motifs["MOUSE_gene_symbol"]
else:
    motifscan.motifs["symbol"] = motifscan.motifs["HUMAN_gene_symbol"]

# %%
motifs_oi = motifscan.motifs.sort_values("quality").copy().reset_index().groupby("symbol").first().reset_index().set_index("motif")
motifs_oi["gene"] = [transcriptome.gene_id(symbol) if symbol in transcriptome.var["symbol"].tolist() else None for symbol in motifs_oi["symbol"]]
motifs_oi = motifs_oi.dropna(subset=["gene"])
len(motifs_oi)

# %%
# adata_raw = transcriptome.adata.raw.to_adata()
# X = np.array(adata_raw.X.todense())
# X = X / X.sum(1, keepdims=True) * X.sum(1).mean()
# X = np.log(X + 1)
# X = pd.DataFrame(X, index = adata_raw.obs.index, columns = adata_raw.var.index)
# cluster_transcriptome = X.groupby(clustering.labels).mean()
# diffexp = cluster_transcriptome - cluster_transcriptome.mean(0)

# %%
cluster_transcriptome = pd.DataFrame(transcriptome.layers["magic"][:], index = transcriptome.obs.index, columns = transcriptome.var.index).groupby(clustering.labels).mean()
diffexp = cluster_transcriptome - cluster_transcriptome.mean(0)
diffexp = diffexp.T.unstack().to_frame("score")

# %%
# t-test
import scanpy as sc
adata_raw = transcriptome.adata.raw.to_adata()
adata_raw.obs["cluster"] = clustering.labels
adata_raw = adata_raw[:, transcriptome.adata.var.index]
adata_raw = adata_raw[adata_raw.obs["cluster"].isin(clusters_oi)]
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
import scanpy as sc
sc.tl.rank_genes_groups(adata_raw, groupby="cluster", method="wilcoxon", groups = clusters_oi)

diffexp = sc.get.rank_genes_groups_df(adata_raw, None).rename(columns = {"names":"gene", "group":"cluster"}).set_index(["cluster", "gene"])
diffexp["significant_up"] = (diffexp["pvals_adj"] < 0.01) & (diffexp["scores"] > 10)
diffexp["significant_down"] = (diffexp["pvals_adj"] < 0.01) & (diffexp["scores"] < -10)
diffexp["significant"] = diffexp["significant_up"] | diffexp["significant_down"]
diffexp["score"] = diffexp["scores"]

# %%
enrichment["gene"] = motifs_oi["gene"].reindex(enrichment.index.get_level_values("motif")).values
enrichment_peak["gene"] = motifs_oi["gene"].reindex(enrichment_peak.index.get_level_values("motif")).values

# %%
geneclustermapping = diffexp.T.unstack().to_frame("score")

# %%
diffexp_oi = diffexp.reindex(enrichment.reset_index().set_index(["cluster", "gene"]).index)
diffexp_oi.index = enrichment.index
enrichment[diffexp_oi.columns] = diffexp_oi

diffexp_oi = diffexp.reindex(enrichment_peak.reset_index().set_index(["cluster", "gene"]).index)
diffexp_oi.index = enrichment_peak.index
enrichment_peak[diffexp_oi.columns] = diffexp_oi

# %%
enrichment["aggregate"] = enrichment["log_odds"] * enrichment["score"]
enrichment_peak["aggregate"] = enrichment_peak["log_odds"] * enrichment_peak["score"]
enrichment.sort_values("aggregate", ascending = False).query("odds > 1").head(50)

# %%
fig, ax = plt.subplots()
norm = mpl.colors.Normalize(vmin = -1, vmax = 1)
cmap = mpl.cm.RdBu_r

plotdata = enrichment.join(enrichment_peak, lsuffix = "_chd", rsuffix = "_peak")
plotdata = plotdata.loc[(plotdata["score_chd"] > np.log(1.2)) | (plotdata["score_chd"] < np.log(1.2))]
# plotdata = plotdata.loc[(plotdata["score_chd"] > 10) | (plotdata["score_chd"] < -10)]
# plotdata = plotdata.loc[plotdata["score_chd"] > 10]
# plotdata = plotdata.loc[plotdata["score_chd"] < -10]
# plotdata = plotdata.loc[plotdata["score_chd"] < -np.log(1.2)]

plt.scatter(plotdata["log_odds_chd"], plotdata["log_odds_peak"], c = cmap(norm(plotdata["score_chd"])), s = 2)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axvline(0, color = "grey")
ax.axhline(0, color = "grey")
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
ax.set_aspect(1)

# %% [markdown]
# ### GRN capture

# %%
import chromatinhd_manuscript as chdm

# %%
# motifclustermapping = chdm.motifclustermapping.get_motifclustermapping(dataset_name, motifscan, clustering)

enrichment_q_value_cutoff = 0.01
enrichment_odds_cutoff = 1.5
n_top_motifs = 20
motifclustermapping = []
for ct in tqdm.tqdm(clusters_oi):
    motifs_oi = (
        enrichment.loc[ct]
        # enrichment_peak.loc[ct]
        .query("q_value < @enrichment_q_value_cutoff")
        .query("odds > @enrichment_odds_cutoff")
        .sort_values("odds", ascending=False)
        .head(n_top_motifs)
        .index
    )
    motifclustermapping.append(pd.DataFrame({"cluster":ct, "motif":motifs_oi}))
motifclustermapping = pd.concat(motifclustermapping)
motifclustermapping = motifclustermapping.query("cluster != 'Stromal'")

# %%
motifclustermapping = pd.DataFrame([
    *[["Lymphoma", motif] for motif in enrichment.sort_values("aggregate", ascending = False).query("odds > 1").loc["Lymphoma"].head(10).index],
    # *[["Lymphoma", motifscan.select_motif(symbol = symbol)] for symbol in ["TCF4", "POU2F2", "ETV6", "AHR", "BATF", "ZEB2", "IRF8", "GABPA", "MEIS2", "TP63"]],
    # ["Lymphoma", motifscan.select_motif("PO2F2")],
    # ["Lymphoma", motifscan.select_motif("ITF2")],
    # ["Lymphoma", motifscan.select_motif(symbol = "NFATC1")],
    # ["Lymphoma", motifscan.select_motif("NFAC1.H12CORE.1.PS.A")],
    ["Lymphoma", motifscan.select_motif("P63")],
    ["Lymphoma", motifscan.select_motif("PAX5")],
    # ["B", motifscan.select_motif("COE1")],
    ["B", motifscan.select_motif("MEF2C")],
], columns = ["cluster", "motif"])

# %%
np.exp(enrichment.loc[pd.MultiIndex.from_frame(motifclustermapping)]["log_odds"].mean() -  enrichment_peak.loc[pd.MultiIndex.from_frame(motifclustermapping)]["log_odds"].mean())

# %%
diffexp.loc["Lymphoma"].join(diffexp.loc["B"], lsuffix = "_Lymphoma", rsuffix = "_B")

# %%
enrichment.loc[pd.MultiIndex.from_frame(motifclustermapping)]["log_odds"]

# %%
fig, ax = plt.subplots()
plt.scatter(enrichment["log_odds"], enrichment_peak["log_odds"])
plt.scatter(enrichment.loc[pd.MultiIndex.from_frame(motifclustermapping)]["log_odds"], enrichment_peak.loc[pd.MultiIndex.from_frame(motifclustermapping)]["log_odds"], color = "red")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axvline(0, color = "grey")
ax.axhline(0, color = "grey")
ax.set_ylabel("log odds peak")
ax.set_xlabel("log odds chd")
# diagonal
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
ax.axline(xy1 = (0, 0), slope = np.log(2))
ax.set_aspect(1)

# %%
# expression_lfc_cutoff = 0.
# expression_lfc_cutoff = -np.inf
# expression_lfc_cutoff = np.log(1.2)
# expression_lfc_cutoff = np.log(1.1)

score_cutoff = -np.inf
# score_cutoff = 0.
# score_cutoff = 10

# %%
enrichment["expected"] = enrichment["contingency"].str[0].str[1] / enrichment["contingency"].str[0].str[0]
enrichment_peak["expected"] = enrichment_peak["contingency"].str[0].str[1] / enrichment_peak["contingency"].str[0].str[0]


# %%
def enrich_per_gene(motifclustermapping, diffexp, enrichment, slicescores, slicecounts, score_cutoff = -np.inf, ):
    founds = []
    for ct in tqdm.tqdm(motifclustermapping["cluster"].unique()):
        motifs_oi = motifclustermapping.query("cluster == @ct")["motif"]
        # genes_oi = diffexp.columns[(diffexp.idxmax() == ct) & (diffexp.max() > expression_lfc_cutoff)]
        # genes_oi = diffexp.columns[diffexp.loc[ct] > expression_lfc_cutoff]
        genes_oi = diffexp.loc[ct].query("score > @score_cutoff").index

        slicescores_foreground = slicescores.query("cluster == @ct").query("region in @genes_oi")

        slicecounts_oi = slicecounts.loc[slicescores_foreground["slice"], motifs_oi]

        print(f"{len(motifs_oi)=} {len(genes_oi)=} {ct=}")

        found = slicecounts_oi.groupby(slicescores_foreground["region"].values.astype(str)).sum()
        found.index.name = "gene"
        found = found.reindex(genes_oi, fill_value = 0)
        expected = (slicescores_foreground.groupby("region")["length"].sum().reindex(genes_oi).values[:, None] * enrichment.loc[ct].loc[motifs_oi]["expected"].values[None, :]) + 1e-4
        ratio = found / expected
        
        found = found.unstack().to_frame(name = "found")
        ratio = ratio.unstack().to_frame(name = "ratio")
        ratio["expected"] = expected.flatten()

        found = found.join(ratio).assign(cluster = ct)

        if len(found):
            founds.append(found)
    founds = pd.concat(founds).reset_index().set_index(["cluster", "gene", "motif"])
    return founds


# %%
founds = enrich_per_gene(motifclustermapping, diffexp, enrichment, slicescores, slicecounts, score_cutoff = score_cutoff)

# %%
founds_peak = enrich_per_gene(motifclustermapping, diffexp, enrichment_peak, slicescores_peak, slicecounts_peak, score_cutoff = score_cutoff)

# %%
founds["ratio"].mean()

# %%
# founds["diffexp"] = diffexp.T.unstack().loc[founds.index.droplevel("motif")].values
# founds_peak["diffexp"] = diffexp.T.unstack().loc[founds_peak.index.droplevel("motif")].values

founds["diffexp"] = diffexp["score"].loc[founds.index.droplevel("motif")].values
founds_peak["diffexp"] = diffexp["score"].loc[founds_peak.index.droplevel("motif")].values

# %%
# sc.pl.umap(transcriptome.adata, color = ["celltype", "ENSG00000102145"], layer = "magic", legend_loc = "on data")

# %%
pd.DataFrame({"chd":founds.groupby(["cluster", "motif"])["ratio"].mean(), "peak":founds_peak.groupby(["cluster", "motif"])["ratio"].mean()}).T.style.bar()

# %%
founds["ratio"].sum() / founds_peak["ratio"].sum()

# %%
founds["ratio"].sum(), founds_peak["ratio"].sum(), founds["ratio"].sum() / founds_peak["ratio"].sum()

# %%
fig, ax = plt.subplots()
plt.scatter(founds["diffexp"], np.log1p(founds["ratio"]))
ax.axhline(np.log(2), color = "grey")

# %%
founds["detected"] = founds["ratio"] > 1
founds_peak["detected"] = founds_peak["ratio"] > 1

# %%
founds["detected"].mean(), founds_peak["detected"].mean(), founds["detected"].mean()/founds_peak["detected"].mean()

# %%
diffexp["symbol"] = transcriptome.var["symbol"].reindex(diffexp.index.get_level_values("gene")).values

# %%
fig, ax = plt.subplots()
ax.scatter(founds["diffexp"], np.log1p(founds["ratio"]))

# %%
diffexp.loc["Lymphoma"].sort_values("score", ascending = False).head(50)

# %%
founds.loc["Lymphoma"].loc[transcriptome.gene_id(["TCF4", "POU2F2", "NFATC1"])].join(transcriptome.var[["symbol"]]).reset_index().set_index(["symbol", "motif"])["ratio"].unstack() > 1

# %%
founds.join(founds_peak, rsuffix = "_peak").loc["Lymphoma"].loc[diffexp.loc["Lymphoma"].sort_values("score", ascending = False).head(50).index].sort_values("ratio", ascending = False).join(transcriptome.var[["symbol"]]).head(30)

# %%
# gene_oi = region_id = transcriptome.gene_id("KLF1")
# window = [-5000, 5000]

# gene_oi = region_id = transcriptome.gene_id("GATA1")
# window = [-10000, 20000]

# gene_oi = region_id = transcriptome.gene_id("HBB")
# window = [-20000, 20000]

# gene_oi = region_id = transcriptome.gene_id("CCL4")
# window = [-1000, 1000]

# gene_oi = region_id = transcriptome.gene_id("QKI")
# window = [-10000, 10000]

gene_oi = region_id = transcriptome.gene_id("IRF4")
# gene_oi = region_id = "ENSG00000196247"
window = [-100000, 100000]

founds.join(founds_peak, rsuffix = "_peak").query("gene == @gene_oi").groupby(["cluster", "gene", "motif"]).sum()

# %%
founds.join(founds_peak, rsuffix = "_peak").query("gene == @gene_oi").sort_values("ratio", ascending = False)

# %%
# motifs_oi = pd.DataFrame([
#     [motifscan.select_motif("GATA1"), "Gata1"],
#     [motifscan.select_motif("TAL1"), "Tal1"],
#     [motifscan.select_motif("STA5"), "Stat5"],
#     *[[motif, "Stat5"] for motif in motifscan.select_motifs("STA5")],
#     [motifscan.select_motif("KLF1"), "klf1"],
# ], columns = ["motif", "symbol"]).set_index("motif")

motifs_oi = pd.DataFrame({"motif":motifclustermapping.loc[motifclustermapping["cluster"] == diffexp.xs(gene_oi, level = 'gene')["score"].idxmax(), "motif"]}).reset_index().set_index("motif")

# %%
import chromatinhd.data.associations
import chromatinhd.data.associations.plot

# %%
motifscan_name = "gwas_immune_main"
# motifscan_name = "gtex_immune"
associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

resolution = 1/5000
width = (window[1] - window[0]) *resolution

region = fragments.regions.coordinates.loc[region_id]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10" if dataset_name == "liver" else "GRCh38")
fig.main.add_under(panel_genes)

relative_to = "B"
cluster_info = clustering.cluster_info.loc[clusters_oi]
plotdata, plotdata_mean = regionpositional.get_plotdata(region_id)

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome=transcriptome, clustering=clustering, gene=region_id, panel_height=0.4, order = True, cluster_info = cluster_info
)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=cluster_info, panel_height=0.4, width=width, window = window, order = panel_expression.order, ymax = 5, relative_to =relative_to,
)

fig.main.add_under(panel_differential)
fig.main.add_right(panel_expression, row=panel_differential)

panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, region_id, motifs_oi = motifs_oi, width = width, window = window)
fig.main.add_under(panel_motifs)

# panel_association = chd.data.associations.plot.Associations(associations, region_id, width = width, window = window, show_ld = False)
# fig.main.add_under(panel_association)

import chromatinhd_manuscript as chdm
panel_peaks = chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = window, width = width)
fig.main.add_under(panel_peaks)

fig.plot()

# %%
f"{fragments.regions.coordinates.loc[region_id]['chrom']}:{fragments.regions.coordinates.loc[region_id]['start']}-{fragments.regions.coordinates.loc[region_id]['end']}"

# %%
slicescores.query("cluster == 'Lymphoma'")["length"].sum()

# %%
slicescores.query("region == @region_id").query("cluster == 'Lymphoma'")["length"].sum()

# %% [markdown]
# ## Global GO enrichment

# %%
import gseapy

# %%
var = transcriptome.adata.raw.var.reset_index().groupby("symbol").first().reset_index().set_index("gene")

# %%
genesets = gseapy.get_library(name='GO_Biological_Process_2023', organism='Human')
# genesets = gseapy.get_library(name='KEGG_2016', organism='Human')
# genesets = {name:transcriptome.gene_id(genes, optional = True).tolist() for name, genes in genesets.items()}
genesets = {name:var.set_index("symbol").reindex(genes)["gene_ids"].dropna().tolist() for name, genes in genesets.items()}

# %%
gset_info = pd.DataFrame({
    "gset":pd.Series(genesets.keys()),
    # "label":pd.Series(genesets.keys()).str.split("GO:").str[0].str[:-2],
    # "go":"GO:" + pd.Series(genesets.keys()).str.split("GO:").str[1].str[:-1],
    "n":[len(gset) for gset in genesets.values()],
}).sort_values("n", ascending = False)


# %%
def find_go(xs):
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    matches = np.stack([gset_info["label"].str.contains(x, case = False) for x in xs])
    gset_info["matches"] = matches.sum(0)
    if gset_info["matches"].max() == 0:
        raise ValueError("No matches found")
    return gset_info.sort_values(["matches", "n"], ascending = False)["gset"].iloc[0]


# %%
genes_oi = diffexp.loc["Lymphoma"].query("score > 3").sort_values("score").index.tolist()
# genes_oi = diffexp.loc["Lymphoma"].query("(logfoldchanges > 1) & (pvals_adj < 0.0001)").sort_values("score").index.tolist()
# background = diffexp.loc["Lymphoma"].index.tolist()
background = var.index

len(genes_oi)

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata, color = [genes_oi[0], transcriptome.gene_id("IRF8"), "celltype"], use_raw = False)

# %%
goenrichment = []
import fisher
for gset_id, gset in genesets.items():
    gset = set(gset)
    n_genes = len(gset)
    n_genes_oi = len(set(gset).intersection(genes_oi))
    n_background = len(background)
    n_background_oi = len(set(background).intersection(genes_oi))
    pvalue = fisher.pvalue(n_background - n_background_oi - n_genes_oi, n_background_oi, n_genes - n_genes_oi, n_genes_oi).two_tail
    odds = (n_genes_oi * n_background) / (n_background_oi * n_genes+1e-4)
    goenrichment.append([gset_id, n_genes, n_genes_oi, n_background, n_background_oi, odds, pvalue])
goenrichment = pd.DataFrame(goenrichment, columns = ["gset", "n_genes", "n_found", "n_background", "n_background_oi", "odds", "p_value"]).set_index("gset")
goenrichment = goenrichment.loc[goenrichment["n_found"] > 1]
goenrichment["q_value"] = chd.utils.fdr(goenrichment.query("n_found > 5")["p_value"])

import statsmodels.api as sm
goenrichment.loc[goenrichment["n_found"] > 5, "q_value2"] = sm.stats.fdrcorrection(goenrichment.loc[goenrichment["n_found"] > 5, "p_value"], alpha=0.05, method='indep', is_sorted=False)[1]

# %%
# goenrichment = gseapy.enrich(genes_oi, background = background, gene_sets = genesets, top_term = 100000)
# goenrichment = goenrichment.res2d.sort_values("Adjusted P-value").set_index("Term")
# goenrichment["n_found"] = goenrichment["Overlap"].str.split("/").str[0].astype(int)
# goenrichment["q_value"] = chd.utils.fdr(goenrichment.query("n_found > 5")["P-value"])

# %%
goenrichment.sort_values("q_value").head(50)

# %% [markdown]
# ## VS

# %%
motif_1 = motifscan.select_motif("PO2F2")
# motif_1 = motifscan.select_motif("ITF2")
# motif_2 = motifscan.select_motif("PAX5")
motif_2 = motifscan.select_motif("NFAC1.H12CORE.1.PS.A")

# founds_oi_1 = founds.loc["Lymphoma"].loc[genes_oi].xs(motif_1, level = "motif")
# founds_oi_2 = founds.loc["Lymphoma"].loc[genes_oi].xs(motif_2, level = "motif")
founds_oi_1 = founds.loc["Lymphoma"].xs(motif_1, level = "motif")
founds_oi_2 = founds.loc["Lymphoma"].xs(motif_2, level = "motif")
(founds_oi_1["found"].sum() / founds_oi_1["expected"].sum()), (founds_oi_2["found"].sum() / founds_oi_2["expected"].sum())

# %%
# founds["positions"] = slicescores.groupby(["cluster", "region"])["length"].sum().reindex(founds.index.droplevel("motif")).values
# founds["enrichment"] = np.log1p((founds["found"] / founds["positions"] * 10e3).fillna(0.))
# founds["enrichment"] = (founds["found"] / slicescores.groupby(["cluster", "region"])["length"].sum().reindex(founds.index.droplevel("motif")).values * 10e3).fillna(0.)

# %%
scores = []
for gset_id in goenrichment.sort_values("q_value").query("q_value < 0.05").head(50).index:
    gset = genesets[gset_id]
    gset = [g for g in gset if g in transcriptome.var.index]
    gset = [g for g in gset if g in founds_oi_1.index]
    gset = [g for g in gset if g in founds_oi_2.index]
    enriched1 = (founds_oi_1.loc[gset]["found"].sum() / founds_oi_1.loc[gset]["expected"].sum())# / (founds_oi_1["found"].sum() / founds_oi_1["expected"].sum())
    enriched2 = (founds_oi_2.loc[gset]["found"].sum() / founds_oi_2.loc[gset]["expected"].sum())# / (founds_oi_2["found"].sum() / founds_oi_2["expected"].sum())

    scores.append({
        "gset":gset_id,
        "found1":(founds_oi_1.loc[gset]["found"].sum() / founds_oi_1.loc[gset]["expected"].sum()),
        "detected1":founds_oi_1.loc[gset]["detected"].sum(),
        "expected1":founds_oi_1.loc[gset]["expected"].sum(),
        "enriched1":enriched1,
        "found2":(founds_oi_2.loc[gset]["found"].sum() / founds_oi_2.loc[gset]["expected"].sum()),
        "detected2":founds_oi_2.loc[gset]["detected"].sum(),
        "expected2":founds_oi_2.loc[gset]["expected"].sum(),
        "enriched2":enriched2,
    })
scores = pd.DataFrame(scores).set_index("gset")

# %%
fig, ax = plt.subplots()
ax.scatter(scores["enriched1"], scores["enriched2"])
# ax.scatter(scores["found1"], scores["found2"])

# %%
scores.sort_values("enriched1", ascending = False)

# %%
founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2").loc[gset]

# %%
fig, ax = plt.subplots()
ax.scatter(founds_oi_1["enrichment"], founds_oi_2["enrichment"])

# %%
sns.heatmap(np.log10(1+founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2").groupby(["found_1", "found_2"]).size().unstack()), annot = True)

# %%
import gseapy

# %%
gsets = founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2").reset_index().groupby(["found_1", "found_2"])["gene"].apply(list).to_frame("genes").reset_index()
gsets["n"] = gsets["genes"].apply(len)
gsets

# %% [markdown]
# ### Multiple

# %%
dat = founds.loc["Lymphoma"]["ratio"].unstack().loc[genes_oi]
# dat = founds_peak.loc["Lymphoma"]["ratio"].unstack().loc[genes_oi]

# cluster dat
import scipy.cluster.hierarchy
import scipy.spatial.distance
hierarchy = scipy.cluster.hierarchy.linkage(np.log1p(dat), method = "ward")
dendro = scipy.cluster.hierarchy.dendrogram(hierarchy, no_plot = True)

# get order of leaves
order = scipy.cluster.hierarchy.leaves_list(hierarchy)
dat = dat.iloc[order]
fig, ax = plt.subplots()
ax.matshow(dat, cmap = "RdBu_r", vmin = 1, vmax = 3, aspect = "auto")
ax.set_xticks(np.arange(dat.shape[1]))
ax.set_xticklabels(dat.columns, rotation = 90)

# %%
gsets_oi = goenrichment.sort_values("q_value").query("q_value < 0.05").head(50).index


# %%
def enrich(genes_oi, background, genesets):
    goenrichment = []
    import fisher
    genes_oi = set(genes_oi)
    background = set(background)
    # for gset_id, gset in genesets.items():
    for gset_id in gsets_oi:
        gset = set(genesets[gset_id])
        gset = set(set(gset).intersection(background))
        overlap = gset.intersection(genes_oi)
        n_genes = len(gset)
        n_genes_oi = len(gset.intersection(genes_oi))
        n_background = len(background)
        n_background_oi = len(set(background).intersection(genes_oi))
        contingency = np.array([[n_background - n_background_oi, n_background_oi], [n_genes - n_genes_oi, n_genes_oi]])
        pvalue = fisher.pvalue(*contingency.flatten()).two_tail
        odds = (contingency[1, 1] * contingency[0, 0]) / (contingency[0, 1] * contingency[1, 0]+1e-4)
        if len(overlap) > 0:
            symbols = transcriptome.symbol(list(overlap)).tolist()
        else:
            symbols = []
        goenrichment.append([gset_id, n_genes, n_genes_oi, n_background, n_background_oi, odds, pvalue, contingency, symbols])
    goenrichment = pd.DataFrame(goenrichment, columns = ["gset", "n_genes", "n_found", "n_background", "n_background_oi", "odds", "p_value", "contingency", "overlap"]).set_index("gset")
    goenrichment["q_value"] = chd.utils.fdr(goenrichment.query("n_found > 3")["p_value"])

    import statsmodels.api as sm
    goenrichment.loc[goenrichment["n_found"] > 3, "q_value2"] = sm.stats.fdrcorrection(goenrichment.loc[goenrichment["n_found"] > 3, "p_value"], alpha=0.05, method='indep', is_sorted=False)[1]
    return goenrichment


# %%
background = dat.index
goenrichments = []
for motif, motifdat in dat.items():
    goenrichment = enrich(motifdat.index[motifdat > 1], background, genesets)
    goenrichment["motif"] = motif
    goenrichments.append(goenrichment)
goenrichments = pd.concat(goenrichments).reset_index().set_index(["motif", "gset"])

# %%
goenrichments.sort_values("odds", ascending = False).head(20)

# %%
# goenrichments.xs("Positive Regulation Of Transcription By RNA Polymerase II (GO:0045944)", level = "gset").sort_values("q_value", ascending = True)
goenrichments.xs("Regulation Of Canonical Wnt Signaling Pathway (GO:0060828)", level = "gset").sort_values("q_value", ascending = True)

# %%
gene_ids = transcriptome.gene_id(goenrichments.loc[motifscan.select_motif("PO2F2")].loc["Regulation Of Canonical Wnt Signaling Pathway (GO:0060828)"]["overlap"])
len(gene_ids)
print("c(" + ", ".join([f'"{gene_id}"' for gene_id in transcriptome.symbol(gene_ids)]) + ")")

# %%
sns.ecdfplot(founds.loc["Lymphoma"].loc[gene_ids].xs(motifscan.select_motif("PO2F2"), level = "motif")["ratio"])
sns.ecdfplot(founds_peak.loc["Lymphoma"].loc[gene_ids].xs(motifscan.select_motif("PO2F2"), level = "motif")["ratio"])

# %%
for gene_id in gene_ids:
    print(
        transcriptome.symbol(gene_id),
        founds.loc["Lymphoma"].loc[gene_id].sort_values("ratio", ascending = False).loc[motifscan.select_motif("PO2F2")]["ratio"],
        founds.loc["Lymphoma"].loc[gene_id].sort_values("ratio", ascending = False).loc[motifscan.select_motif("PO2F2")]["found"],
        founds_peak.loc["Lymphoma"].loc[gene_id].sort_values("ratio", ascending = False).loc[motifscan.select_motif("PO2F2")]["ratio"],
        founds_peak.loc["Lymphoma"].loc[gene_id].sort_values("ratio", ascending = False).loc[motifscan.select_motif("PO2F2")]["found"],
    )
print("--")


# %%
for gene_id in gene_ids:
    print(founds_peak.loc["Lymphoma"].loc[gene_id].sort_values("ratio", ascending = False).loc[motifscan.select_motif("PO2F2")]["ratio"])

# %%
founds_peak.loc["Lymphoma"].loc[transcriptome.gene_id("ZEB2")].sort_values("ratio", ascending = False)

# %%
goenrichments.sort_values("q_value").query("odds > 1").head(20)

# %% [markdown]
# ## Compare

# %%
