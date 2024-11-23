# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import torch

# %%
pp.paths.results().mkdir(parents=True, exist_ok=True)

# %%
diffexp_folder = pathlib.Path(
    "/home/wouters/fs4/u_mgu/private/JeanFrancois/For_Wouter/Alveolar_Mac/RNA/"
)

# %%
diffexp = pd.read_excel(diffexp_folder / "DEG.xlsx")
diffexp

# %%
dataset_name = "lung"
dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
regions_name = "100k100k"
regions = chd.data.Regions(dataset_folder / "regions" / regions_name)

# %%
motifscan_name = "hocomocov12" + "_" + "1e-4"
# motifscan_name = "hocomocov12" + "_" + "5"
genome_folder = pathlib.Path("/srv/data/genomes/GRCm39")
# genome_folder = chd.get_output() / "genomes" / "GRCm39"
motifscan_genome = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)

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
clustering = chd.data.Clustering(
    dataset_folder / "clusterings" / "cluster"
)

# %%
fragments = chd.data.fragments.FragmentsView(
    dataset_folder / "fragments" / regions_name,
    # overwrite=True,
)
regions.var = fragments.var

# %%
model_folder = (
    chd.get_output() / "diff" / "lung" / "binary" / "split" / regions_name / "cluster"
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
# motifscan_name = "hocomocov12_1e-4"
# # motifscan_name = "hocomocov12_5e-4"
# # motifscan_name = "hocomocov12_cutoff_5"
# motifscan = chd.data.motifscan.MotifscanView(
#     chd.get_output()
#     / "datasets"
#     / dataset_name
#     / "motifscans"
#     / regions_name
#     / motifscan_name
# )
# motifscan.parent = chd.data.motifscan.Motifscan(
#     "/srv/data/wouters/projects/ChromatinHD_manuscript/output/genomes/GRCm39/motifscans/"
#     + motifscan_name
# )
# motifscan.motifs["label"] = motifscan.motifs["HUMAN_gene_symbol"]
# clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
# this cutoff depends on your dataset
# typically 5% of the genome
slices = regionpositional.calculate_slices(5.0, step=25)
differential_slices = regionpositional.calculate_differential_slices(
    slices, fc_cutoff=1.5,
)

# %%
differential_slices.get_slice_scores(regions = regions, clustering = clustering)

# %%
# check if its about 5%
((slices.end_position_ixs - slices.start_position_ixs) * slices.step).sum() / (regions.coordinates["end"] - regions.coordinates["start"]).sum()

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
# ### Enrichment

# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)

# %% [markdown]
# #### Enrichment in gene regions

# %%
# count motifs in slices
slicecounts = motifscan.count_slices(slicescores)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
    slicescores, slicecounts
)

motifs_selected = motifscan.motifs.loc[motifscan.motifs.quality.isin(["A", "B"])]
enrichment = enrichment.query("motif in @motifs_selected.index")

# %%
enrichment.loc["IL33-AM"].query("q_value < 0.05").sort_values("odds")

# %%
slicecounts.index = pd.MultiIndex.from_frame(slicescores[["cluster", "region"]])
motifcounts_gene = slicecounts.groupby(level = (0, 1)).sum()

# %% [markdown]
# #### Store for each gene and TF group the putative target genes

# %%
# group_irf = enrichment_grouped["members"].apply(lambda x: any(["IRF4" in y for y in x]))
# members_irf = enrichment_grouped["members"].loc[group_irf.index[group_irf][0]]
members_irf = motifscan.motifs.query("tf == 'IRF4'").index
# group_batf = enrichment_grouped["members"].apply(lambda x: any(["BATF" in y for y in x]))
# members_batf = enrichment_grouped["members"].loc[group_batf.index[group_batf][0]]
members_batf = motifscan.motifs.query("tf in ['BATF', 'BATF3']").index
# group_pparg = enrichment_grouped["members"].apply(lambda x: any(["PPARG" in y for y in x]))
# members_pparg = enrichment_grouped["members"].loc[group_pparg.index[group_pparg][0]]
members_pparg = motifscan.motifs.query("tf in ['PPARG']").index

# %%
links = {
    "IRF4":motifcounts_gene.loc["IL33-AM"].index[motifcounts_gene.loc["IL33-AM"][members_irf].sum(1) > 0].tolist(),
    "BATF":motifcounts_gene.loc["IL33-AM"].index[motifcounts_gene.loc["IL33-AM"][members_batf].sum(1) > 0].tolist(),
    "PPARG":motifcounts_gene.loc["Ctrl-AM"].index[motifcounts_gene.loc["Ctrl-AM"][members_pparg].sum(1) > 0].tolist(),
}
import json
json.dump(links, open(pathlib.Path("/home/wouters/fs4/u_mgu/private/wouters/projects/spam/results").resolve() / "motif_gene_links.json", "w"))

links_symbol = {
    k:fragments.var.loc[v]["symbol"].tolist() for k, v in links.items()
}
json.dump(links_symbol, open(pathlib.Path("/home/wouters/fs4/u_mgu/private/wouters/projects/spam/results").resolve() / "motif_gene_links_symbol.json", "w"))

# %% [markdown]
# #### Enrichment in original regions

# %%
def map_slicescores_to_parent_regions(slicescores, regions):
    merged_slicescores = []
    slicescores2 = chd.data.regions.uncenter_multiple(slicescores, regions.coordinates)
    for cluster, clusterdata in slicescores2.groupby("cluster", observed=True):
        for chrom, chromdata in clusterdata.groupby("chrom"):
            chromdata2 = chromdata.sort_values("start_genome", ascending = True)
            chromdata2["overlap"] = (chromdata2["start_genome"].shift(-1) < chromdata2["end_genome"]).shift(1).fillna(0)
            chromdata2["overlap"] = chromdata2["overlap"].fillna(False)
            chromdata2["group"] = (~chromdata2["overlap"]).cumsum()

            chromdata3 = chromdata2.copy()[["chrom", "start_genome", "end_genome"]].rename(columns = {"start_genome":"start", "end_genome":"end"})

            chromdata3 = chromdata2.groupby("group").agg(
                chrom = ("chrom", "first"),
                start_genome = ("start_genome", "min"),
                end_genome = ("end_genome", "max")
            ).rename(columns = {"start_genome":"start", "end_genome":"end"})

            merged_slicescores.append(chromdata3.assign(cluster = cluster))
    merged_slicescores = pd.concat(merged_slicescores)
    merged_slicescores["cluster"] = pd.Categorical(merged_slicescores["cluster"], slicescores["cluster"].cat.categories)
    merged_slicescores["length"] = merged_slicescores["end"] - merged_slicescores["start"]
    merged_slicescores.index = (
        merged_slicescores["chrom"] + ":" + merged_slicescores["start"].astype(str) + "-" + merged_slicescores["end"].astype(str)
    )
    merged_slicescores.index.name = "slice"
    return merged_slicescores
merged_slicescores = map_slicescores_to_parent_regions(slicescores, fragments.regions)

# %%
merged_slicescores

# %%
# count motifs in slices
merged_slicescores["region_ix"] = motifscan_genome.regions.coordinates.index.get_indexer_for(merged_slicescores["chrom"])

slicecounts = motifscan_genome.count_slices(merged_slicescores)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
    merged_slicescores, slicecounts
)

motifs_selected = motifscan.motifs.loc[motifscan.motifs.quality.isin(["A", "B"])]
enrichment = enrichment.query("motif in @motifs_selected.index")

# %%
enrichment.loc["IL33-AM"].query("q_value < 0.05").sort_values(
    "odds", ascending=False
).head(30)

# %% [markdown]
# #### Calculate GC

# %%
import pysam
import genomepy
genome = genomepy.Genome("GRCm39")

# %%
fasta = pysam.FastaFile(genome.filename)
def get_gc(fasta, slicescores):
    def extract_gc(fasta, chrom, start, end, strand):
        """
        Extract GC content in a region
        """

        actual_start = start
        if actual_start < 0:
            actual_start = 0
        seq = fasta.fetch(chrom, actual_start, end)
        if strand == -1:
            seq = seq[::-1]

        if len(seq) == 0:
            return np.nan

        # gc = np.isin(np.array(list(seq)), ["c", "g", "C", "G"]).mean()

        gc = (seq.lower().count("cg") + seq.lower().count("gc")) / len(seq)

        return gc

    gcs = []
    for chrom, start, end in tqdm.tqdm(
        zip(
            slicescores.chrom,
            slicescores.start,
            slicescores.end,
        ),
        total=len(slicescores),
    ):
        gcs.append(extract_gc(fasta, chrom, start, end, 1))
    return gcs

# %%
merged_slicescores["gc"] = get_gc(fasta, merged_slicescores)

# %%
merged_slicescores.groupby("cluster", observed = True)["gc"].mean()

# %% [markdown]
# #### Calculate GC fits

# %%
motif = "CEBPB.H12CORE.1.SM.B"
motif = "KAISO.H12CORE.1.P.B"
motif = "KMT2A.H12CORE.0.P.B"
# motif = "JUND.H12CORE.0.PM.A"
# motif = "ZEB1.H12CORE.0.P.B"
# motif = "PPARG.H12CORE.0.P.B"
# motif = "HINFP.H12CORE.2.S.B"
merged_slicescores[motif] = slicecounts[motif] / merged_slicescores["length"]
np.corrcoef(merged_slicescores["gc"], merged_slicescores[motif])

def smooth_spline_fit_se(x, y, x_smooth):
    import rpy2.robjects as ro
    ro.globalenv["x"] = ro.FloatVector(x)
    ro.globalenv["y"] = ro.FloatVector(y)
    ro.globalenv["x_pred"] = ro.FloatVector(x_smooth)
    script = """
    # Install and load the mgcv package if not yet done
    if (!require(mgcv)) {
    install.packages('mgcv')
    library(mgcv)
    }

    # Fit a GAM with a smoothing spline, just like smooth.spline
    gam_model <- gam(y ~ s(x, sp = 0.5), method = 'REML')

    # Make predictions
    y_pred <- predict(gam_model, newdata = data.frame(x = x_pred), type = 'response', se.fit = TRUE)

    # Extract predicted values and standard errors
    # fit <- y_pred
    # se <- y_pred
    fit <- y_pred$fit
    se <- y_pred$se.fit

    list(fit = fit, se = se)
    """
    out = ro.r(script)
    fit = np.array(out[0])
    se = np.array(out[1])
    return np.stack([fit, se]).T
plt.scatter(merged_slicescores["gc"], merged_slicescores[motif])
x_smooth = np.linspace(0, 0.6, 500)
fit = smooth_spline_fit_se(merged_slicescores["gc"], merged_slicescores[motif], x_smooth)
fit[:, 0] = np.clip(fit[:, 0], 0, 1)
fig, ax = plt.subplots()
plt.plot(x_smooth, fit[:, 0])

# %%
merged_slicescores["expected"] = np.interp(merged_slicescores["gc"], x_smooth, fit[:, 0])

# %%
merged_slicescores.groupby("cluster")["expected"].mean(), merged_slicescores.groupby("cluster")[motif].mean()

# %%
fits = {}
x_smooth = np.linspace(0, 0.6, 100)
for motif in tqdm.tqdm(motifscan.motifs.index):
    try:
        scores = slicecounts[motif] / merged_slicescores["length"]
        if (scores == 0).all():
            fit = np.zeros((len(x_smooth), 2))
        else:
            fit = smooth_spline_fit_se(merged_slicescores["gc"], scores, x_smooth)
        fit[:, 0] = np.clip(fit[:, 0], 0, 1)
        fits[motif] = fit
    except KeyboardInterrupt:
        raise ValueError("Interrupted")
    
# %%
pickle.dump(fits, open(pp.paths.results() / "fits.pickle", "wb"))

# %% [markdown]
# #### Correct for gc

# %%
x_smooth = np.linspace(0, 0.6, 100)
fits = pickle.load(open(pp.paths.results() / "fits.pickle", "rb"))

# %%
fit = np.array([fit[:, 0] for k, fit in fits.items()])
expected = pd.DataFrame((np.array([np.interp(merged_slicescores["gc"], x_smooth, fit[i]) for i in range(fit.shape[0])]) * merged_slicescores["length"].values[None, :]).T, index = slicecounts.index, columns = slicecounts.columns)

# %%
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
    merged_slicescores, slicecounts, expected = expected
)

# %%
enrichment.query("q_value < 0.05").sort_values("odds", ascending=False).loc["Ctrl-AM"].head(20)

# %%
enrichment.loc["Ctrl-AM"].loc["KMT2A.H12CORE.0.P.B"]

# %%
enrichment.loc["IL33-AM"].query("q_value < 0.05").sort_values(
    "odds", ascending=False
).head(30)

# %%
enrichment_grouped = chd.models.diff.interpret.enrichment.group_enrichment(
    enrichment, slicecounts, clustering, merge_cutoff=0.2
)

# %%
enrichment_grouped.loc["IL33-AM"].query("q_value < 0.05").sort_values(
    "odds", ascending=False
).head(8)

# %%
enrichment_grouped.loc["Ctrl-AM"].query("q_value < 0.05").sort_values(
    "odds", ascending=False
).head(20)

# %%
enrichment.loc["Ctrl-AM"].loc[
    motifscan.motifs.index[
        motifscan.motifs.symbol.isin(["Pparg"])
        & motifscan.motifs.quality.isin(["A", "B"])
    ]
][["odds", "q_value"]]

# %%
enrichment["symbol"] = motifscan.motifs.loc[enrichment.index.get_level_values("motif")][
    "MOUSE_gene_symbol"
].values

# %% [markdown]
# ### Plot enrichment

# %%
enrichment_oi = enrichment.loc["IL33-AM"]
enrichment_oi = enrichment_oi.loc[enrichment_oi["symbol"].isin(diffexp["gene"])]
enrichment_oi["logFC"] = (
    diffexp.set_index("gene").loc[enrichment_oi["symbol"]]["logFC"].values
)
enrichment_oi = (
    enrichment_oi.sort_values("odds", ascending=False)
    .reset_index()
    .groupby("symbol", as_index=False)
    .first()
    .set_index("motif")
)

# %%
enrichment_grouped = enrichment_grouped.sort_values(
    ["cluster", "q_value"], ascending=False
)

# %%
groups_highlight_up = (
    enrichment_grouped.query("cluster == 'IL33-AM'")
    .sort_values("log_odds", ascending=False)
    # .sort_values("q_value", ascending=True)
    .head(8)
)
warm_colors = [
    "#FF0000",
    "#FFA500",
    "#CCCC00",
    "#e6ba93",
    "#FF7F50",
    "#FFBF00",
    "#FFD700",
    "#FF00FF",
    "#800000",
    "#B7410E",
]
groups_highlight_up["color"] = warm_colors[: len(groups_highlight_up)]
groups_highlight_down = (
    enrichment_grouped.query("cluster == 'Ctrl-AM'")
    .sort_values("log_odds", ascending=False)
    # .sort_values("q_value", ascending=True)
    .head(8)
    # .head(3)
)
groups_highlight_down = pd.concat([
    groups_highlight_down,
    enrichment_grouped.loc[[("Ctrl-AM", "PPARD.H12CORE.0.PSM.A")]]
]
)
cold_colors = [
    "#0000FF",
    "#00FFFF",
    "#008080",
    "#00FF00",
    "#008000",
    "#00FF7F",
    "#00FFD4",
    "#008B8B",
    "#000080",
    "#0000CD",
]
groups_highlight_down["color"] = cold_colors[: len(groups_highlight_down)]
groups_highlight = pd.concat(
    [
        groups_highlight_up,
        groups_highlight_down,
    ]
)

# %%
motifs_highlight = groups_highlight[["group", "members", "color"]].explode("members")
motifs_highlight

# %%
enrichment_oi["color"] = (
    enrichment_oi[[]]
    .join(motifs_highlight.set_index("members")["color"])["color"]
    .fillna("lightgray")
)
enrichment_oi["group"] = enrichment_oi[[]].join(
    motifs_highlight.set_index("members")["group"]
)["group"]

# %%
enrichment_oi["abs_logFC"] = np.abs(enrichment_oi["logFC"])
group_representatives = (
    enrichment_oi.dropna(subset=["group"])
    .sort_values("abs_logFC", ascending=False)
    .reset_index()
    .groupby("group")
    .first()
)

# remove groups with no representative TF
groups_highlight = groups_highlight.loc[groups_highlight.index.get_level_values("motif").isin(group_representatives.index)].copy()

groups_highlight["label"] = group_representatives.loc[
    groups_highlight.index.get_level_values("motif")
]["symbol"].values
enrichment_highlight = group_representatives
enrichment_highlight.sort_values("odds").groupby("group")["odds"].max()

# %%
fig = pp.grid.Figure()
ax = fig.main.add(pp.grid.Panel((2, 2)))
alpha = 0.4 + (enrichment_oi.index.isin(enrichment_highlight["motif"]) * 0.6)
points = ax.scatter(
    enrichment_oi["logFC"],
    enrichment_oi["log_odds"],
    c=enrichment_oi["color"],
    s=30,
    clip_on=False,
    zorder=10,
    alpha=alpha,
    lw=0,
)
texts = []
for i, row in enrichment_highlight.iterrows():
    x, y = row[["logFC", "log_odds"]]
    text = ax.annotate(
        row["symbol"],
        (x, y),
        fontsize=12,
        color=row["color"],
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color=row["color"]),
        bbox=dict(pad=-5, facecolor="none", edgecolor="none"),
        zorder=20,
    )

    text.set_path_effects([mpl.patheffects.withStroke(linewidth=2, foreground="white")])
    texts.append(text)

# axis in middle
ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)
xmax = max(abs(enrichment_oi["logFC"].min()), enrichment_oi["logFC"].max())
ymax = max(abs(enrichment_oi["log_odds"].min()), enrichment_oi["log_odds"].max())

# spines
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ticks
xticks = np.log2([1 / 16, 1 / 4, 1, 4, 16])
yticks = np.log2([1 / 2, 1, 2])
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(["1/16", "1/4", "1", "4", "16"])
ax.set_yticklabels(["1/2", "1", "2"])
ax.set_xlabel("TF mRNA fold-change")
ax.set_ylabel("TFBS\nodds-ratio", rotation=0, ha="right", va="center")

import adjustText

fig.plot()
adjustText.adjust_text(texts, ax=ax, ensure_inside_axes=False)

fig.display()

# %%
fig.plot()
fig.savefig(pp.paths.results() / f"enrichment.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## Plot regions

# %%
motifs_oi = pd.concat([
    pd.DataFrame({
        "motif":motifs_highlight["members"][motifs_highlight.group.str.contains("IRF")].values,
        # "motif":motifs_highlight["members"][motifs_highlight.group.str.contains("IRF4")].values,
        "group":"IRF4",
        "color":"#FF4136",
    }),
    pd.DataFrame({
        "motif":motifs_highlight["members"][motifs_highlight.group.str.contains("BATF")].values,
        "group":"BATF3",
        "color":"#FF851B",
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["STAT6"])],
        "group":"STAT6",
        "color":"#FFDC00",
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["SPI1"])],
        "group":"SPI1",
        "color":"green",
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["EGR2"])],
        "group":"EGR2",
        "color":"purple",
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["PPARG"])],
        "group":"PPARG",
        "color":"#0074D9",
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["CEBPA", "CEBPD"])],
        "group":"CEBPA",
        "color":cold_colors[0],
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["NFIL3"])],
        "group":"NFIL3",
        "color":cold_colors[0],
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["ZEB1", "ZEB2"])],
        "group":"ZEB1/2",
        "color":cold_colors[1],
    }),
    pd.DataFrame({
        "motif":motifscan.motifs.index[motifscan.motifs.tf.isin(["NFAC1"])],
        "group":"NFTAC1",
        "color":cold_colors[3],
    }),
]).set_index("motif")
motif_groups = pd.DataFrame({"group": motifs_oi["group"].unique()}).set_index("group")
motif_groups["label"] = motif_groups.index

# motif_groups["color"] = mpl.colormaps["Set1"].colors[: len(motif_groups)]

# motifs_oi = motifs_highlight.loc["IL33-AM"].copy().set_index("members")
# motif_groups = groups_highlight.loc["IL33-AM"].copy()

# %%
fs4_folder = pathlib.Path("/home/wouters/fs4").resolve()
folder = (
    fs4_folder
    / "u_mgu/private/JeanFrancois/Epigenetic_data/Stijn_CUT_and_RUN_IRF4_AM_IL33/Processed_data/Bedgraph"
)
design = pd.DataFrame(
    {
        "sample": [
            "Clean_AM_IRF4_IL33_rep1",
            "Clean_AM_IRF4_IL33_rep2",
            "Clean_AM_IRF4_PBS_rep1",
            "Clean_AM_IRF4_PBS_rep2",
        ],
        "condition": ["IL33", "IL33", "PBS", "PBS"],
        "replicate": [1, 2, 1, 2],
    }
)

# %%
# convert to bigwig
# although pyBedGraph exists, it doesn't have the same functionality as pyBigWig, so we convert to bw
import os

genome_folder = pathlib.Path("/srv/data/genomes/GRCm39")
for sample in design["sample"]:
    bedgraph = folder / f"{sample}.bedgraph"
    if not bedgraph.exists():
        raise FileNotFoundError(bedgraph)
    if not pathlib.Path(f"{sample}.bw").exists():
        os.system(
            f"bedGraphToBigWig {bedgraph} {genome_folder / 'GRCm39.fa.sizes'} ./{sample}.bw"
        )

# %%
# peakcallers
bed_folder = pathlib.Path(
    "/home/wouters/fs4/u_mgu/private/JeanFrancois/Epigenetic_data/temp_For_Stijn_UBLA/Cleaned/"
)
peakcallers = pd.DataFrame(
        [
            [
                "Il33-1",
                bed_folder / "IL33_AM1_NarrowP.bed",
            ],
            [
                "Il33-2",
                bed_folder / "IL33_AM2_NarrowP.bed",
            ],
            [
                "Il33-3",
                bed_folder / "IL33_AM3_NarrowP.bed",
            ],
            [
                "Ctrl-1",
                bed_folder / "Ctrl_AM1_NarrowP.bed",
            ],
            [
                "Ctrl-2",
                bed_folder / "Ctrl_AM2_NarrowP.bed",
            ],
            [
                "Ctrl-3",
                bed_folder / "Ctrl_AM3_NarrowP.bed",
            ],
        ],
        columns=["label", "path"],
    )
peakcallers["label"] = peakcallers["label"].str.split("-").str[0]

# %%
def make_plot(symbol, only_differential=True, show_peaks = False, show_cutrun = False, show_pparg = False, show_stat6 = False, show_zeb = False, show_cebpa = False, show_spi1 = False, show_nfatc1 = False, show_egr2 = False, show_nfil3 = False):
    if symbol not in regions.var["symbol"].tolist():
        raise ValueError(f"Symbol {symbol} not found in regions")
    gene_id = regions.var.query("symbol == @symbol").index[0]

    windows = regionpositional.select_windows(
        gene_id,
        prob_cutoff=5.5,
        differential_prob_cutoff=3.0 if only_differential else 0.0001,
        keep_tss=True,
    )
    breaking = pp.grid.Breaking(windows, resolution=4000, gap=0.03)

    fig = pp.grid.Figure(pp.grid.Grid(padding_height=0.05, padding_width=0.05))

    region = regions.coordinates.loc[gene_id]

    panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
        region,
        breaking=breaking,
        genome="GRCm39",
        xticks=[-100000, 0, 100000],
        gene_overlap_padding=10000 if windows["length"].sum() > 20000 else 100000,

        show_others=False,
    )
    fig.main.add_under(panel_genes, padding=0.0)

    panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
        region.name,
        regionpositional,
        cluster_info=clustering.cluster_info.loc[["IL33-AM"]],
        breaking=breaking,
        relative_to="Ctrl-AM",
        ylintresh=500,
        ymax=8000, # <--------------------------
        label_accessibility=False,
        norm_atac_diff=mpl.colors.Normalize(np.log(1 / 8), np.log(8.0), clip=True),
        label_cluster=False,
    )
    panel_differential[0][0, 0].set_ylabel(
        "Accessibility\nIL33 vs PBS", rotation=0, ha="right", va="center"
    )
    # panel_differential.add_differential_slices(differential_slices)
    fig.main.add_under(panel_differential)

    motifs_oi2 = motifs_oi.copy()
    motif_groups2 = motif_groups.copy()
    if not show_pparg:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["PPARG"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["PPARG"])]

    if not show_stat6:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["STAT6"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["STAT6"])]

    if not show_cebpa:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["CEBPA"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["CEBPA"])]

    if not show_zeb:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["ZEB1/2"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["ZEB1/2"])]

    if not show_spi1:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["SPI1"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["SPI1"])]

    if not show_nfatc1:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["NFATC1"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["NFATC1"])]

    if not show_egr2:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["EGR2"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["EGR2"])]

    if not show_nfil3:
        motifs_oi2 = motifs_oi2.loc[~motifs_oi2["group"].isin(["NFIL3"])]
        motif_groups2 = motif_groups2.loc[~motif_groups2.index.isin(["NFIL3"])]

    panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
        motifscan,
        gene_id,
        motifs_oi2,
        breaking,
        group_info=motif_groups2,
        slices_oi=differential_slices.get_slice_scores(
            clustering=clustering, regions=regions
        ),
    )
    fig.main.add_under(panel_motifs)

    # panel_associations = chd.data.associations.plot.AssociationsBroken(associations, region.name, breaking, show_ld = True)
    # fig.main.add_under(panel_associations)

    if show_cutrun:
        panel_bw = chd.plot.tracks.TracksBroken.from_bigwigs(
            design.iloc[[0, 1]].copy(), region, breaking, height=0.3
        )
        fig.main.add_under(panel_bw, padding = 0.)
        panel_bw.elements[0][0].set_ylabel("Cut&Run\nIRF4 IL33", rotation=0, ha="right", va="center")
        panel_bw = chd.plot.tracks.TracksBroken.from_bigwigs(
            design.iloc[[2, 3]].copy(), region, breaking, height=0.3
        )
        fig.main.add_under(panel_bw)
        panel_bw.elements[0][0].set_ylabel("IRF4 PBS", rotation=0, ha="right", va="center")

    if show_peaks:
        import chromatinhd.data.peakcounts
        panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed(
            regions.coordinates.loc[gene_id],
            peakcallers = peakcallers.iloc[[0, 3]],
            breaking=breaking,
            label_methods_side="left",
            row_height=0.4,
            label_rows="Peaks",
        )
        fig.main.add_under(panel_peaks)

    ### Add legend
    # legend_grid = pp.grid.Grid()
    # fig.main = pp.grid.Grid([[fig.main, legend_grid]], ncol = 2, nrow = 1)
    
    # cbar_ax = pp.grid.Panel((0.15, 0.8))

    # cmap = chd.models.diff.plot.differential.cmap_atac_diff
    # norm=mpl.colors.Normalize(np.log(1 / 8), np.log(8.0), clip=True)
    # cba = mpl.colorbar.ColorbarBase(cbar_ax, cmap = cmap, norm=norm, extend="both")
    # legend_grid.add_right(cbar_ax)

    return fig

# %%
import pathlib
plot_folder = pathlib.Path("/home/wouters/fs4/u_mgu/private/wouters/grants/2024_ERC_StG/figs")
# symbol = "Chil4"
# symbol = "Irf4"
# symbol = "Ccl24"
# symbol = "Dcstamp"
# symbol = "Ocstamp"
# symbol = "Kcnn4"
# symbol = "Tgfb2"
# symbol = "Fpr1"
symbol = "Pparg"
# symbol = "Hmgn2"
# symbol = "Clcn3"
# symbol = "Mccc1"
fig = make_plot(symbol, only_differential = True, show_peaks = True, show_stat6=True, show_cutrun=True, show_nfatc1=True, show_nfil3 = False)

fig.display()
# fig.savefig(plot_folder / f"{symbol}.pdf", dpi = 600, bbox_inches = "tight")

# %%
fig = chd.models.diff.plot.differential.create_colorbar_horizontal()

# %%
plot_parent_folder = pathlib.Path(
    "/home/wouters/fs4/u_mgu/transfer/AMEpigenetics/IL33vsPBS"
)
plot_folder = plot_parent_folder / "differential_Irf4Batf3"
plot_folder.mkdir(parents=True, exist_ok=True)

plot_folder_all = plot_parent_folder / "all_Irf4Batf3"
plot_folder_all.mkdir(parents=True, exist_ok=True)

# for symbol in diffexp["gene"]:
for symbol in ["Irf4", "Batf3", "Pparg", "Slc30a4", "Pdcd1lg2"]:
    fig = make_plot(symbol, only_differential=True, show_peaks=False, show_cutrun=False, show_pparg=False, show_stat6=False)
    fig.savefig(plot_folder / f"{symbol}.pdf", dpi=300, bbox_inches="tight")

    fig = make_plot(symbol, only_differential=False, show_peaks=False, show_cutrun=False, show_pparg=False, show_stat6=False)
    fig.savefig(plot_folder_all / f"{symbol}.pdf", dpi=300, bbox_inches="tight")

# %%
plot_parent_folder = pathlib.Path(
    "/home/wouters/fs4/u_mgu/transfer/AMEpigenetics/IL33vsPBS"
)
plot_folder = plot_parent_folder / "differential_Irf4Batf3Pparg_cutrun"
plot_folder.mkdir(parents=True, exist_ok=True)

plot_folder_all = plot_parent_folder / "all_Irf4Batf3Pparg_cutrun"
plot_folder_all.mkdir(parents=True, exist_ok=True)

# for symbol in ["Irf4", "Batf3", "Pparg", "Slc30a4", "Pdcd1lg2"]:
for symbol in [
        "Marco", "Perp", "Fpr1", "Vstm2a", "Lipf", "Afap1", "Htr2c", "Atp10a",
        "Mmp12", "Mmp14", "Batf3", "Slc30a4", "Pdcd1lg2", "Rnase2a", "Pparg",
    ]:
    print(symbol)
    fig = make_plot(symbol, only_differential=True, show_peaks=False, show_cutrun=True, show_pparg=True, show_stat6=False)
    fig.savefig(plot_folder / f"{symbol}.pdf", dpi=300, bbox_inches="tight")

    fig = make_plot(symbol, only_differential=False, show_peaks=False, show_cutrun=True, show_pparg=True, show_stat6=False)
    fig.savefig(plot_folder_all / f"{symbol}.pdf", dpi=300, bbox_inches="tight")

# %%
plot_parent_folder = pathlib.Path(
    "/home/wouters/fs4/u_mgu/transfer/AMEpigenetics/IL33vsPBS"
)
plot_folder = plot_parent_folder / "differential_Irf4Batf3Stat6_cutrun"
plot_folder.mkdir(parents=True, exist_ok=True)

plot_folder_all = plot_parent_folder / "all_Irf4Batf3Stat6_cutrun"
plot_folder_all.mkdir(parents=True, exist_ok=True)

# for symbol in ["Irf4", "Batf3", "Pparg", "Slc30a4", "Pdcd1lg2"]:
for symbol in [
        "Ccl17", "Ccl24", "Irf4", "Ocstamp", "Dcstamp", "Mapk14", "Cd81",
    ]:
    print(symbol)
    fig = make_plot(symbol, only_differential=True, show_peaks=False, show_cutrun=True, show_pparg=False, show_stat6=True)
    fig.savefig(plot_folder / f"{symbol}.pdf", dpi=300, bbox_inches="tight")

    fig = make_plot(symbol, only_differential=False, show_peaks=False, show_cutrun=True, show_pparg=False, show_stat6=True)
    fig.savefig(plot_folder_all / f"{symbol}.pdf", dpi=300, bbox_inches="tight")

# %%
!cp -r /home/wouters/fs4/u_mgu/transfer/AMEpigenetics /home/wouters/fs4/u_bla/transfer/

# %%
# %% [markdown]
# ## Enrichment of genes

# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
slicecounts = motifscan.count_slices(slicescores)
slicecounts_gene = slicecounts.copy().loc[slicescores.index]
slicecounts_gene.index = pd.MultiIndex.from_frame(slicescores[["cluster", "region"]])
slicecounts_gene = slicecounts_gene.groupby(level=[0, 1], observed=False).sum()
slicecounts_gene = slicecounts_gene.reindex(
    chd.utils.crossing(clustering.cluster_info.index, fragments.regions.var.index),
    fill_value=0,
)

totalcounts_gene = slicescores.groupby(["cluster", "region"], observed=False)[
    "length"
].sum()
totalcounts_gene = totalcounts_gene.reindex(
    chd.utils.crossing(clustering.cluster_info.index, fragments.regions.var.index),
    fill_value=0,
)

slicescores_reference = slicescores.query("cluster == 'Ctrl-AM'")
expected_motif = (
    slicecounts.loc[slicescores_reference.index].sum(0)
    / slicescores_reference["length"].sum()
)

# %% [markdown]
# ### Indiviudal motifs

# %%
gene_enrichment = (slicecounts_gene) / (
    totalcounts_gene.values[:, None] * expected_motif.values[None, :]
)

gene_id = fragments.regions.var.query("symbol == 'Dcstamp'").index[0]
gene_enrichment.loc["IL33-AM"].loc[gene_id].loc[
    motifs_oi.index
].dropna().sort_values(ascending=False)

# %%
gene_ids = fragments.regions.var.query("symbol in ['Kcnn4', 'Ocstamp', 'Dcstamp', 'Pparg', 'Irf4', 'Ccl17', 'Ccl24']").index
gene_enrichment.loc["IL33-AM"].loc[gene_ids, 
    # slice(None)
    ["NFAC1.H12CORE.0.P.B", "NFAC1.H12CORE.2.SM.B", "NFAC1.H12CORE.3.SM.B"]
    # enrichment_grouped.loc["IL33-AM"].sort_values("odds", ascending = False).index[:10]
    # ["HINFP.H12CORE.2.S.B"]
]

# %% [markdown]
# ### Grouped motifs

# %%
groups_oi = enrichment_grouped.loc["IL33-AM"].sort_values("odds", ascending=False).head(15)
gene_enrichment = pd.concat([(slicecounts_gene[members].sum(1)) / (
    totalcounts_gene.values * expected_motif[members].sum()
) for members in groups_oi["members"]], axis = 1)
gene_enrichment.columns = groups_oi["group"]

gene_enrichment.loc["IL33-AM"]

# %%
symbols = ["Irf4", "Batf3", "Pparg", "Marco", "Lipf", 'Ccl17', 'Ccl24']
gene_ids = fragments.regions.var.reset_index().set_index("symbol").loc[symbols, "gene"]
plotdata = gene_enrichment.loc["IL33-AM"].loc[gene_ids].apply(np.log).T
plotdata.columns = symbols
plotdata.style.bar(vmin = -1., vmax = 1.0, align=0., color = "lightblue")


# %% [markdown]
# ## GWAS enrichment
# %%
genes_oi = diffexp.sort_values("logFC", ascending=False).head(200)["gene"]
diffexp.sort_values("logFC", ascending=False).head(50)

# %%
import chromatinhd.data.associations

associations = chd.data.associations.Associations(
    chd.get_output()
    / "datasets"
    / dataset_name
    / "motifscans"
    / "100k100k"
    / "gwas_asthma"
)
associations.regions.coordinates["n"] = np.bincount(
    associations.region_indices[:], minlength=len(associations.regions.coordinates)
)

associations.regions.coordinates.sort_values("n", ascending=False).query(
    "symbol in @genes_oi"
).head(20)