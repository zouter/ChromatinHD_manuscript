# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
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

import chromatinhd as chd

# %%
dataset_folder = chd.get_output() / "datasets" / "liverphx"

# %%
regions_name = "100k100k"

# %%
fragments = chd.data.fragments.FragmentsView(dataset_folder / "fragments" / regions_name)

# %%
clustering = chd.data.Clustering(dataset_folder / "clusterings" / "cluster")
model_folder = chd.get_output() / "diff" / "liverphx" / "binary" / "split" / regions_name / "cluster"

# %%
dataset_name = "liverphx"
motifscan_name = "hocomocov12_1e-4"
motifscan_name = "jaspar2024_4.5"

# %%
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %% [markdown]
# ## Model

# %%
import chromatinhd.models.diff.model.binary
model = chd.models.diff.model.binary.Model.create(
    fragments,
    clustering,
    fold = fold,
    encoder = "shared",
    # encoder = "split",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale = 0.5,
        bias_regularization=True,
        bias_p_scale = 0.5,
        # binwidths = (5000, 1000)
        # binwidths = (5000, 1000, 500, 100, 50)
        binwidths = (5000, 1000, 500, 100, 50, 25)
    ),
    path = model_folder / "model",
    overwrite = True,
)

# %%
model.train_model(n_epochs = 40, n_regions_step = 50, early_stopping=False, do_validation = True, lr = 1e-2)

# %%
model.trace.plot();

# %%
model.save_state()

# %% [markdown]
# ## Interpret

# %%
genepositional = chd.models.diff.interpret.RegionPositional.create(path = model_folder / "scoring" / "genepositional")
if not len(genepositional.probs) == len(fragments.var):
    genepositional.score(
        fragments = fragments,
        clustering = clustering,
        models = [model],
        # regions = fragments.var.reset_index().set_index("symbol").loc[["Kit", "Odc1", "Dll4", "Dll1", "Jag1", "Meis1", "Efnb2"]]["gene"],
        force = True,
        normalize_per_cell=1,
        device = "cpu",
    )

# %%
import sklearn.decomposition

# %%
prob_cutoff = 1.
# prob_cutoff = 0.

import xarray as xr
probs = xr.concat([scores for _, scores in genepositional.probs.items()], dim = pd.Index(genepositional.probs.keys (), name = "gene"))
probs = probs.load()
lr = probs - probs.mean("cluster")

probs_stacked = probs.stack({"coord-gene":["coord", "gene"]})
probs_stacked = probs_stacked.values[:, (probs_stacked.mean("cluster") > prob_cutoff).values]
probs_stacked = (probs_stacked - probs_stacked.mean(axis = 0)) / probs_stacked.std(axis = 0)
probs_stacked = pd.DataFrame(probs_stacked, index = probs.coords["cluster"])
sns.heatmap(probs_stacked.T.corr())

# %%
out = sklearn.decomposition.PCA(n_components = 3, whiten = True).fit_transform(probs_stacked)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1])
texts = []
for i, gene in enumerate(probs_stacked.index):
    text = ax.annotate(gene, out[i, :2])
    texts.append(text)
import adjustText
adjustText.adjust_text(texts)

# %%
# plotdata = pd.DataFrame({
#     "modelled":np.exp(probs).sum(["gene", "coord"]).to_pandas().sort_values(),
#     # "modelled":probs.sum(["gene", "coord"]).to_pandas().sort_values(),
#     "libsize":pd.Series(model.libsize, index = clustering.labels).sort_values()
# })
# plotdata.sort_values("libsize").style.bar()

# %% [markdown]
# ## Most differential

# %%
probs_mask = (probs > 0.5).any("cluster")
lr_masked = lr.where(probs_mask).fillna(0.)

genes_oi = (lr_masked.mean("coord") **2).mean("cluster").to_pandas().sort_values(ascending = False).head(80).index

plotdata = lr_masked.sel(gene = genes_oi).mean("coord").to_pandas()
# plotdata = plotdata.loc[fragments.var.index]
plotdata.index = fragments.var.loc[plotdata.index,"symbol"]

fig, ax = plt.subplots(figsize = (3, len(plotdata) * 0.2))
sns.heatmap(plotdata, vmax = 0.2, vmin = -0.2, cmap = "RdBu_r", center = 0, cbar_kws = dict(label = "log likelihood ratio"))

# %% [markdown]
# ## Visualize single

# %%
# symbol = "Apln"
# symbol = "Pdgfb"
# symbol = "Thbd"
# symbol = "Kit"
# symbol = "Icam1"
# symbol = "Rspo3"
# symbol = "Dll1"
symbol = "Dll4"
symbol = "Cd36"
symbol = "Pdgfb"
# symbol = "Mecom"
# symbol = "Fabp4"
# symbol = "Odc1"
# symbol = "Wnt9b"
# symbol = "Wnt2"
# symbol = "Cdh13"
# symbol = "Ltbp4"
# symbol = "Ccn4"
# symbol = "Adam15"
# symbol = "Adamts9"

gene_id = fragments.var.index[fragments.var["symbol"] == symbol][0]
gene_ix = fragments.var.index.get_loc(gene_id)

# %%
genepositional.clustering = clustering
genepositional.regions = fragments.regions

# %%
# motifs
motifs_oi = pd.DataFrame([
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("SUH")][0], "Rbpj", "Dll → Rbpj/Hey1/Hes1", mpl.cm.Blues(0.7)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("HEY1")][0], "Hey1", "Dll → Rbpj/Hey1/Hes1", mpl.cm.Blues(0.5)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("HES1")][0], "Hes1", "Dll → Rbpj/Hey1/Hes1", mpl.cm.Blues(0.6)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("TCF7")][0], "Tcf7", "Wnt → Tcf7/Lef1/Sox7/Sox18", mpl.cm.Greens(0.6)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("LEF1")][0], "Lef1", "Wnt → Tcf7/Lef1/Sox7/Sox18", mpl.cm.Greens(0.5)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("SOX7")][0], "Sox7", "Wnt → Tcf7/Lef1/Sox7/Sox18", mpl.cm.Greens(0.7)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("SOX18")][0], "Sox18", "Wnt → Tcf7/Lef1/Sox7/Sox18", mpl.cm.Greens(0.9)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("FOS")][0], "Fos", "Fos/Jun", mpl.cm.Purples(0.6)],
    [motifscan.motifs.index[motifscan.motifs.index.str.contains("JUND")][0], "Jund", "Fos/Jun", mpl.cm.Purples(0.7)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("BACH2")][0], "Bach2", "Fos/Jun", mpl.cm.Purples(0.7)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("FOXP2")][0], "Foxp2", "Foxp", mpl.cm.Reds(0.5)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("SOX9")][0], "Sox9", "Sox", mpl.cm.Reds(0.5)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("GATA4")][0], "Gata4", "Gata", mpl.cm.Reds(0.5)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("MEF2A")][0], "Mef2a", "Mef", mpl.cm.Reds(0.5)],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("IRF5")][0], "Irf5", "IRF", "pink"],
    # [motifscan.motifs.index[motifscan.motifs.index.str.contains("IRF9")][0], "Irf9", "IRF", "pink"],
], columns = ["motif", "label", "group", "color"]
).set_index("motif")

# %%
motifs_oi = pd.DataFrame([
    [motifscan.motifs.query("human_gene_name == 'RBPJ'").index[0], "Rbpj", "Dll → Rbpj/Hey1/Hes1", mpl.cm.Blues(0.7)],
], columns = ["motif", "label", "group", "color"]).set_index("motif")

# %%
dataset_folder2 = chd.get_output() / "datasets" / "liverphx_48h"

# %%
transcriptome = chd.data.transcriptome.Transcriptome(dataset_folder2 / "transcriptome")
clustering2 = chd.data.clustering.Clustering(dataset_folder2 / "clusterings" / "portal_central")

# %%
# cluster_info = clustering.var.loc[["lsec-portal-sham", "lsec-portal-24h", "lsec-central-sham", "lsec-central-24h"]]
# cluster_info2 = clustering2.var.loc[["lsec-portal-sham", "lsec-portal-48h", "lsec-central-sham", "lsec-central-48h"]]

cluster_info = clustering.var.loc[["lsec-portal-sham", "lsec-central-24h"]]
cluster_info2 = clustering2.var.loc[["lsec-portal-sham", "lsec-central-48h"]]

# %%
# !cat {chd.get_output() / "data" / "liverphx" / "LsecCvPh24h.fwp.filter.non_overlapping.bed"} {chd.get_output() / "data" / "liverphx" / "LsecPvPh24h.fwp.filter.non_overlapping.bed"} | sort -k1,1 -k2,2n | mergeBed -i stdin > {chd.get_output() / "data" / "liverphx" / "merged.bed"}

# %%
# peakcallers = pd.DataFrame(
#     {"path":[
#         chd.get_output() / "data" / "liverphx" / "LsecCvPh24h.fwp.filter.non_overlapping.bed",
#         chd.get_output() / "data" / "liverphx" / "LsecPvPh24h.fwp.filter.non_overlapping.bed"
#     ], "label":[
#         "LSEC CV 24h",
#         "LSEC PV 24h",
#     ]}
# )
peakcallers = pd.DataFrame(
    {"path":[
        chd.get_output() / "data" / "liverphx" / "merged.bed",
    ], "label":[
        ""
    ]}
)

# %%
import chromatinhd.data.peakcounts

# %%
motifscan.parent.create_region_indptr(overwrite = True)

# %%
motifscan.parent.create_indptr(overwrite = True)

# %%
symbol = fragments.var.loc[gene_id, "symbol"]

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

import dataclasses
@dataclasses.dataclass
class Breaking():
    regions: pd.DataFrame
    gap: int
    resolution: int = 2000


regions = genepositional.select_regions(gene_id, prob_cutoff = 0.5)
# regions = regions.iloc[:2]
breaking = Breaking(regions, 0.05)

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking,
    genome="mm10",
    label_positions=True,
)
fig.main.add_under(panel_genes)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    gene_id, genepositional, breaking, panel_height=0.4, cluster_info=cluster_info, label_accessibility=False, relative_to = "lsec-portal-sham"
)
fig.main.add_under(panel_differential)

if gene_id in transcriptome.var.index:
    panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
        transcriptome,
        clustering2,
        gene_id,
        cluster_info=cluster_info2,
        panel_height=0.4,
        show_n_cells=False,
    )
    fig.main.add_right(panel_expression, panel_differential)

panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(motifscan, gene_id, motifs_oi, breaking)
fig.main.add_under(panel_motifs)

panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed( 
    region, peakcallers, breaking
)
fig.main.add_under(panel_peaks)

fig.plot()

# %% [markdown]
# ## Visualize all

# %%
symbols_oi = """
Atf3
Dll1
Mecom
Dll4
Meis1
Odc1
Cdkn1a
Thbd
Fabp4
Armcx4
Apln
Jag1
Meis1
Kcne3
Angpt2
Hey1
Heyl
Hes1
Sox18
Sox17
Sox7
Adam15
Efnb1
Efnb2
Efna1
Foxo1
Ltbp4
Lama4
Esm1
Stc1
Lrp4
Wnt2
Wnt9b
Rspo3
Kit
Thbd
Ccnd2
Ccn4

Dll1
Thbd
Kit
Slco2a1
Wnt2
Dab2
Ackr3
Gja4
Ptgs1
Actg1
Plvap
Slc26a10
Tcim
Tlnrd1
Bcam
Lhx6
Adam15
Arl4d
Jpt1
Jam2
Myadm
Cd9
Slc9a3r2
Slc38a2
Wnt9b
Klf4
C1qb
Sema4c
Rab3b
Mcfd2
Car8
Cbfa2t3
Ccnd2
Lgals1
Klf10
Adamts1
Ralgapa2
Klf2
Ier5
Tuba1a
""".strip().split("\n")
gene_ids = fragments.var.index[fragments.var["symbol"].isin(symbols_oi)]
gene_ids

# %%
plot_folder = chd.get_output() / "liverphx" / "diff" / "examples"
plot_folder.mkdir(parents = True, exist_ok = True)

# %%
# !rm {plot_folder / "*"}

# %%
pd.DataFrame({"gene":gene_ids, "symbol":fragments.var.loc[gene_ids, "symbol"]}).set_index("gene").to_csv(plot_folder / "genes.csv")

# %%
for gene_id in tqdm.tqdm(gene_ids):
    symbol = fragments.var.loc[gene_id, "symbol"]
    if (plot_folder / f"{symbol}.pdf").exists():
        continue

    fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05, padding_width=0.05))

    import dataclasses
    @dataclasses.dataclass
    class Breaking():
        regions: pd.DataFrame
        gap: int
        resolution: int = 2000


    regions = genepositional.select_regions(gene_id, prob_cutoff = 0.5)
    breaking = Breaking(regions, 0.05)

    region = fragments.regions.coordinates.loc[gene_id]
    panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
        region,
        breaking,
        genome="mm10",
        label_positions=True,
    )
    fig.main.add_under(panel_genes)

    panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
        gene_id, genepositional, breaking, panel_height=0.4, cluster_info=cluster_info, label_accessibility=False, relative_to = "lsec-portal-sham"
    )
    fig.main.add_under(panel_differential)

    if gene_id in transcriptome.var.index:
        panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
            transcriptome,
            clustering2,
            gene_id,
            cluster_info=cluster_info2,
            panel_height=0.45,
            show_n_cells=False,
        )
        fig.main.add_right(panel_expression, panel_differential)

    panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(motifscan, gene_id, motifs_oi, breaking)
    fig.main.add_under(panel_motifs)

    panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed(
        region, peakcallers, breaking
    )
    fig.main.add_under(panel_peaks)

    fig.plot()
    
    # save
    fig.savefig(plot_folder / f"{symbol}.pdf", bbox_inches = "tight")
    plt.close()

# %%
fig.plot()
fig

# %%
# !echo "rsync -avz -e 'ssh -p 22345' {plot_folder} wouters@cp0001.irc.ugent.be:/dmbr/t/u_mgu/private/WouterS/transfer/chromatinhd_liverphx/results/landscapes"

# %% [markdown]
# ## Enrichment

# %%
# cluster_info_oi = clustering.var
# cluster_info_oi = clustering.var.query("celltype == 'lsec'").query("zonation == 'portal'")
# cluster_info_oi = clustering.var.query("celltype == 'lsec'").query("treatment == 'sham'")
cluster_info_oi = clustering.var.loc[["lsec-portal-sham", "lsec-central-24h"]]

# %%
genepositional.fragments = fragments
genepositional.regions = fragments.regions
genepositional.clustering = clustering

# slices = genepositional.calculate_slices(1., clusters_oi = cluster_info_oi.index.tolist(), step = 25)
slices = genepositional.calculate_slices(0., clusters_oi = cluster_info_oi.index.tolist(), step = 5)
differential_slices = genepositional.calculate_differential_slices(slices, 1.5)


# %%
def symbol_to_gene(symbols):
    return fragments.var.index[fragments.var["symbol"].isin(symbols)].tolist()


# %%
slicescores = differential_slices.get_slice_scores(regions = fragments.regions, cluster_info = cluster_info_oi)

slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

# %%
# motifscan_name = "hocomocov12_1e-4"
# motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
# motifscan.motifs["label"] = motifscan.motifs["MOUSE_gene_symbol"]

motifscan_name = "jaspar2024_4.5"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
motifscan.motifs["label"] = motifscan.motifs["human_gene_name"]

clustering.var["n_cells"] = clustering.labels.value_counts()

# %%
slicecounts = motifscan.count_slices(slices)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores, slicecounts)

# %%
enrichment["log_odds"] = np.log2(enrichment["odds"])
# enrichment = enrichment.loc[enrichment.index.get_level_values("motif").isin(motifscan.motifs.index[motifscan.motifs["quality"].isin(["A", "B"])])]
# enrichment["significant"] = True
enrichment["significant"] = (enrichment["q_value"] < 0.05)# & (enrichment["odds"] > 1.5)

# %%
enrichment_differential = enrichment

# %%
enrichment_oi = enrichment.loc[enrichment.groupby("motif")["significant"].any()[enrichment.index.get_level_values("motif")].values]
sns.heatmap(enrichment_oi["log_odds"].unstack().T.corr(), vmax = 0.5, vmin = -0.5, cmap = "RdBu_r", center = 0)

# %%
enrichment.loc["lsec-central-24h"].sort_values("log_odds", ascending = False).join(motifscan.motifs).head(30)

# %% [markdown]
# ### Determine similarity

# %%
indices_oi = motifscan.parent.indices[:2000000]
coordinates_oi = motifscan.parent.coordinates[:2000000]

# %%
coordinate_sets = []
for motif_ix in range(motifscan.motifs.shape[0]):
    coordinate_sets.append(set(coordinates_oi[indices_oi == motif_ix]//20))

# %%
jaccards = {}
for motif_ix1 in tqdm.tqdm(range(motifscan.motifs.shape[0])):
    for motif_ix2 in range(motifscan.motifs.shape[0]):
        overlap = len(coordinate_sets[motif_ix1].intersection(coordinate_sets[motif_ix2]))
        union = len(coordinate_sets[motif_ix1].union(coordinate_sets[motif_ix2]))

        if union > 0:
            jaccards[(motif_ix1, motif_ix2)] = overlap / union
        else:
            jaccards[(motif_ix1, motif_ix2)] = 1.

# %%
motif_ix1 = motifscan.motifs.index.get_loc("FOS.H12CORE.0.P.B")
motif_ix2 = motifscan.motifs.index.get_loc("FOSB.H12CORE.0.P.B")
# motif_ix2 = motifscan.motifs.index.get_loc("FOSL1.H12CORE.0.P.B")
# motif_ix2 = motifscan.motifs.index.get_loc("ATF3.H12CORE.0.P.B")
# motif_ix2 = motifscan.motifs.index.get_loc(motifscan.select_motif("TAL1"))

# %%
overlap = len(coordinate_sets[motif_ix1].intersection(coordinate_sets[motif_ix2]))
union = len(coordinate_sets[motif_ix1].union(coordinate_sets[motif_ix2]))

overlap / union

# %%
jaccards[(motif_ix1, motif_ix2)]

# %%
jaccards2 = pd.Series(jaccards.values(), index = pd.MultiIndex.from_tuples(jaccards.keys()))
jaccards2 = jaccards2.unstack()
jaccards2.index = motifscan.motifs.index
jaccards2.columns = motifscan.motifs.index

# %%
enrichment_oi = enrichment.loc["lsec-portal-sham"].query("significant").query("odds > 1.2").sort_values("odds", ascending = False)
sns.heatmap(jaccards2.loc[enrichment_oi.index, enrichment_oi.index] > 0.05)

# %%
enrichment_condensed = []
# enrichment_raw = enrichment.query("q_value < 0.05").query("odds > 1").sort_values("q_value", ascending = False).copy()
enrichment_raw = enrichment.loc["lsec-portal-sham"].query("q_value < 0.05").query("odds > 1").sort_values("odds", ascending = False).copy()
# enrichment_raw = enrichment.loc["lsec-central-24h"].query("q_value < 0.05").query("odds > 1").sort_values("odds", ascending = False).copy()
enrichment_raw = enrichment_raw.loc[(~enrichment_raw.index.str.startswith("ZFP")) & (~enrichment_raw.index.str.startswith("ZN")) & (~enrichment_raw.index.str.startswith("ZIC"))]
enrichment_raw = enrichment_raw.loc[motifscan.motifs.loc[enrichment_raw.index]["quality"].isin(["B", "A", "C"])]
while len(enrichment_raw) > 0:
    row = enrichment_raw.iloc[0].copy()
    motif = row.name
    to_remove = jaccards2.columns[jaccards2.loc[motif] > 0.05]
    to_remove = [motif for motif in enrichment_raw.index if motif in to_remove]

    symbols = motifscan.motifs.loc[to_remove, "symbol"].tolist()
    symbols = list(set(symbols))
    order = np.argsort(-transcriptome.var.set_index("symbol").reindex(symbols)["mean"].fillna(-1))
    row["symbols"] = [symbols[i] for i in order]

    if len(to_remove) == 0:
        raise ValueError("No more motifs to remove")

    enrichment_condensed.append(row)

    enrichment_raw = enrichment_raw.loc[~enrichment_raw.index.isin(to_remove)]
enrichment_condensed = pd.DataFrame(enrichment_condensed)

# %%
enrichment.loc["lsec-portal-sham"].loc[motifscan.select_motif("SUH")], enrichment.loc["lsec-portal-sham"].loc[motifscan.select_motif("HES")]

# %%
enrichment_condensed.head(10)

# %% [markdown]
# ## Enrichment top slices

# %%
slices_selected = genepositional.calculate_slices(0., clusters_oi = cluster_info_oi.index.tolist(), step = 5)

# %%
top_slices = genepositional.calculate_top_slices(slices_selected, 1.5)

# %%
slicescores = top_slices.get_slice_scores(regions = fragments.regions)

slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

# %%
# motifscan_name = "hocomocov12_1e-4"
# motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
# motifscan.motifs["label"] = motifscan.motifs["MOUSE_gene_symbol"]

motifscan_name = "jaspar2024_4.5"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
motifscan.motifs["label"] = motifscan.motifs["human_gene_name"]

# %%
slicecounts = motifscan.count_slices(slices)

# %%
slices_background = slices.copy()
np.random.seed(0)
slices_background["region_ix"] = np.random.choice(slices_background["region_ix"], size = slices_background.shape[0], replace = True)

slices_background = pd.DataFrame({
    "region_ix":np.arange(len(fragments.var.index)),
    "start":fragments.regions.window[0],
    "end":fragments.regions.window[1],
})

slices_background.index = pd.Index("background:" + slices_background.index.astype(str), name = "slice")
slicecounts_background = motifscan.count_slices(slices_background)

# %%
slicecounts_all = pd.concat([slicecounts, slicecounts_background])
slices["length"] = slices["end"] - slices["start"]
slices_background["length"] = slices_background["end"] - slices_background["start"]

# %%
enrichment = chd.models.diff.interpret.enrichment.enrichment_foreground_vs_background(slices.reset_index(), slices_background.reset_index(), slicecounts_all)
enrichment["log_odds"] = np.log2(enrichment["odds"])

# %%
# enrichment.loc[motifscan.select_motif("SUH")], enrichment.loc[motifscan.select_motif("HES1")]
enrichment.loc[motifscan.motifs.query("human_gene_name == 'RBPJ'").index[0]], enrichment.loc[motifscan.motifs.query("human_gene_name == 'HEY1'").index[0]]

# %%
motifscan.motifs.query("mouse_gene_name == 'Mecom'")

# %%
enrichment.sort_values("odds", ascending = False).join(motifscan.motifs).head(20)

# %%
jaccards2.loc["GABPA.H12CORE.0.PSM.A", "SPDEF.H12CORE.1.SM.B"] = 1.
jaccards2.loc["GABPA.H12CORE.0.PSM.A", "SPDEF.H12CORE.0.PSM.A"] = 1.
jaccards2.loc["ETV7.H12CORE.0.SM.B", "ELK4.H12CORE.0.PSM.A"] = 1.
jaccards2.loc["ETV7.H12CORE.0.SM.B", "ETV5.H12CORE.0.PS.A"] = 1.
jaccards2.loc["ETV7.H12CORE.0.SM.B", "E4F1.H12CORE.0.P.B"] = 1.
jaccards2.loc["ETV7.H12CORE.0.SM.B", "ELF5.H12CORE.0.PSM.A"] = 1.
jaccards2.loc["NFIX.H12CORE.0.SM.B", "ZFX.H12CORE.0.P.B"] = 1.
jaccards2.loc["MLX.H12CORE.0.PM.A", "HEN1.H12CORE.0.S.B"] = 1.
jaccards2.loc["E2F4.H12CORE.1.P.B", "E2F2.H12CORE.0.S.B"] = 1.
jaccards2.loc["E2F4.H12CORE.1.P.B", "E2F4.H12CORE.2.S.B"] = 1.
jaccards2.loc["NFE2.H12CORE.1.SM.B", "ATF1.H12CORE.1.P.B"] = 1.
jaccards2.loc["NFE2.H12CORE.1.SM.B", "JUNB.H12CORE.1.S.C"] = 1.
jaccards2.loc["SMAD5.H12CORE.0.M.C", "SMAD4.H12CORE.0.P.B"] = 1.
jaccards2.loc[jaccards2.index.str.contains("E2F"), jaccards2.columns.str.contains("E2F")] = 1.

# %%
enrichment_condensed = []
# enrichment_raw = enrichment.query("q_value < 0.05").query("odds > 1").sort_values("q_value", ascending = False).copy()
enrichment_raw = enrichment.query("q_value < 0.05").query("odds > 1.2").sort_values("odds", ascending = False).copy()
enrichment_raw = enrichment_raw.loc[(~enrichment_raw.index.str.startswith("ZF")) & (~enrichment_raw.index.str.startswith("ZFP")) & (~enrichment_raw.index.str.startswith("ZN")) & (~enrichment_raw.index.str.startswith("ZIC")) & (~enrichment_raw.index.str.startswith("ZSC")) & (~enrichment_raw.index.str.startswith("KMT")) & (~enrichment_raw.index.str.startswith("GCM"))]
enrichment_raw = enrichment_raw.loc[motifscan.motifs.loc[enrichment_raw.index]["quality"].isin(["B", "A", "C"])]
while len(enrichment_raw) > 0:
    row = enrichment_raw.iloc[0].copy()
    motif = row.name
    to_remove = jaccards2.columns[jaccards2.loc[motif] > 0.05]
    to_remove = [motif for motif in enrichment_raw.index if motif in to_remove]

    symbols = motifscan.motifs.loc[to_remove, "symbol"].tolist()
    symbols = list(set(symbols))
    order = np.argsort(-transcriptome.var.set_index("symbol").reindex(symbols)["mean"].fillna(-1))
    row["symbols"] = [symbols[i] for i in order]

    if len(to_remove) == 0:
        raise ValueError("No more motifs to remove")

    enrichment_condensed.append(row)

    enrichment_raw = enrichment_raw.loc[~enrichment_raw.index.isin(to_remove)]
enrichment_condensed = pd.DataFrame(enrichment_condensed)

# %%
enrichment_condensed["label"] = [", ".join([str(x) for x in symbols[:5]]) for symbols in enrichment_condensed["symbols"]]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid())

n = 30

resolution = 0.25

plotdata = enrichment_condensed["log_odds"].values[:n, None]
panel, ax = fig.main.add_right(polyptich.grid.Panel((np.array(plotdata.shape)[::-1] * resolution)))
ax.matshow(plotdata, vmin = 0)
ax.set_xticks([])
ax.set_yticks(np.arange(plotdata.shape[0]))
ax.set_yticklabels(enrichment_condensed["label"].iloc[:len(plotdata)], fontsize = 12)

plotdata = enrichment_differential.loc["lsec-central-24h"].loc[enrichment_condensed.index]["log_odds"].values[:n, None]
panel, ax = fig.main.add_right(polyptich.grid.Panel((np.array(plotdata.shape)[::-1] * resolution)), padding = 0.05)

norm = mpl.colors.Normalize(vmin = -1, vmax = 1)
ax.matshow(plotdata, cmap = "RdBu_r", norm = norm)
ax.set_xticks([])
ax.set_yticks([])

fig.plot()

# %%
fig_colorbar = plt.figure(figsize=(3.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=chdm.plotting.fragments.length_norm, cmap=chdm.plotting.fragments.length_cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal")
colorbar.set_label("Fragment length")
manuscript.save_figure(fig_colorbar, "1", "colorbar_length")

# %%
sns.heatmap(enrichment_condensed["log_odds"].values[:10, None], vmin = 0)

# %%
motifscan.motifs["tfclass_family"] = motifscan.motifs["masterlist_info"].str["tfclass_family"].values

# %%
enrichment_condensed.head(30)

# %%
enrichment_condensed.head(30)

# %%
enrichment_condensed.index.get_loc(motifscan.select_motif("HES")), enrichment_condensed.index.get_loc(motifscan.select_motif("SUH"))

# %%
transcriptome.var["mean"] = transcriptome.X[:].mean(0)

# %%
enrichment_condensed.head(20)

# %%
motifs_reference = ["NFE2.H12CORE.1.SM.B", motifscan.select_motif("FOS"), motifscan.select_motif("JUN"), motifscan.select_motif("FOSB")]

# %%
slices2 = slices.loc[((slicecounts[motifs_reference] > 0).any(axis = 1))].copy()
slices_background2 = slices.loc[~((slicecounts[motifs_reference] > 0).any(axis = 1))].copy()

# %%
enrichment2 = chd.models.diff.interpret.enrichment.enrichment_foreground_vs_background(slices2.reset_index(), slices_background2.reset_index(), slicecounts_all2)

# %%
enrichment_condensed = []
# enrichment_raw = enrichment.query("q_value < 0.05").query("odds > 1").sort_values("q_value", ascending = False).copy()
enrichment_raw = enrichment2.query("q_value < 0.05").query("odds > 1").sort_values("odds", ascending = False).copy()
enrichment_raw = enrichment_raw.loc[(~enrichment_raw.index.str.startswith("ZFP")) & (~enrichment_raw.index.str.startswith("ZN")) & (~enrichment_raw.index.str.startswith("ZIC"))]
enrichment_raw = enrichment_raw.loc[motifscan.motifs.loc[enrichment_raw.index]["quality"].isin(["B", "A", "C"])]
while len(enrichment_raw) > 0:
    row = enrichment_raw.iloc[0].copy()
    motif = row.name
    to_remove = jaccards2.columns[jaccards2.loc[motif] > 0.05]
    to_remove = [motif for motif in enrichment_raw.index if motif in to_remove]

    symbols = motifscan.motifs.loc[to_remove, "symbol"].tolist()
    symbols = list(set(symbols))
    order = np.argsort(-transcriptome.var.set_index("symbol").reindex(symbols)["mean"].fillna(-1))
    row["symbols"] = [symbols[i] for i in order]

    if len(to_remove) == 0:
        raise ValueError("No more motifs to remove")

    enrichment_condensed.append(row)

    enrichment_raw = enrichment_raw.loc[~enrichment_raw.index.isin(to_remove)]
enrichment_condensed = pd.DataFrame(enrichment_condensed)

# %%
enrichment_condensed.sort_values("q_value").head(10)


# %% [markdown]
# ## Specific enrichment

# %%
def find_motif(motif):
    return motifscan.motifs.index[motifscan.motifs.index.str.contains(motif)][0]
motifclustermapping = pd.DataFrame(
    [
        [find_motif("SUH"), ["lsec-central-24h"]],
        [find_motif("FOS"), ["lsec-central-24h"]],
        [find_motif("JUN"), ["lsec-central-24h"]],
        [find_motif("GATA4"), ["lsec-central-24h"]],
        [find_motif("FOXP2"), ["lsec-central-24h"]],
        [find_motif("MEF2A"), ["lsec-central-24h"]],
        [find_motif("TCF7"), ["lsec-central-24h"]],
    ],
    columns=["motif", "clusters"],
).set_index("motif")


motifclustermapping = (
    motifclustermapping.explode("clusters")
    .rename(columns={"clusters": "cluster"})
    .reset_index()[["cluster", "motif"]]
)
motifclustermapping["motif_ix"] = motifscan.motifs.index.get_indexer(motifclustermapping["motif"])
motifclustermapping["cluster_ix"] = clustering.var.index.get_indexer(motifclustermapping["cluster"])

# %%
step = 25
desired_x = np.arange(*fragments.regions.window, step=step) - fragments.regions.window[0]

# %%
probs_mean_bins = pd.DataFrame(
    {"cut": np.array([-3., -2., -1., 0., 1., 2., np.inf])}
)
# probs_mean_bins["label"] = ["<-3", "<-2", "<-1", "<0", "<1", "<2", ">2"]
probs_mean_bins["label"] = ["Absent", "", "", "Moderate", "", "", "High"]

clusterprobs_diff_bins = pd.DataFrame(
    {"cut": np.log(np.array([0.25, 0.5, 2/3, 4/3, 2, 4, np.inf]))}
)
clusterprobs_diff_bins["label"] = ["<1/4", "", "", "1", "", "", ">4"]
# clusterprobs_diff_bins["label"] = ["<1/4", "<1/2", "<2/3", "2/3-4/3", ">4/3", ">2", ">4"]

# %%
regionpositional = genepositional

# %%
found = np.zeros((len(motifclustermapping), len(clusterprobs_diff_bins) * len(probs_mean_bins)), dtype=int)
tot = np.zeros((len(motifclustermapping), len(clusterprobs_diff_bins) * len(probs_mean_bins)), dtype=int)
for region_id in tqdm.tqdm(fragments.var.index):
# for region_id in [transcriptome.gene_id("IL1B")]:
    probs = regionpositional.probs[region_id]
    region_ix = fragments.regions.coordinates.index.get_indexer([region_id])[0]

    x_raw = probs.coords["coord"].values - fragments.regions.window[0]
    y_raw = probs.values

    y = chd.utils.interpolate_1d(
        torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
    ).numpy()
    ymean = y.mean(0)
    ymax = y.max(0)

    z = y - ymean

    ybin = np.searchsorted(probs_mean_bins["cut"].values, ymean)
    zbin = np.searchsorted(clusterprobs_diff_bins["cut"].values, z)

    # get association
    positions, indices = motifscan.get_slice(region_ix = region_ix, return_scores = False, return_strands = False)

    # get motifs
    for mapping_ix, (motif_ix, cluster_ix) in enumerate(motifclustermapping[["motif_ix", "cluster_ix"]].values):
        positions_oi = positions[indices == motif_ix] - fragments.regions.window[0]
        ixs = np.clip((positions_oi // step).astype(int), 0, len(ybin) - 1)
        tot[mapping_ix] += np.bincount(ybin * len(clusterprobs_diff_bins) + zbin[cluster_ix], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))
        found[mapping_ix] += np.bincount(ybin[ixs] * len(clusterprobs_diff_bins) + zbin[cluster_ix][ixs], minlength=len(clusterprobs_diff_bins) * len(probs_mean_bins))

# %%
tot_reshaped = tot.reshape(-1, len(probs_mean_bins), len(clusterprobs_diff_bins))
found_reshaped = found.reshape(-1, len(probs_mean_bins), len(clusterprobs_diff_bins))

# %%
plotdata = pd.DataFrame(tot_reshaped[0], columns = clusterprobs_diff_bins["label"], index = probs_mean_bins["label"])
sns.heatmap(np.log1p(plotdata))

# %%
d = found_reshaped
c = found_reshaped.sum(-1, keepdims=True).sum(-2, keepdims=True) - found_reshaped
b = tot_reshaped - found_reshaped
a = tot_reshaped.sum(-1, keepdims=True).sum(-2, keepdims=True) - b - c - d
odds = ((a*d)/(b*c))

# %%
fig = polyptich.grid.Figure(polyptich.grid.Wrap(padding_width=0.1, padding_height=0.0, ncol = 7))

cmap = mpl.cm.PiYG
norm = mpl.colors.Normalize(vmin=np.log(0.125), vmax=np.log(8.0))

for (_, (cluster, motif)), odds_motifcluster in zip(
    motifclustermapping[["cluster", "motif"]].iterrows(), odds
):
    panel, ax = fig.main.add(polyptich.grid.Panel((0.8, 0.8)))
    ax.matshow(np.log(odds_motifcluster), cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    symbol = motifscan.motifs.loc[motif]["symbol"]

    ax.set_title(f"{cluster}\n{symbol}", fontsize=8)
    ax.invert_yaxis()

# set ticks for bottom left
panel, ax = fig.main.get_bottom_left_corner()
ax.set_ylabel("Mean\naccessibility")
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.tick_params(
    axis="x", rotation=0, bottom=True, top=False, labelbottom=True, labeltop=False
)
ax.set_xlabel("Fold\naccessibility\nchange")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"])

fig.plot()

# %%
