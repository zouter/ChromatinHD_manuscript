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

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import chromatinhd_manuscript as chdm
from manuscript import Manuscript

manuscript = Manuscript(chd.get_git_root() / "manuscript")

# %%
import chromatinhd as chd
import tempfile

# %%
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_gran"
# dataset_name = "pbmc20k"
# dataset_name = "alzheimer"
# dataset_name = "e18brain"
# dataset_name = "hspc"
# dataset_name = "lymphoma"
dataset_name = "liver"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
if dataset_name == "pbmc10k/subsets/top250":
    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")

regions_name = "100k100k"
# regions_name = "10k10k"
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
import scanpy as sc

# %%
sc.pl.umap(transcriptome.adata, color = "celltype")

# %%
sc.pl.umap(transcriptome.adata, color = [transcriptome.gene_id("Cdh5")], layer = "counts")

# %%
models = chd.models.diff.model.binary.Models(chd.get_output() / "diff"/dataset_name/regions_name/"5x1"/"v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %% [markdown]
# ## Just simple examples

# %%
regions = pd.DataFrame({
    "start":[32950, 32750, 30000],
    "end":[33200, 34000, 35000],
    "resolution":[250, 500, 1000]
})
regions["resolution"] = (regions["end"] - regions["start"])

# %%
region_id = transcriptome.gene_id("IRF1")

# %%
symbol = transcriptome.var.loc[region_id, "symbol"]
breaking = chd.grid.broken.Breaking(regions, 0.05)

# %%
breaking.regions

# %%
plotdata

# %%
unstacked = plotdata["prob"].unstack()
new = []

for cluster_id, y in unstacked.iterrows():
    x = torch.linspace(*fragments.regions.window, 40000)
    z = chd.utils.interpolate_1d(x, torch.from_numpy(y.index.values), torch.from_numpy(y.values))
    new.append(z)

# %%
plotdata2 = pd.DataFrame({"prob":pd.DataFrame(
    torch.stack(new),
    columns = pd.Index(x.numpy(), name = "coord"),
    index = unstacked.index
).stack()})
plotdata2_mean = plotdata2.groupby("coord").mean()

# %%
transcriptome.gene_id("IRF1")

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

# region = fragments.regions.coordinates.loc[gene_id]
# panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10" if dataset_name == "liver" else "GRCh38", symbol = symbol, label_genome = True, show_genes = False)
# fig.main.add_under(panel_genes)
# panel_genes.ax.set_xlabel(f"{symbol}", fontstyle = "italic")

cluster_ids = clustering.cluster_info.index
cluster_info = clustering.cluster_info
cluster_info = clustering.cluster_info.loc[["CD14+ Monocytes", "CD8 activated T", "CD8 naive T", "memory B"]]
plotdata, plotdata_mean = regionpositional.get_plotdata(region_id, clusters = cluster_info.index)

panel_differential = chd.models.diff.plot.DifferentialBroken(
    # plotdata, plotdata_mean, cluster_info=cluster_info, panel_height=0.4, breaking=breaking, ymax = 5, label_accessibility = False
    plotdata2, plotdata2_mean, cluster_info=cluster_info, panel_height=0.4, breaking=breaking, ymax = 5, label_accessibility = False
)

fig.main.add_under(panel_differential)

fig.plot()

manuscript.save_figure(fig, "1", "chromatinhd_diff")

# %% [markdown]
# ## QTL

# %% [markdown]
# ### Select slices

# %%
coordinates = fragments.regions.coordinates

# %%
import chromatinhd.data.peakcounts

# %%
# slices = regionpositional.get_slices(0.)
# slices = regionpositional.calculate_slices(0., step = 25)
# slices = regionpositional.calculate_slices(-1., step = 5)
# slices = regionpositional.calculate_slices(-2., step = 25)
# top_slices = regionpositional.calculate_top_slices(slices, 1.5)

# scoring_folder = regionpositional.path / "top" / "-1-2"
# top_slices = pickle.load(open(scoring_folder / "top_slices.pkl", "rb"))

scoring_folder = regionpositional.path / "top" / "-1-1.5"
top_slices = pickle.load(open(scoring_folder / "top_slices.pkl", "rb"))

slicescores = top_slices.get_slice_scores(regions = fragments.regions)
coordinates = fragments.regions.coordinates
slices = chd.data.peakcounts.plot.uncenter_multiple_peaks(slicescores, fragments.regions.coordinates)
slicescores["slice"] = pd.Categorical(slicescores["chrom"].astype(str) + ":" + slicescores["start_genome"].astype(str) + "-" + slicescores["end_genome"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end", "chrom", "start_genome", "end_genome"]].first()

# %%
import pyranges
pr = pyranges.PyRanges(slices[['chrom', 'start_genome', 'end_genome']].rename(columns = {"chrom":"Chromosome", "start_genome":"Start", "end_genome":"End"})).merge()
pr = pr.sort()

# %%
n_desired_positions = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
n_desired_positions

# %%
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_100" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_500" / "t-test" / "scoring" / "regionpositional"
scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "encode_screen" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test-foldchange" / "scoring" / "regionpositional"
differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
differential_slices_peak.window = fragments.regions.window

# %%
slicescores_peak = differential_slices_peak.get_slice_scores(regions = fragments.regions).set_index(["region_ix", "start", "end", "cluster_ix"])["score"].unstack().max(1).to_frame("score").reset_index()

slicescores_peak = chd.data.peakcounts.plot.uncenter_multiple_peaks(slicescores_peak, fragments.regions.coordinates)
slicescores_peak["slice"] = pd.Categorical(slicescores_peak["chrom"].astype(str) + ":" + slicescores_peak["start_genome"].astype(str) + "-" + slicescores_peak["end_genome"].astype(str))
slicescores_peak = slicescores_peak.groupby("slice")[["region_ix", "start", "end", "chrom", "start_genome", "end_genome", "score"]].first()

slicescores_peak = slicescores_peak.sort_values("score", ascending=False)
slicescores_peak["length"] = slicescores_peak["end"] - slicescores_peak["start"]
slicescores_peak["cum_length"] = slicescores_peak["length"].cumsum()
slices_peak = slicescores_peak[slicescores_peak["cum_length"] <= n_desired_positions].reset_index(drop=True)

# %%
import pyranges
pr_peak = pyranges.PyRanges(slices_peak[['chrom', 'start_genome', 'end_genome']].rename(columns = {"chrom":"Chromosome", "start_genome":"Start", "end_genome":"End"})).merge()
pr_peak = pr_peak.sort()

 # %%
 (pr.as_df()["End"] - pr.as_df()["Start"]).sum()/(pr_peak.as_df()["End"] - pr_peak.as_df()["Start"]).sum()

# %% [markdown]
# ### Enrichment

# %%
import chromatinhd.data.associations
import chromatinhd.data.associations.plot
import pyranges

# %%
# motifscan_name = "gwas_immune"
# motifscan_name = "gwas_immune_main"
# motifscan_name = "gtex_immune"
# motifscan_name = "gtex_caviar_immune"
# motifscan_name = "gtex_caviar_immune_differential"
# motifscan_name = "gtex_caveman_immune_differential"
# motifscan_name = "causaldb_immune"

# motifscan_name = "gwas_hema_main"

# motifscan_name = "gwas_liver"
motifscan_name = "gwas_liver_main"
# motifscan_name = "gtex_caviar_liver"
# motifscan_name = "causaldb_liver"
# motifscan_name = "gtex_liver"

# motifscan_name = "gwas_lymphoma"
# motifscan_name = "gwas_lymphoma_main"
# motifscan_name = "causaldb_lymphoma"

# motifscan_name = "gwas_cns_main"
associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
association = associations.association

# %%
association["start"] = (association["pos"]).astype(int)
association["end"] = (association["pos"] + 1).astype(int)

# %%
pr_snps = pyranges.PyRanges(association.reset_index()[["chr", "start", "end", "index"]].rename(columns = {"chr":"Chromosome", "start":"Start", "end":"End"}))
overlap = pr_snps.intersect(pr)

haplotype_scores = association[["snp", "disease/trait"]].copy()
haplotype_scores["n_matched"] = (haplotype_scores.index.isin(overlap.as_df()["index"])).astype(int)
haplotype_scores["n_total"] = 1

# %%
matched = haplotype_scores["n_matched"].sum()
total_snps = haplotype_scores["n_total"].sum()
total_diff = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
total_positions = fragments.regions.width * fragments.n_regions

contingency = pd.DataFrame([
    [matched, total_snps - matched],
    [total_diff - matched, total_positions - total_snps - total_diff + matched]
], index = ["SNP", "Not SNP"], columns = ["In slice", "Not in slice"])
contingency

from scipy.stats import fisher_exact
fisher_exact(contingency)

# %%
pr_snps = pyranges.PyRanges(association.reset_index()[["chr", "start", "end", "index"]].rename(columns = {"chr":"Chromosome", "start":"Start", "end":"End"}))
overlap = pr_snps.intersect(pr_peak)

haplotype_scores_peak = association[["snp", "disease/trait"]].copy()
haplotype_scores_peak["n_matched"] = (haplotype_scores_peak.index.isin(overlap.as_df()["index"])).astype(int)
haplotype_scores_peak["n_total"] = 1

# %%
matched = haplotype_scores_peak["n_matched"].sum()
total_snps = haplotype_scores_peak["n_total"].sum()
total_diff = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
total_positions = fragments.regions.width * fragments.n_regions

contingency = pd.DataFrame([
    [matched, total_snps - matched],
    [total_diff - matched, total_positions - total_snps - total_diff + matched]
], index = ["SNP", "Not SNP"], columns = ["In slice", "Not in slice"])
contingency

from scipy.stats import fisher_exact
fisher_exact(contingency)

# %%
contingency

# %%
(
    (haplotype_scores.join(haplotype_scores_peak, rsuffix = "_peak").query("(n_matched > 0) & (n_matched_peak == 0)").shape[0] + 1e-3)/
    (haplotype_scores.join(haplotype_scores_peak, rsuffix = "_peak").query("(n_matched == 0) & (n_matched_peak > 0)").shape[0] + 1e-3)
)

# %% [markdown]
# ### Find good example SNPs

# %%
# # !wget https://storage.googleapis.com/adult-gtex/references/v8/reference-tables/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.lookup_table.txt.gz
# lookup_table = pd.read_csv("GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.lookup_table.txt.gz", sep = "\t")

# %%
# lookup_table["chr2"] = lookup_table["chr"].str.replace("chr", "")
# lookup_table["id2"] = lookup_table["chr2"] + "_" + lookup_table["variant_pos"].astype(str)
# id2_to_id = dict(zip(lookup_table["id2"].values.tolist(), lookup_table["rs_id_dbSNP151_GRCh38p7"].values.tolist()))

# %%
# haplotype_scores.join(haplotype_scores_peak, rsuffix = "_peak").query("(n_matched > 0)").head(20)

# variants not detected by peaks
snps_oi = haplotype_scores.join(haplotype_scores_peak, rsuffix = "_peak").query("(n_matched > 0) & (n_matched_peak == 0)").groupby(["snp"]).agg({"disease/trait":tuple})
snps_oi["n_traits"] = [len(x)+1 if isinstance(x, tuple) else 0 for x in snps_oi["disease/trait"]]
# snps_oi.sample(5, replace = False).sort_values("disease/trait")
snps_oi.sort_values("n_traits", ascending = False).head(20)

# variants not detected by ChromatinHD but that are detected by peaks
# haplotype_scores.join(haplotype_scores_peak, rsuffix = "_peak").query("(n_matched == 0) & (n_matched_peak > 0)").head(20)

# map to rsid
# snps_oi["rsid"] = [id2_to_id.get(x) for x in snps_oi.index]

# %%
# for adadastra
# print("\n".join(snps_oi.index))
# print("\n".join(snps_oi["rsid"]))

# %%
# in GTEX, we can filter on gene
if "gtex" in motifscan_name:
    association_oi = association.query("snp in @snps_oi.index").copy()
    association_oi["symbol"] = transcriptome.var["symbol"].reindex(association_oi["gene"]).values

    print(association_oi.query("symbol == 'CCL4'"))

# %%
# pbmc10k rs367809717: example in "desert"
# pbmc10k rs10156618: example of rare-cell type specific
# pbmc10k rs1077667: example of shoulder/in background
# pbmc10k rs2284553: example of shoulder/in background
# pbmc10k rs7767167: example of within-CRE heterogeneity
# pbmc10k rs642135: example of shoulder
# pbmc10k rs6062498: example of shoulder

# lymphoma rs1800682: example of broader opening
# lymphoma rs4845725: example of too narrow encode screen

# liver rs763607523: example of in shoulder of promoter with broader opening (-> multiscale), probably a promoter being reused for a distant gene because the gene is not expressed
# liver rs6598541: example in the "desert" with broader opening
# liver rs6882076: example in the "desert", which is actually the TSS of Timd4
# liver rs55682243: example in the "desert"
# liver rs7412: example of a more complex accessibility landscape
# liver rs36209093: example of shoulder of a clear peak (has no LD! https://www.ebi.ac.uk/gwas/variants/rs36209093)

# rs_oi = association.query("snp == 'rs7668673'").iloc[0] # PBMC

# gtex
# liver 7_8261212 (gtex_caviar_liver): example of within the 

# rs_oi = association.query("snp == 'rs888801'").iloc[0]
# rs_oi = association.query("snp == 'rs80207740'").iloc[0] # WTF just a small thing in one cell type
# rs_oi = association.query("snp == 'rs9619658'").iloc[0] # WTF just a small blip
# rs_oi = association.query("snp == 'rs12718598'").iloc[0]
# rs_oi = association.query("snp == 'rs6592965'").iloc[0]
# rs_oi = association.query("snp == 'rs17840121'").iloc[0]

# rs_oi = association.query("snp == 'rs6592965'").iloc[0]

# rs_oi = association.query("snp == 'rs443623'").iloc[0]
# rs_oi = association.query("snp == 'rs10751776'").iloc[0]
# rs_oi = association.query("snp == 'rs7668673'").iloc[0]
# rs_oi = association.query("snp == 'rs3784789'").iloc[0]

# rs_oi = association.loc[association["snp"] == snps_oi.query("rsid == 'rs4752263'").index[0]].iloc[0]

cluster_ids = clustering.cluster_info.index
relative_to = None


#### rs443623 PBMC20K
# assert dataset_name == "pbmc20k"
# assert motifscan_name == "causaldb_immune"
# rs_oi = association.query("snp == 'rs443623'").iloc[0]
# peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name)
# peakcallers = peakcallers.loc[["macs2_leiden_0.1_merged", "macs2_summits", "encode_screen"]]
# cluster_ids = ["naive B", "FCGR3A+ Monocytes", "cDCs"]
# relative_to = ["naive B"]


#### rs7668673 PBMC10K
assert dataset_name == "pbmc10k"
assert motifscan_name == "gwas_immune_main"
rs_oi = association.query("snp == 'rs7668673'").iloc[0]
peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name)
peakcallers = peakcallers.loc[["macs2_leiden_0.1_merged", "macs2_summits", "encode_screen"]]
cluster_ids = ["naive B", "cDCs", "FCGR3A+ Monocytes"];relative_to = None


# ### rs875742 HSPC
# assert dataset_name == "hspc"
# assert motifscan_name == "gwas_hema_main"
# rs_oi = association.query("rsid == 'rs875742'").iloc[0]
# peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name)
# peakcallers = peakcallers.loc[["macs2_leiden_0.1_merged", "macs2_summits", "encode_screen"]]
# cluster_ids = ["MEP", "Erythrocyte precursors", "Erythroblast"];relative_to = ["MEP"]

# %%
# association.loc[association.snp == rs_oi.snp]

# %%
def dist(x):
    return (rs_oi["pos"] - x["tss"])


# %%
# find nearest region (i.e. gene) to SNP
regions_oi = fragments.regions.coordinates.query("chrom == @rs_oi.chr").copy()
regions_oi["dist"] = regions_oi.apply(dist, axis=1)
regions_oi["absdist"] = regions_oi["dist"].abs()
region_info = regions_oi.iloc[np.argsort(regions_oi["dist"].abs().values)].iloc[0]

region_info.name, rs_oi["snp"], transcriptome.symbol(region_info.name), region_info["tss"], rs_oi["pos"], region_info["strand"], region_info["dist"]

# %%
window = np.array([region_info["dist"] - 1500, region_info["dist"] + 1500])
if region_info["strand"] == -1:
    window = -window[::-1]

gene_id = region_info.name
symbol = transcriptome.symbol(gene_id)
association["rsid"] = association["snp"]

# %%
motifscan_name2 = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name2)
motifs_oi = pd.DataFrame([
    # [motifscan.select_motif("TAL1")],
    # [motifscan.select_motif("GATA1")],
    # [motifscan.select_motif("CEBPD")],
    # [motifscan.select_motif("NFIL3")],
    # [motifscan.select_motif("FOS")],
    # [motifscan.select_motif("ATF3")],
    # [motifscan.select_motif("NFIC")],
    # [motifscan.select_motif("PATZ1")],
    # [motifscan.select_motif("HCFC1")],
    # [motifscan.select_motif("CREB1")],
    [motifscan.select_motif("SPI1")],
    # [motifscan.select_motif("TEAD4")],
#     [motifscan.select_motif("IRF3")],
#     [motifscan.select_motif("CEBPD")],
#     [motifscan.select_motif("ATF7")],
#     [motifscan.select_motif("NFAC3")],
    [motifscan.select_motif("CEBPA")],
    # [motifscan.select_motif("CEBPE")],
    # [motifscan.select_motif("CEBPD")],
    # [motifscan.select_motif("ELF1")],
    # [motifscan.select_motif("GABPA")],
    # ["CTCFL.H12CORE.0.P.B"],
], columns = ["motif"]).set_index("motif")

# motifs_oi = pd.DataFrame(columns = ["motif"]).set_index("motif")

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

width = (window[1] - window[0]) / 2000

region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10" if dataset_name == "liver" else "GRCh38", symbol = symbol, label_genome = True, show_genes = False)
fig.main.add_under(panel_genes)
panel_genes.ax.set_xlabel(f"{symbol}", fontstyle = "italic")

# cluster_info = clustering.cluster_info; relative_to = None
# cluster_info = clustering.cluster_info.loc[["cDCs", "FCGR3A+ Monocytes", "naive B", "memory B"]]; relative_to = "naive B"
# cluster_info = clustering.cluster_info.loc[["HSPC", "MEP", "Erythroblast", "Erythrocyte precursors"]]; relative_to = "HSPC"
# cluster_info = clustering.cluster_info.loc[["cDCs", "FCGR3A+ Monocytes", "pDCs"]]
# cluster_info = clustering.cluster_info.loc[["cDCs", "FCGR3A+ Monocytes", "CD14+ Monocytes", "pDCs", "naive B", "memory B", "CD4 naive T"]]; relative_to = ["pDCs", "naive B", "memory B"]
# cluster_info = clustering.cluster_info; relative_to = ["cDCs"]
cluster_info = clustering.cluster_info.loc[cluster_ids]
plotdata, plotdata_mean = regionpositional.get_plotdata(gene_id, clusters = cluster_info.index)

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome=transcriptome, clustering=clustering, gene=gene_id, panel_height=0.4, order = False, cluster_info = cluster_info
)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=cluster_info, panel_height=0.4, width=width, window = window, ymax = 5, relative_to = relative_to, label_accessibility = False
)

fig.main.add_under(panel_differential)
# fig.main.add_right(panel_expression, row=panel_differential)

# if len(motifs_oi) > 0:
#     panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, gene_id, motifs_oi = motifs_oi, width = width, window = window)
#     fig.main.add_under(panel_motifs)

panel_association = chd.data.associations.plot.Associations(associations, gene_id, width = width, window = window, show_ld = False, label_y = False)
fig.main.add_under(panel_association)

import chromatinhd_manuscript as chdm
# panel_peaks = chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, peakcallers = peakcallers, window = window, width = width, label_rows = False, label_methods = False)
panel_peaks = chd.data.peakcounts.plot.Peaks.from_bed(region, peakcallers = peakcallers, window = window, width = width, label_rows = False, label_methods = False)
fig.main.add_under(panel_peaks)

fig.plot()

manuscript.save_figure(fig, "3", "qtl_examples", rs_oi.rsid)

# %%
sc.pl.umap(transcriptome.adata, color = [transcriptome.gene_id("GATA3"), "celltype"])

# %%
transcriptome.var.loc[transcriptome.var.symbol.str.contains("GATA")]

# %%
for rsid in snps_oi.sort_values("n_traits", ascending = False).head(50).index:
    rs_oi = association.query("snp == @rsid").iloc[0]

    # find nearest region (i.e. gene) to SNP
    regions_oi = fragments.regions.coordinates.query("chrom == @rs_oi.chr").copy()
    regions_oi["dist"] = regions_oi.apply(dist, axis=1)
    region_info = regions_oi.iloc[np.argsort(regions_oi["dist"].abs().values)].iloc[0]

    region_info.name, rs_oi["snp"], transcriptome.symbol(region_info.name), region_info["tss"], rs_oi["pos"], region_info["strand"], region_info["dist"]

    window = np.array([region_info["dist"] - 1500, region_info["dist"] + 1500])
    if region_info["strand"] == -1:
        window = -window[::-1]
    gene_id = region_info.name

    fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

    width = (window[1] - window[0]) / 2000

    region = fragments.regions.coordinates.loc[gene_id]
    panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10" if dataset_name == "liver" else "GRCh38")
    fig.main.add_under(panel_genes)

    cluster_info = clustering.cluster_info; relative_to = None
    # cluster_info = clustering.cluster_info.loc[["cDCs", "FCGR3A+ Monocytes", "pDCs"]]
    # cluster_info = clustering.cluster_info.loc[["cDCs", "FCGR3A+ Monocytes", "pDCs", "naive B", "memory B", "CD4 naive T"]]; relative_to = ["pDCs", "naive B", "memory B"]
    # cluster_info = clustering.cluster_info; relative_to = ["cDCs"]
    plotdata, plotdata_mean = regionpositional.get_plotdata(gene_id, clusters = cluster_info.index)

    panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
        transcriptome=transcriptome, clustering=clustering, gene=gene_id, panel_height=0.4, order = True, cluster_info = cluster_info
    )

    panel_differential = chd.models.diff.plot.Differential(
        plotdata, plotdata_mean, cluster_info=cluster_info, panel_height=0.4, width=width, window = window, order = panel_expression.order, ymax = 5, relative_to = relative_to
    )

    fig.main.add_under(panel_differential)
    fig.main.add_right(panel_expression, row=panel_differential)

    # panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, gene_id, motifs_oi = motifs_oi, width = width, window = window)
    # fig.main.add_under(panel_motifs)

    panel_association = chd.data.associations.plot.Associations(associations, gene_id, width = width, window = window, show_ld = False)
    fig.main.add_under(panel_association)

    import chromatinhd_manuscript as chdm
    panel_peaks = chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = window, width = width)
    fig.main.add_under(panel_peaks)

    fig.plot()

    fig.savefig(chd.get_output() / "plots" / f"{rsid}.png", dpi = 300, bbox_inches = "tight")
    plt.close()

# %%
(chd.get_output() / "plots").mkdir(exist_ok = True, parents = True)

# %% [markdown]
# ### Temporary SPI1 - ADADASTRA look

# %%
import chromatinhd.data.associations
import chromatinhd.data.associations.plot

# motifscan_name = "causaldb_immune"
# motifscan_name = "gwas_hema"
motifscan_name = "gwas_immune_main"
motifscan_name = "gwas_immune"
# motifscan_name = "gtex_immune"
associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)
snps_spi1 = pd.read_table("snps_spi1.tsv", sep = "\t").query("motif_conc == 'Discordant'")
associations.association.loc[associations.association["snp"].isin(snps_spi1["ID"])]["snp"].value_counts().head(10)

# rs3806407 nyet?
# rs41306506 in peak
# rs4851252 at shoulder AFF3!!
# rs6797467 nyet
# rs16970707 at semi-shoulder
# rs2060984 nyet
# rs12476895 in peak
# rs12476896 in peak
# rs72835101 in specific part of the peak
# rs7596236 nyet?
# rs62489382 nyet ?
# rs7692976 in shoulder!
# rs3784789 nyet
# rs2835357 nyet
# rs608793 nyet
# rs608793 in peak
# rs12508069 nyet
# rs228590 outside of peak, desert!
# rs9310077 outside of peak, desert!

# %%
pd.read_table("snps_spi1.tsv", sep = "\t").motif_conc.value_counts()

# %% [markdown]
# ## Motifs

# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

clustering.var["n_cells"] = clustering.labels.value_counts()

# %% [markdown]
# ### Load

# %%
# slices = regionpositional.calculate_slices(-2., step = 5)
# differential_slices = regionpositional.calculate_differential_slices(slices, fc_cutoff = 4.)

# slices = regionpositional.calculate_slices(-1., step = 5)
# differential_slices = regionpositional.calculate_differential_slices(slices, fc_cutoff = 4.)

# slices = regionpositional.calculate_slices(0., step = 5)
# differential_slices = regionpositional.calculate_differential_slices(slices, fc_cutoff = 4.)

# scoring_folder = regionpositional.path / "differential" / "-1-1.5"
# differential_slices = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

scoring_folder = regionpositional.path / "differential" / "-1-3"
differential_slices = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

# scoring_folder = regionpositional.path / "differential" / "-1-2"
# differential_slices = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

# %%
slicescores = differential_slices.get_slice_scores(regions = fragments.regions, clustering = clustering)

slicescores["slice"] = pd.Categorical(slicescores["region_ix"].astype(str) + ":" + slicescores["start"].astype(str) + "-" + slicescores["end"].astype(str))
slices = slicescores.groupby("slice")[["region_ix", "start", "end"]].first()

# %%
n_desired_positions = slicescores.groupby("cluster")["length"].sum()
n_desired_positions

# %%
slicecounts = motifscan.count_slices(slices)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(slicescores, slicecounts)
enrichment["log_odds"] = np.log(enrichment["odds"])

# %%
# enrichment.query("q_value < 0.05").sort_values("odds", ascending = False).loc["pDCs"].head(30)

# %% [markdown]
# ### Compare directly with a peak-based approach

# %%
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_100" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_500" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_500" / "t-test-foldchange" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_summits" / "logreg" / "scoring" / "regionpositional"
scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_summits" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_summits" / "t-test-foldchange" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_summit" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "encode_screen" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "encode_screen" / "t-test-foldchange" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test-foldchange" / "scoring" / "regionpositional"
differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
differential_slices_peak.window = fragments.regions.window

# %%
# match # of differential within each cluster
slicescores_peak_full = differential_slices_peak.get_slice_scores(regions = fragments.regions, clustering = clustering)
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
enrichment["symbol"] = motifscan.motifs["symbol"].reindex(enrichment.index.get_level_values("motif")).values

# %%
# enrichment.loc["Erythroblast"].sort_values("odds", ascending = False).head(50)

# %%
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"].str.contains("KLF1")]
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"] == "LHX2"]
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"] == "SMAD2"]
motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"] == "TCF7L2"]
# motifs_oi = motifscan.motifs.index[motifscan.motifs["HUMAN_gene_symbol"] == "RBPJ"]
enrichment_peak.loc[enrichment_peak.index.get_level_values("motif").isin(motifs_oi)].join(enrichment.loc[enrichment.index.get_level_values("motif").isin(motifs_oi)], lsuffix = "_peak", rsuffix = "_chd")[["log_odds_peak", "log_odds_chd"]].sort_values("log_odds_chd").style.background_gradient(vmin = -2, vmax = 2, cmap = "RdBu_r")

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

# %% [markdown]
# ### Diffexp compare

# %%
if dataset_name in ["liver", "e18brain"]:
    motifscan.motifs["symbol"] = motifscan.motifs["MOUSE_gene_symbol"]
else:
    motifscan.motifs["symbol"] = motifscan.motifs["HUMAN_gene_symbol"]

# %%
motifs_oi = motifscan.motifs.sort_values("quality").copy()#.reset_index().groupby("symbol").first().reset_index().set_index("motif")
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
adata_raw = adata_raw[:, transcriptome.var.index]
adata_raw.obs["cluster"] = clustering.labels
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
import scanpy as sc
sc.tl.rank_genes_groups(adata_raw, groupby="cluster", method="t-test")

diffexp = sc.get.rank_genes_groups_df(adata_raw, None).rename(columns = {"names":"gene", "group":"cluster"}).set_index(["cluster", "gene"])
diffexp["significant_up"] = (diffexp["pvals_adj"] < 0.01) & (diffexp["scores"] > 10)
diffexp["significant_down"] = (diffexp["pvals_adj"] < 0.01) & (diffexp["scores"] < -10)
diffexp["significant"] = diffexp["significant_up"] | diffexp["significant_down"]
diffexp["score"] = diffexp["scores"]

# %%
# sc.tl.rank_genes_groups(adata_raw, groupby="cluster", groups = ["Myeloid"], reference = "Erythroblast", method="t-test", key_added = "myeloid_vs_erythroblast")
# myeloid_vs_erythroblast = pd.DataFrame({
#     "gene":adata_raw.uns["myeloid_vs_erythroblast"]["names"].tolist(),
#     "scores":adata_raw.uns["myeloid_vs_erythroblast"]["scores"].tolist(),
#     "pvals_adj":adata_raw.uns["myeloid_vs_erythroblast"]["pvals_adj"].tolist(),
#     "logfoldchanges":adata_raw.uns["myeloid_vs_erythroblast"]["logfoldchanges"].tolist(),
# }).apply(lambda x:x.str[0])

# %%
enrichment["gene"] = motifs_oi["gene"].reindex(enrichment.index.get_level_values("motif")).values
enrichment_peak["gene"] = motifs_oi["gene"].reindex(enrichment_peak.index.get_level_values("motif")).values

# %%
# enrichment.loc["MEP"].loc[motifscan.select_motifs("GATA1")]

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
enrichment.loc["KC"].query("q_value < 0.1").sort_values("log_odds", ascending = False).head(20)

# %%
enrichment["aggregate"] = enrichment["log_odds"] * enrichment["score"]
enrichment_peak["aggregate"] = enrichment_peak["log_odds"] * enrichment_peak["score"]
enrichment.sort_values("aggregate", ascending = False).head(20)

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

# %%
# enrichment.loc["CD14+ Monocytes"].sort_values("odds", ascending = False).query("score > 1")[["q_value", "odds", "pvals_adj", "logfoldchanges", "scores", "gene"]].head(40).style.bar()

# %% [markdown]
# ### Odds across motifs

# %%
enrichment_oi = enrichment.query("(score > 3) & (odds > 1.5) & (q_value < 0.1)")
enrichment_oi = enrichment_oi.loc[~enrichment_oi.index.get_level_values("motif").str.startswith("E2")]
# enrichment_oi = enrichment_oi.loc[motifscan.motifs.loc[enrichment_oi.index.get_level_values("motif"), "quality"].isin(["A", "B"]).values]
enrichment_peak_oi = enrichment_peak.loc[enrichment_oi.index]

enrichment_oi = enrichment_oi.join(enrichment_peak_oi, lsuffix = "_chd", rsuffix = "_peak")

enrichment_oi["odds_ratio"] = enrichment_oi["odds_chd"]/enrichment_oi["odds_peak"]

enrichment_oi["n_gained_peak"] = [row["contingency_peak"][1, 1] for _, row in enrichment_oi.iterrows()]
enrichment_oi["n_gained_chd"] = [row["contingency_chd"][1, 1] for _, row in enrichment_oi.iterrows()]

enrichment_oi["n_gained_chd"].mean() / enrichment_oi["n_gained_peak"].mean()

enrichment_oi["rel_gained_chd"] = enrichment_oi["n_gained_chd"] / enrichment_oi[["n_gained_chd", "n_gained_peak"]].min(1)
enrichment_oi["rel_gained_peak"] = enrichment_oi["n_gained_peak"] / enrichment_oi[["n_gained_chd", "n_gained_peak"]].min(1)
enrichment_oi["diff"] = enrichment_oi["rel_gained_chd"] - enrichment_oi["rel_gained_peak"]
enrichment_oi["diff_gained"] = enrichment_oi["n_gained_chd"] - enrichment_oi["n_gained_peak"]
enrichment_oi["rel_gained"] = (enrichment_oi["diff_gained"]) / enrichment_oi[["n_gained_chd", "n_gained_peak"]].max(1)
enrichment_oi["rel"] = (enrichment_oi["n_gained_chd"]) / (enrichment_oi["n_gained_peak"])
pseudocount = 100
enrichment_oi["rel_pseudocount"] = (enrichment_oi["n_gained_chd"]+pseudocount) / (enrichment_oi["n_gained_peak"]+pseudocount)
# enrichment_oi["rel"] = enrichment_oi["odds_chd"] - enrichment_oi["odds_peak"]
# enrichment_oi["rel"] = np.exp(enrichment_oi["log_odds_chd"] - enrichment_oi["log_odds_peak"])

# %%
fig, ax = plt.subplots()
ax.scatter(np.log(enrichment_oi["odds_ratio"]), (enrichment_oi["rel"]))

# %%
import textwrap

# %%
plotdata = enrichment_oi
plotdata = plotdata.sort_values("rel", ascending=False)
plotdata["ix"] = np.arange(len(plotdata))

plotdata_oi = None
# plotdata_oi = plotdata.iloc[[0]].assign(cluster = "0")
if dataset_name == "hspc":
    plotdata_oi = pd.concat(
        [
            plotdata.loc["HSPC"].loc[[motifscan.select_motifs("ERG")[0]]].assign(cluster = "HSPC"),
            plotdata.loc["Erythroblast"].loc[[motifscan.select_motifs("GATA1")[0]]].assign(cluster = "Erythroblast"),
            plotdata.loc["Granulocyte precursor"].loc[[motifscan.select_motifs("GATA2")[0]]].assign(cluster = "Granulocyte precursor"),
        ]
    )

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

# panel, ax = fig.main.add_under(chd.grid.Panel((3, 0.2)))

# ax.boxplot([np.log(plotdata["rel"])], vert=False, showfliers=False, widths = 0.8)
# ax.set_xlim(np.log(0.5), np.log(3))
# ax.set_ylim(0.5, 1.5)
# ax.set_xticks(np.log([0.5, 1, 1.5, 2.0, 3.0]))
# ax.set_xticklabels(["-50%", "0", "+50%", "+100%", "+200%"])
# sns.despine(ax=ax, left=True, bottom=True, top = False)
# ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# ax.set_yticks([])

panel, ax = fig.main.add_under(chd.grid.Panel((2, 2.)))
# ax.set_xlim(0.1, 10)
ax.barh(y=plotdata["ix"], width=(np.log(plotdata["rel"])), color="#333", lw=0.0, height=1, left=0)
ax.axvline(plotdata["rel"].mean())
ax.set_xlim(np.log(0.5), np.log(4))
ax.set_xticks(np.log([0.5, 0.66666, 1, 1.5, 2.0, 3.0]))
ax.set_xticklabels(["×½", "×⅔", "0", "×1.5", "×2", "×3"])
sns.despine(ax=ax, left=True, bottom=False)
ax.set_yticks([])
ax.set_ylim(-1, len(plotdata))
# put ticks on top

if plotdata_oi is not None:
    ax.barh(y=plotdata_oi["ix"], width=(np.log(plotdata_oi["rel"])), color="tomato", lw=0.0, height=1, left=0)
    for i, (ix, row) in enumerate(plotdata_oi.iterrows()):
        if i == 0:
            offset = (10, -30)
            ha = "right"
            va = "top"
        elif i == 1:
            offset = (5, 30)
            ha = "left"
            va = "center"
        elif i == 2:
            offset = (-50, 40)
            ha = "left"
            va = "center"

        label = "\n".join(textwrap.wrap(row["cluster"], width = 20))
        text = f"{motifscan.motifs.loc[ix, 'symbol']}\n{label}\n{row['n_gained_chd']} vs {row['n_gained_peak']}"
        ax.annotate(
            # xy=(0, row["ix"]),
            xy=(np.log(row["rel"]), row["ix"]),
            xytext=offset,
            textcoords="offset points",
            text=text,
            ha=ha,
            va=va,
            arrowprops=dict(arrowstyle="-", lw=0.5, color = "tomato", shrinkB = 0., shrinkA = 0.),
            bbox=dict(facecolor="white", edgecolor="tomato", alpha=0.8),
            clip_on=False,
            annotation_clip = False,
            zorder = 20,
        )
mean = np.log(plotdata["rel"]).mean()
ax.axvline(mean, color = "grey", ls = "--", zorder = 10)
ax.annotate(
    xy=(mean, len(plotdata)),
    xytext=(0, 5),
    textcoords="offset points",
    text=f"x{np.exp(mean):.2f}",
    ha="center",
    va="center",
    zorder = 20,
)
ax.set_xlabel("Ratio differential TFBS\nChromatinHD vs MACS2 summits")

fig.plot()

if dataset_name == "hspc":
    manuscript.save_figure(fig, "3", "relative_motif_enrichment")

# %%
stripe_factors = pd.read_table(chd.get_output() / "data" / "stripe_factors.tsv", names = ["tf", "n_datasets", "perc_stripe"])
stripe_factors["perc_stripe"] = stripe_factors["perc_stripe"].str.rstrip("%").astype(int)

# %%
plotdata["symbol"] = motifscan.motifs["symbol"].reindex(plotdata.index.get_level_values("motif")).values

# %%
transcriptome.adata.obs["cluster"] = clustering.labels  

# %%
sc.pl.umap(transcriptome.adata, color = ["cluster"])

# %%
plotdata["transient"] = plotdata.index.get_level_values("cluster").isin(["Granulocyte precursor", "MPP", "GMP", "MEP"])
# plotdata["transient"] = plotdata.index.get_level_values("cluster").isin(["2"])
plotdata["stripe"] = plotdata["symbol"].str.upper().isin(stripe_factors.query("perc_stripe > 25")["tf"])

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

plotdata_oi = plotdata.loc[plotdata["transient"]]

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2.)))
ax.barh(y=plotdata["ix"], width=(np.log(plotdata["rel"])), color="#333", lw=0.0, height=1, left=0)
ax.barh(y=plotdata_oi["ix"], width=(np.log(plotdata_oi["rel"])), color="#0074D9", lw=0.0, height=1, left=0)
ax.set_title("Enrichment in\ntransient cell states")
ax.set_xlim(np.log(0.5), np.log(4))
ax.set_xticks(np.log([0.5, 0.66666, 1, 1.5, 2.0, 3.0]))
ax.set_xticklabels(["×½", "×⅔", "0", "×1.5", "×2", "×3"])
sns.despine(ax=ax, left=True, bottom=False)
ax.set_yticks([])
ax.set_ylim(-1, len(plotdata))

plotdata_oi = plotdata.loc[plotdata["stripe"]]

panel, ax = fig.main.add_right(chd.grid.Panel((2, 2.)))
ax.barh(y=plotdata["ix"], width=(np.log(plotdata["rel"])), color="#333", lw=0.0, height=1, left=0)
ax.barh(y=plotdata_oi["ix"], width=(np.log(plotdata_oi["rel"])), color="#FF4136", lw=0.0, height=1, left=0)
ax.set_title("Enrichment of stripe-factors")
ax.set_xlim(np.log(0.5), np.log(4))
ax.set_xticks(np.log([0.5, 0.66666, 1, 1.5, 2.0, 3.0]))
ax.set_xticklabels(["×½", "×⅔", "0", "×1.5", "×2", "×3"])
sns.despine(ax=ax, left=True, bottom=False)
ax.set_yticks([])
ax.set_ylim(-1, len(plotdata))

fig.plot()

if dataset_name == "hspc":
    manuscript.save_figure(fig, "3", "relative_motif_enrichment_features")

# %%
plotdata.sort_values("rel", ascending = False).head(20)

# %%
links_file = chd.get_output() / "data" / "stringdb" / "10090.protein.physical.links.full.v12.0.txt"
links_file.parent.mkdir(exist_ok = True, parents = True)
if not links_file.exists():
    # !wget https://stringdb-downloads.org/download/protein.physical.links.full.v12.0/10090.protein.physical.links.full.v12.0.txt.gz -O {links_file}.gz
    # !gunzip {links_file}.gz

protein_file = chd.get_output() / "data" / "stringdb" / "10090.protein.info.v12.0.txt"
if not protein_file.exists():
    # !wget https://stringdb-downloads.org/download/protein.info.v12.0/10090.protein.info.v12.0.txt.gz -O {protein_file}.gz
    # !gunzip {protein_file}.gz


links_file = chd.get_output() / "data" / "stringdb" / "9606.protein.physical.links.full.v12.0.txt"
if not links_file.exists():
    # !wget https://stringdb-downloads.org/download/protein.physical.links.full.v12.0/9606.protein.physical.links.full.v12.0.txt.gz -O {links_file}.gz
    # !gunzip {links_file}.gz

protein_file = chd.get_output() / "data" / "stringdb" / "9606.protein.info.v12.0.txt"
if not protein_file.exists():
    # !wget https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz -O {protein_file}.gz
    # !gunzip {protein_file}.gz

# %%
# protein_file = chd.get_output() / "data" / "stringdb" / "10090.protein.info.v12.0.txt"
# links_file = chd.get_output() / "data" / "stringdb" / "10090.protein.physical.links.full.v12.0.txt"

protein_file = chd.get_output() / "data" / "stringdb" / "9606.protein.info.v12.0.txt"
links_file = chd.get_output() / "data" / "stringdb" / "9606.protein.physical.links.full.v12.0.txt"

protein_info = pd.read_table(protein_file, index_col = 0)
links = pd.read_table(links_file, sep = " ")

# %%
links["symbol1"] = protein_info.reindex(links["protein1"])["preferred_name"].str.upper().values
links["symbol2"] = protein_info.reindex(links["protein2"])["preferred_name"].str.upper().values

links = links.query("symbol1 != symbol2")
links = links.loc[links["symbol1"].isin(motifscan.motifs["HUMAN_gene_symbol"])]
links = links.loc[links["symbol2"].isin(motifscan.motifs["HUMAN_gene_symbol"])]
links = links.loc[links["combined_score"] > 500]
links.shape

# %%
dataset = chd.biomart.Dataset()
uniprot_mapping = dataset.get(
    attributes = [
        dataset.attribute("ensembl_gene_id"),
        dataset.attribute("uniprotswissprot")
    ],
    filters = [
        dataset.filter("ensembl_gene_id", value = motifs_oi["gene"].unique())
    ]
)
uniprots = uniprot_mapping.dropna().groupby("ensembl_gene_id").first()["uniprotswissprot"]

# %%
import requests
import io

if not (chd.get_output() / "data" / "iupred3.pkl").exists():
    iupred_scores = {}
else:
    iupred_scores = pickle.load((chd.get_output() / "data" / "iupred3.pkl").open("rb"))

for uniprot in tqdm.tqdm(uniprots):
    if uniprot in iupred_scores:
        continue
    response = requests.get('https://iupred3.elte.hu/iupred3/' + uniprot)
    iupred_scores[uniprot] = pd.read_table(io.StringIO(response.text.replace("<pre>", "").replace("</pre>", "")), comment = "#", sep = "\t", names = ["pos", "aa", "iupred_score", "?"])

pickle.dump(iupred_scores, (chd.get_output() / "data" / "iupred3.pkl").open("wb"))

idrs = {}
for uniprot, iupred_score in iupred_scores.items():
    idrs[uniprot] = (iupred_score["iupred_score"] >= 0.5).mean()
idrs = pd.Series(idrs)[uniprots.values]
idrs.index = uniprots.index
transcriptome.var["idr_50_2"] = idrs.reindex(transcriptome.var.index)

# %%
# # idr_4 (consensus d2p2)
# dataset = chd.biomart.Dataset()
# uniprot_mapping = dataset.get(
#     attributes = [
#         dataset.attribute("ensembl_gene_id"),
#         dataset.attribute("uniprotswissprot")
#     ],
#     filters = [
#         dataset.filter("ensembl_gene_id", value = motifs_oi["gene"].unique())
#     ]
# )

# uniprots = uniprot_mapping.dropna().groupby("ensembl_gene_id").first()["uniprotswissprot"]

# import requests
# import json
# data = json.dumps(uniprots.tolist())
# response = requests.get('http://d2p2.pro/api/seqid/' + data)
# response_data = response.json()

# idrs = []
# for uniprot in uniprots:
#     try:
#         consensus = np.array(response_data[uniprot][0][2]["disorder"]["consensus"])
#         idrs.append((consensus >= 4).sum())
#     except:
#         idrs.append(np.nan)
# idrs = pd.Series(idrs, index = uniprots.index)

# transcriptome.var["idr_4"] = idrs.reindex(transcriptome.var.index)

# %%
# idr_50 (guido)
tfs = pd.read_csv("human_protein.txt")
idr = tfs.set_index("HGNC.symbol")["idr_50"] * tfs.set_index("HGNC.symbol")["n"]
idr = tfs.set_index("HGNC.symbol")["idr_50"] * tfs.set_index("HGNC.symbol")["length"]
idr = tfs.set_index("HGNC.symbol")["idr_50"]
transcriptome.var["idr_50"] = transcriptome.var["symbol"].str.upper().map(idr)

# %%
# motifscan.motifs["gene"] = transcriptome.var.reset_index().set_index("symbol").reindex(motifscan.motifs["symbol"])["gene"].values
# transcriptome.var.reindex(motifscan.motifs["gene"].dropna())["idr_50"]

# %%
n_links = links.groupby("symbol1").size() + links.groupby("symbol2").size()
if dataset_name in ["liver"]:
    n_links.index = n_links.index.str.capitalize()
# n_links = links.groupby("symbol1")["combined_score"].sum() + links.groupby("symbol2")["combined_score"].sum()
n_links.index =transcriptome.var.reset_index().set_index("symbol").reindex(n_links.index)["gene"]
n_links = n_links.dropna()
n_links = n_links[~pd.isnull(n_links.index)]
transcriptome.var["n_links"] = n_links

# %%
plotdata = enrichment_oi.reset_index().groupby("gene_peak").agg({"diff":"mean", "rel":"mean", "motif":"first"})
# plotdata["gene"] = plotdata["gene_peak"]
plotdata["idr_50"] = transcriptome.var.reindex(plotdata.index)["idr_50"].values
plotdata["idr_50_2"] = transcriptome.var.reindex(plotdata.index)["idr_50_2"].values
# plotdata["idr_4"] = transcriptome.var.reindex(plotdata.index)["idr_4"].values
plotdata["n_links"] = transcriptome.var.reindex(plotdata.index)["n_links"].values

# %%
symbols_oi = [
    "GATA1"
]
symbols_oi = []
genes_oi = transcriptome.gene_id(symbols_oi)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

dim = 1.5

# TF-TF interactions

panel, ax = fig.main.add(chd.grid.Panel((dim, dim)))

plotdata_oi = plotdata.dropna(subset=["n_links"]).copy()
plotdata_oi["x"] = np.log(plotdata_oi["n_links"])
plotdata_oi["y"] = np.log(plotdata_oi["rel"])
# plotdata_oi["y"] = (plotdata_oi["rel"])

ax.scatter(plotdata_oi["x"], plotdata_oi["y"], s=5, color="black", lw=0.0)
sns.regplot(x=plotdata_oi["x"], y=plotdata_oi["y"], ax=ax, scatter=False, color="black")

ax.axhline(0.0, color="grey", ls="--")
ax.set_xlabel("# of TF-TF interactions")
ax.set_ylabel("Ratio differential positions\nChromatinHD vs MACS2 summits")

ax.set_xticks(np.log([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]))
ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

# add cor
cor = np.corrcoef(plotdata_oi["x"], plotdata_oi["y"])
ax.text(0.05, 0.95, f"r = {cor[0, 1]:.2f}", transform=ax.transAxes, ha="left", va="top")

# ax.set_ylim(np.log(1/3), np.log(3.))

# annotate
plotdata_oi2 = plotdata_oi.loc[genes_oi]
ax.scatter(plotdata_oi2["x"], plotdata_oi2["y"], s=5, color="red", lw=0.0)
for motif, row in plotdata_oi2.iterrows():
    text = ax.annotate(
        xy=(row["x"], row["y"]),
        xytext=(0, -50),
        textcoords="offset points",
        text=transcriptome.symbol(motif),
        ha="center",
        va="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="-", color="red", shrinkA=0, shrinkB=0),
    )
    text.set_path_effects([mpl.patheffects.withStroke(linewidth=3, foreground="#FFFFFFCC")])

# IDR
panel, ax = fig.main.add_right(chd.grid.Panel((dim, dim)))

score = "idr_50_2"
# score = "idr_50"
plotdata_oi = plotdata.dropna(subset=[score]).copy()
# plotdata_oi["x"] = np.log(plotdata_oi[score])
plotdata_oi["x"] = (plotdata_oi[score])
plotdata_oi["y"] = np.log(plotdata_oi["rel"])
# plotdata_oi["y"] = plotdata_oi["rel"]

ax.scatter(plotdata_oi["x"], plotdata_oi["y"], s=5, color="black", lw=0.0)
sns.regplot(x=plotdata_oi["x"], y=plotdata_oi["y"], ax=ax, scatter=False, color="black")

# ax.set_ylim(np.log(1/3), np.log(3.))
ax.set_yticks([])
ax.axhline(0.0, color="grey", ls="--")
ax.set_xlabel("% IDR-50 amino acids")

# add cor
cor = np.corrcoef(plotdata_oi["x"], plotdata_oi["y"])
ax.text(0.05, 0.95, f"r = {cor[0, 1]:.2f}", transform=ax.transAxes, ha="left", va="top")

# annotate
plotdata_oi2 = plotdata_oi.loc[genes_oi]
ax.scatter(plotdata_oi2["x"], np.log(plotdata_oi2["rel"]), s=5, color="red", lw=0.0)
for motif, row in plotdata_oi2.iterrows():
    text = ax.annotate(
        xy=(row["x"], row["y"]),
        xytext=(0, -50),
        textcoords="offset points",
        text=transcriptome.symbol(motif),
        ha="center",
        va="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="-", color="red", shrinkA=0, shrinkB=0),
    )
    text.set_path_effects([mpl.patheffects.withStroke(linewidth=3, foreground="#FFFFFFCC")])

fig.plot()

# %%
# enrichment_oi.xs("KLF9.H12CORE.0.P.B", level = "motif")

# %%
plotdata.sort_values("rel").style.bar()

# %%
plotdata_oi = plotdata.dropna(subset=["n_links", "idr_50_2"]).copy()
plotdata_oi["log_n_links"] = np.log(plotdata_oi["n_links"])
plotdata_oi["merged"] = np.log(plotdata_oi["n_links"])

np.corrcoef(plotdata_oi["merged"], np.log(plotdata_oi["rel"]))

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

# %%
# np.exp(enrichment.loc[pd.MultiIndex.from_frame(motifclustermapping)]["log_odds"].mean() -  enrichment_peak.loc[pd.MultiIndex.from_frame(motifclustermapping)]["log_odds"].mean())

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

# score_cutoff = -np.inf
score_cutoff = 0.
# score_cutoff = 10
# score_cutoff = np.log2(2)

# %%
enrichment["expected"] = enrichment["contingency"].str[0].str[1] / enrichment["contingency"].str[0].str[0]
enrichment_peak["expected"] = enrichment_peak["contingency"].str[0].str[1] / enrichment_peak["contingency"].str[0].str[0]


# %%
def enrich_per_gene(motifclustermapping, diffexp, enrichment, slicescores, slicecounts, score_cutoff = -np.inf):
    founds = []
    for ct in tqdm.tqdm(motifclustermapping["cluster"].unique()):
        motifs_oi = motifclustermapping.query("cluster == @ct")["motif"]
        genes_oi = diffexp.loc[ct].query("logfoldchanges > @score_cutoff").index

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
# founds["diffexp"] = diffexp.T.unstack().loc[founds.index.droplevel("motif")].values
# founds_peak["diffexp"] = diffexp.T.unstack().loc[founds_peak.index.droplevel("motif")].values

founds["diffexp"] = diffexp["logfoldchanges"].loc[founds.index.droplevel("motif")].values
founds_peak["diffexp"] = diffexp["logfoldchanges"].loc[founds_peak.index.droplevel("motif")].values

# %%
# sc.pl.umap(transcriptome.adata, color = ["celltype", "ENSG00000102145"], layer = "magic", legend_loc = "on data")

# %%
pd.DataFrame({"chd":founds.groupby(["cluster", "motif"])["ratio"].mean(), "peak":founds_peak.groupby(["cluster", "motif"])["ratio"].mean()}).T.style.bar()

# %%
(founds["found"].sum() / founds["expected"].sum()) / (founds_peak["found"].sum() / founds_peak["expected"].sum())

# %%
# founds["detected"] = founds["found"] > 1
# founds_peak["detected"] = founds_peak["found"] > 1

founds["detected"] = founds["ratio"] > 0
founds_peak["detected"] = founds_peak["ratio"] > 0

# founds["detected"] = founds["ratio"] > 4
# founds_peak["detected"] = founds_peak["ratio"] > 4

# %%
founds.loc["Erythroblast"]["ratio"].mean(), founds_peak.loc["Erythroblast"]["ratio"].mean()

# %%
founds.loc["Erythroblast"]["detected"].mean(), founds_peak.loc["Erythroblast"]["detected"].mean()

# %%
founds["detected"].mean(), founds_peak["detected"].mean(), founds["detected"].mean()/founds_peak["detected"].mean()

# %%
founds.join(founds_peak, rsuffix = "_peak").query("detected & ~detected_peak").sort_values("diffexp", ascending = False)

# %%
scores = pd.DataFrame({
    "Peak differential":founds_peak.groupby("cluster")["found"].mean(),
    "ChromatinHD":founds.groupby("cluster")["found"].mean(),
})
scores = scores.reindex(clustering.cluster_info.sort_values("n_cells", ascending = False).index)

fig, ax = plt.subplots()
scores.plot.bar(ax = ax)
ax.set_title("% of Target - TF - Cell type links active", rotation = 0)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_xlabel("Cell type")
# ax.set_ylim(0, 1)

fig, ax = plt.subplots(figsize = (1, 2))
scores.mean().plot.bar()
# ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

# %%
scores = pd.DataFrame({
    "Peak differential":founds_peak.groupby(["cluster", "gene"])["detected"].sum().groupby("cluster").mean(),
    "ChromatinHD":founds.groupby(["cluster", "gene"])["detected"].sum().groupby("cluster").mean(),
})
scores = scores.reindex(clustering.cluster_info.sort_values("n_cells", ascending = False).index)

fig, ax = plt.subplots()
scores.plot.bar(ax = ax)
ax.set_title("Number of differential distinct motifs per gene", rotation = 0)
ax.set_xlabel("Cell type")

fig, ax = plt.subplots(figsize = (1, 2))
scores.mean().plot.bar()

# %%
scores = pd.DataFrame({
    "Peak differential":founds_peak.groupby(["cluster", "gene"])["found"].sum().groupby("cluster").mean(),
    "ChromatinHD":founds.groupby(["cluster", "gene"])["found"].sum().groupby("cluster").mean(),
})
scores = scores.reindex(clustering.cluster_info.sort_values("n_cells", ascending = False).index)

fig, ax = plt.subplots()
scores.plot.bar(ax = ax)
ax.set_title("Number of differentially accessible binding sites per gene", rotation = 0)
ax.set_xlabel("Cell type")

fig, ax = plt.subplots(figsize = (1, 2))
scores.mean().plot.bar()

# %%
scores = pd.DataFrame({
    "Peak differential":founds_peak.groupby(["cluster", "gene"])["detected"].any().groupby("cluster").mean(),
    "ChromatinHD":founds.groupby(["cluster", "gene"])["detected"].any().groupby("cluster").mean(),
})
scores = scores.reindex(clustering.cluster_info.sort_values("n_cells", ascending = False).index)

fig, ax = plt.subplots()
scores.plot.bar(ax = ax)
ax.set_title("% of genes for which a single link between a TF binding site can be found", rotation = 0)
ax.set_xlabel("Cell type")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

fig, ax = plt.subplots(figsize = (1, 2))
scores.mean().plot.bar()
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

# %%
diffexp["symbol"] = transcriptome.var["symbol"].reindex(diffexp.index.get_level_values("gene")).values

# %%
# diffexp.loc["MEP"].sort_values("score", ascending = False).head(50)

# %%
plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Erythroblast"].loc[diffexp.loc["Erythroblast"].head(200).index].sort_values("ratio_chd", ascending = False).join(transcriptome.var[["symbol"]])
# plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["MEP"].loc[diffexp.loc["MEP"].head(200).index].sort_values("ratio_chd", ascending = False).join(transcriptome.var[["symbol"]])
# plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Myeloid"].loc[diffexp.loc["Myeloid"].query("score > 5").index].sort_values("ratio_chd", ascending = False).join(transcriptome.var[["symbol"]])

fig, ax = plt.subplots()
ax.scatter(plotdata["ratio_chd"], plotdata["ratio_peak"])
ax.set_yscale("symlog")
ax.set_xscale("symlog")

# %%
myeloid_vs_erythroblast

# %%
# plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Erythroblast"].join(myeloid_vs_erythroblast.set_index("gene"))
# plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Myeloid"].join(myeloid_vs_erythroblast.set_index("gene"))
plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Myeloid"].xs(motifscan.select_motif("SPI1"), level = "motif").join(myeloid_vs_erythroblast.set_index("gene"))
# plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Erythroblast"].xs(motifscan.select_motif("GATA1"), level = "motif").join(myeloid_vs_erythroblast.set_index("gene"))

x = np.linspace(0, 10, 30)
fig, ax = plt.subplots()
n = []
y = []
for ratio_cutoff in x:
    plotdata_oi = plotdata.query("found_chd >= @ratio_cutoff")
    # plotdata_oi = plotdata.query("ratio_chd >= @ratio_cutoff")
    n.append(len(plotdata_oi))
    y.append(plotdata_oi["logfoldchanges"].mean())
ax.plot(n, y, marker = "o")

n = []
y = []
for ratio_cutoff in x:
    plotdata_oi = plotdata.query("found_peak >= @ratio_cutoff")
    # plotdata_oi = plotdata.query("ratio_peak >= @ratio_cutoff")
    n.append(len(plotdata_oi))
    y.append(plotdata_oi["logfoldchanges"].mean())
ax.plot(n, y, marker = "o")
ax.set_xscale("log")

ax.invert_xaxis()

# %%
plotdata["found_chd"].sum()/plotdata["expected_chd"].sum(), plotdata["found_peak"].sum()/plotdata["expected_peak"].sum()

# %%
# key_tfs = ["KLF1", "MEF2C", "GATA1", "CEBPD"]
key_tfs = [
    "KLF1",
    "GATA1",
    # "TAL1",
    # "OVOL"
    "GATA2",
    "NFIA",
    "ZBTB20",
    "MYB",
    "TFCP2",
    "SMAD5",
    "CPEB1",
    "SMAD3",
    "FOXO3",
    "ZNF449",
]
key_tfs_genes = transcriptome.gene_id(key_tfs)
# key_tfs_genes = np.unique(enrichment.loc["Myeloid"].sort_values("odds", ascending = False).query("score > 1")[["q_value", "odds", "pvals_adj", "logfoldchanges", "scores", "gene"]]["gene"].tolist() + transcriptome.gene_id(["CEBPA"]).tolist())
key_tfs_genes = enrichment.loc["Erythroblast"].sort_values("odds", ascending = False).query("score > 1")[["q_value", "odds", "pvals_adj", "logfoldchanges", "scores", "gene"]].head(50)["gene"]

# %%
plotdata["detected_chd"] = plotdata["ratio_chd"] > 1
plotdata["detected_peak"] = plotdata["ratio_peak"] > 1

# %%
plotdata.loc[transcriptome.gene_id("CEBPA")]

# %%
# plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Myeloid"].xs(motifscan.select_motif("SPI1"), level = "motif")
plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Erythroblast"].xs(motifscan.select_motif("GATA1"), level = "motif")
plotdata.loc[key_tfs_genes].join(transcriptome.var[["symbol"]]).sort_values("diffexp_chd", ascending = False).style.bar(subset = ["found_chd", "found_peak"])

# %%
fig, ax = plt.subplots()
plotdata = founds.join(founds_peak, lsuffix = "_chd", rsuffix = "_peak").loc["Myeloid"].join(myeloid_vs_erythroblast.set_index("gene"))
plotdata["ratio_chd2"] = np.log(np.clip(plotdata["ratio_chd"], 1., 10))
ax.scatter((plotdata["ratio_chd2"]), np.clip(plotdata["scores"], 20, -20))
sns.regplot(x = (plotdata["ratio_chd2"]), y = np.clip(plotdata["scores"], 20, -20), ax = ax, scatter = False, color = "red")

# %%
plotdata.join(transcriptome.var[["symbol"]])[["found_chd", "ratio_chd", "ratio_peak", "found_peak", "expected_peak", "symbol", "diffexp_chd"]].query("(ratio_chd > 2) & (ratio_peak < 1)").sort_values("diffexp_chd", ascending = False).head(50).style.bar()

# %%
# ERYTHRO
gene_oi = region_id = transcriptome.gene_id("SLC14A1") # WOW
gene_oi = region_id = transcriptome.gene_id("GALNT10") # Noice
gene_oi = region_id = transcriptome.gene_id("PTMA")
gene_oi = region_id = transcriptome.gene_id("PDCD4") # Yep in peak
gene_oi = region_id = transcriptome.gene_id("ABCC4") # Yep
gene_oi = region_id = transcriptome.gene_id("KLF1") # WOW
gene_oi = region_id = transcriptome.gene_id("TFRC") # Yep
gene_oi = region_id = transcriptome.gene_id("HMGB1") # meh
gene_oi = region_id = transcriptome.gene_id("KEL") # Nice
gene_oi = region_id = transcriptome.gene_id("IL9R") # Yep
gene_oi = region_id = transcriptome.gene_id("ZNF385D") # Yep subtle
gene_oi = region_id = transcriptome.gene_id("RIPOR3") # Yep
gene_oi = region_id = transcriptome.gene_id("MEF2C") # WOW
gene_oi = region_id = transcriptome.gene_id("ICAM4") # Yep
gene_oi = region_id = transcriptome.gene_id("EPSTI1") # Yep
gene_oi = region_id = transcriptome.gene_id("KLF3") # Yep
gene_oi = region_id = transcriptome.gene_id("NFIA") # Yep
gene_oi = region_id = transcriptome.gene_id("GATA2") # Cool
gene_oi = region_id = transcriptome.gene_id("FOXO3") # In peak
gene_oi = region_id = transcriptome.gene_id("NFATC2") # Hmmm
gene_oi = region_id = transcriptome.gene_id("SLC26A2") # Hmmm
gene_oi = region_id = transcriptome.gene_id("CD82") # Yep
gene_oi = region_id = transcriptome.gene_id("IRF6") # Noice

# MYELOID
# gene_oi = region_id = transcriptome.gene_id("CEBPA") # Wow
# gene_oi = region_id = transcriptome.gene_id("CEBPD") # Wow
# gene_oi = region_id = transcriptome.gene_id("AFF3") # Meh
# gene_oi = region_id = transcriptome.gene_id("DPYD") # Yep
# gene_oi = region_id = transcriptome.gene_id("FCER1G") # Yep
# gene_oi = region_id = transcriptome.gene_id("ANXA2") # Yep
# gene_oi = region_id = transcriptome.gene_id("CD44") # Meh
# gene_oi = region_id = transcriptome.gene_id("OSCAR") # Meh
# gene_oi = region_id = transcriptome.gene_id("RASSF2") # Meh
# gene_oi = region_id = transcriptome.gene_id("TNFRSF1B") # Meh

regions = regionpositional.select_regions(region_id, prob_cutoff = np.exp(-1.5))

# gene_oi = region_id = transcriptome.gene_id("CPEB1")
# window = [-100000, 100000]

# gene_oi = region_id = transcriptome.gene_id("GATA1")
# regions = pd.DataFrame([
#     [-10000, 20000]
# ], columns = ["start", "end"])

# gene_oi = region_id = transcriptome.gene_id("KLF1")
# regions = pd.DataFrame([
#     [-10000, 10000]
# ], columns = ["start", "end"])

# gene_oi = region_id = transcriptome.gene_id("CCL4")
# window = [-1000, 1000]

# gene_oi = region_id = transcriptome.gene_id("QKI")
# window = [-10000, 10000]

# gene_oi = region_id = transcriptome.gene_id("APRT")
# gene_oi = region_id = "ENSG00000196247"
# window = [-100000, 100000]

founds.join(founds_peak, rsuffix = "_peak").query("gene == @gene_oi").groupby(["cluster", "gene"])[["found", "found_peak"]].sum()

# %%
founds.join(founds_peak, rsuffix = "_peak").query("gene == @gene_oi").sort_values("found", ascending = False)


# %%
def select_motif(str):
    # return motifscan.motifs.loc[motifscan.motifs.index.str.contains(str)].sort_values("quality").index[0]
    return motifscan.motifs.loc[motifscan.motifs.index.str.contains(str)].index[0]
def select_motifs(str):
    # return motifscan.motifs.loc[motifscan.motifs.index.str.contains(str)].sort_values("quality").index[0]
    return motifscan.motifs.loc[motifscan.motifs.index.str.contains(str)].index
motifs_oi = pd.DataFrame([
    [select_motif("GATA1"), "Gata1"],
    [select_motif("TAL1.H12CORE.1.P.B"), "Tal1"],
    [select_motif("STA5"), "Stat5"],
    [select_motif("SPI1"), "Spi"],
], columns = ["motif", "symbol"]).set_index("motif")

# motifs_oi = pd.DataFrame({"motif":motifclustermapping.loc[motifclustermapping["cluster"] == "Myeloid", "motif"]}).reset_index().set_index("motif")
# motifs_oi = pd.DataFrame({"motif":motifclustermapping.loc[motifclustermapping["cluster"] == "Erythroblast", "motif"]}).reset_index().set_index("motif")
# motifs_oi = pd.DataFrame({"motif":motifclustermapping.loc[motifclustermapping["cluster"] == diffexp.xs(gene_oi, level = 'gene')["score"].idxmax(), "motif"]}).reset_index().set_index("motif")

# %%
import chromatinhd.data.associations
import chromatinhd.data.associations.plot

# motifscan_name = "gwas_hema"
motifscan_name = "gwas_hema_main"
# motifscan_name = "gwas_immune_main"
# motifscan_name = "gtex_immune"
associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
symbol = transcriptome.var.loc[region_id, "symbol"]

breaking = chd.grid.broken.Breaking(regions, 0.05, resolution = 2000)

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

clusters_oi = ["HSPC", "MEP", "Erythrocyte precursors", "Erythroblast", "GMP", "Myeloid"]; relative_to = ["HSPC"]
# clusters_oi = ["HSPC", "GMP", "Myeloid"]; relative_to = ["HSPC"]
# clusters_oi = clustering.cluster_info.index; relative_to = None
cluster_info = clustering.cluster_info.loc[clusters_oi]

region = fragments.regions.coordinates.loc[region_id]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking,
    genome="GRCh38",
    label_positions=True,
)
fig.main.add_under(panel_genes)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region_id, regionpositional, breaking, panel_height=0.4, cluster_info=cluster_info, label_accessibility=False, relative_to = relative_to
)
fig.main.add_under(panel_differential)

if region_id in transcriptome.var.index:
    panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
        transcriptome,
        clustering,
        region_id,
        cluster_info=cluster_info,
        panel_height=0.4,
        show_n_cells=False,
    )
    fig.main.add_right(panel_expression, panel_differential)

panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(motifscan, region_id, motifs_oi, breaking)
fig.main.add_under(panel_motifs)

panel_association = chd.data.associations.plot.AssociationsBroken(associations, region_id, breaking, show_ld = False)
fig.main.add_under(panel_association)

peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name)
peakcallers = peakcallers.loc[["macs2_summits"]]
panel_peaks = chd.data.peakcounts.plot.PeaksBroken.from_bed(
    region, peakcallers, breaking
)
fig.main.add_under(panel_peaks)

fig.plot()

# %%
files_all = []
for folder in ["k562_tf_chipseq_bw", "erythroblast_tf_chipseq_bw"]:
    bed_folder = chd.get_output() / "bed" / folder
    files = pd.read_csv(bed_folder / "files.csv", index_col = 0)
    files["path"] = bed_folder / files["filename"]

    files["n_complaints"] = files["audit_not_compliant"].map(lambda x: x.count(",") if isinstance(x, str) else 1)
    files["n_replicated"] = files["technical_replicate(s)"].map(lambda x: x.count(",")+1 if isinstance(x, str) else 1)
    files["sorter"] = files["n_replicated"]/files["n_complaints"]
    files = files.sort_values("sorter", ascending = False)
    files_all.append(files)
files = pd.concat(files_all)


design = pd.DataFrame([
    ["GATA1\nErythroblast", [motifscan.select_motif("GATA1")], files.query("experiment_target == 'GATA1-human'")["path"].iloc[0]],
    ["TAL1\nK562", [motifscan.select_motif("TAL1")], files.query("experiment_target == 'TAL1-human'")["path"].iloc[0]]

], columns = ["label", "motifs", "bigwig"])
assert all([path.exists() for path in design["bigwig"]])

# %%
import chromatinhd.data.peakcounts

# %%
# fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

panel = chd.grid.Grid()
fig.main.add_under(panel)

self = panel
panel_height = 0.4

region = fragments.regions.coordinates.loc[region_id]
regions_uncentered = chd.data.peakcounts.plot.uncenter_peaks(breaking.regions, region)

import pyBigWig

for _, setting in design.iterrows():
    broken = self.add_under(
        chd.grid.Broken(breaking, height=panel_height, margin_height=0.0, padding_height=0.01), padding=0
    )

    motifs_oi = pd.DataFrame({"motif":setting["motifs"]}).set_index("motif")
    motifs_oi, group_info, motifdata = chd.data.motifscan.plot._process_grouped_motifs(region_id, motifs_oi, motifscan)

    panel, ax = broken[0, 0]
    ax.set_ylabel(setting["label"], rotation = 0, ha = "right", va = "center")

    bw = pyBigWig.open(str(setting["bigwig"]))
    for (subregion_id, subregion_info), (_, subregion_info_uncentered), (panel, ax) in zip(
        breaking.regions.iterrows(), regions_uncentered.iterrows(), broken
    ):
        # ChIP-seq
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
        ax.plot(plotdata_chip["position"], plotdata_chip["value"], color="black", lw=1)

        # Motifs
        plotdata_motifs = motifdata.loc[
            (motifdata["position"] >= subregion_info["start"])
            & (motifdata["position"] <= subregion_info["end"])
        ].copy()
        plotdata_motifs["significant"] = (plotdata_chip.set_index("position").reindex(plotdata_motifs["position"])["value"] > 2.0).values
        
        ax.scatter(
            plotdata_motifs["position"],
            [1] * len(plotdata_motifs),
            transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
            marker="v",
            color=["orange" if x else "grey" for x in plotdata_motifs["significant"]],
            alpha=1,
            s=200,
            zorder=20,
            lw = 1,
            edgecolors = "white",
        )

        ax.set_ylim(0, 10)
        ax.set_xlim(subregion_info["start"], subregion_info["end"])
        ax.axhline(1.5, dashes = (2, 2), color = "grey", lw = 1)

fig.plot()
fig

# %%
plotdata_chip.set_index("position").reindex(plotdata_motifs["position"])["value"]

# %%
# !ls {chd.get_output() / "bed" / "k562_tf_chipseq_bw"}

# %%
files

# %% [markdown]
# ### VS

# %%
# genes_oi = diffexp.loc["Lymphoma"].sort_values("score", ascending = False).head(1000).index
genes_oi = transcriptome.var.index

# %%
# motif_1 = select_motif("GATA1")
# motif_2 = select_motif("SPI1")

motif_1 = motifscan.select_motif("PAX5")
motif_2 = motifscan.select_motif(symbol = "POU2F2")

# %%
# founds_oi_1 = founds.loc["Erythroblast"].loc[genes_oi].xs(motif_1, level = "motif")
# founds_oi_2 = founds.loc["Myeloid"].loc[genes_oi].xs(motif_2, level = "motif")

founds_oi_1 = founds.loc["Lymphoma"].loc[genes_oi].xs(motif_1, level = "motif")
founds_oi_2 = founds.loc["Lymphoma"].loc[genes_oi].xs(motif_2, level = "motif")

founds_oi = founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2")

# %%
founds.groupby("motif")["detected"].mean()

# %%
founds_oi.groupby(["detected_1", "detected_2"]).size().unstack()

# %%
sns.heatmap(founds_oi.groupby(["detected_1", "detected_2"]).size().unstack(), annot = True, vmin = 0)

# %%
import gseapy

# %%
gsets = founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2").reset_index().groupby(["detected_1", "detected_2"])["gene"].apply(list).to_frame("genes").reset_index()
gsets["n"] = gsets["genes"].apply(len)
gsets

# %%
genesets = gseapy.get_library(name='GO_Biological_Process_2023', organism='Human')
genesets = {name:transcriptome.gene_id(genes, optional = True).tolist() for name, genes in genesets.items()}

gset_info = pd.DataFrame({
    "gset":pd.Series(genesets.keys()),
    "label":pd.Series(genesets.keys()).str.split("GO:").str[0].str[:-2],
    "go":"GO:" + pd.Series(genesets.keys()).str.split("GO:").str[1].str[:-1],
    "n":[len(gset) for gset in genesets.values()],
}).sort_values("n", ascending = False)

# %%
background = [g for gs in gsets["genes"] for g in gs]
genes_oi = diffexp.loc["Lymphoma"].sort_values("score", ascending = False).query("score > 5").index.tolist()

# genes_oi = [g for gs in gsets.loc[(gsets["found_2"] == 0) & (gsets["found_1"] == 0)]["genes"] for g in gs]
# genes_oi = [g for gs in gsets.loc[(gsets["found_2"] > 0) & (gsets["found_1"] == 0)]["genes"] for g in gs]
# genes_oi = [g for gs in gsets.loc[(gsets["found_2"] == 0) & (gsets["found_1"] > 0)]["genes"] for g in gs]
# genes_oi = [g for gs in gsets.loc[(gsets["found_1"] > 2)]["genes"] for g in gs]
# genes_oi = [g for gs in gsets.loc[(gsets["detected_1"] > 0)]["genes"] for g in gs]
# genes_oi = [g for gs in gsets.loc[(gsets["detected_2"] > 0)]["genes"] for g in gs]
# background = [g for gs in gsets.loc[(gsets["detected_2"] > 0) | (gsets["detected_1"] > 0)]["genes"] for g in gs]
print(len(background), len(genes_oi))

# %%
goenrichment = gseapy.enrich(genes_oi, background = background, gene_sets = genesets, top_term = 100000)
goenrichment = goenrichment.res2d.sort_values("Adjusted P-value").set_index("Term")
goenrichment["n_found"] = goenrichment["Overlap"].str.split("/").str[0].astype(int)
goenrichment["q_value"] = chd.utils.fdr(goenrichment.query("n_found > 1")["P-value"])

# %%
goenrichment.head(20)


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
import scipy.stats

# %%
gset_info.loc[gset_info["label"].str.contains("Regulation Of B Cell Proliferation", case = False)].sort_values("n", ascending = False).head(10)

# %%
founds_oi = founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2")

# %%
gset = genesets[find_go("Regulation Of B Cell Proliferation")]
founds_oi.reindex(genes_oi).reindex(gset).dropna()[["found_1", "found_2"]].sum() / founds_oi[["found_1", "found_2"]].sum()

# %%
founds_oi.reindex(genes_oi).reindex(gset).dropna()[["found_1", "found_2"]]

# %%
gset = genesets[find_go("Antigen Receptor-Mediated Signaling Pathway")]
founds_oi.reindex(gset)[["found_1", "found_2"]].sum() / founds_oi[["found_1", "found_2"]].sum()

# %%
founds_oi_1["found"].sum(), founds_oi_2["found"].sum()

# %%
# founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2").reindex(genesets["Apoptotic Process (GO:0006915)"]).sort_values("found_1").dropna()
founds_oi_1.join(founds_oi_2, lsuffix = "_1", rsuffix = "_2").reindex(genesets[find_go("Positive Regulation Of B Cell Proliferation")]).sort_values("found_1").dropna()

# %%
enrichment.loc["B Cell Receptor Signaling Pathway (GO:0050853)"]

# %% [markdown]
# ### Motif examples

# %%
# design_name = "GATA1_Erythroblast"
# selection_cluster_ids = ["Erythroblast"]
# cluster_ids = ["Erythroblast"]
# relative_to = "HSPC"
# motif_ids = [motifscan.select_motif("GATA1")]
# region_selection = "chd"
# n = 9
# label_y = True

# design_name = "GATA1_Erythroblast_full"
# n = 19
# label_y = True

# design_name = "GATA2_GranPrec"
# selection_cluster_ids = ["Granulocyte precursor"]
# cluster_ids = ["Granulocyte precursor"]
# relative_to = "HSPC"
# motif_ids = ["GATA2.H12CORE.0.PSM.A"]
# region_selection = "chd"
# n = 6
# label_y = False

# design_name = "GATA2_GranPrec_full"
# n = 20
# label_y = True

# design_name = "HSPC_ERG"
# selection_cluster_ids = ["HSPC"]
# cluster_ids = ["HSPC"]
# relative_to = "GMP"
# motif_ids = [motifscan.select_motif("ERG")]
# region_selection = "unique_peak"
# n = 6
# label_y = False

# design_name = "HSPC_ERG_full"
# n = 20
# label_y = True

design_name = "HSC_LHX2"
selection_cluster_ids = ["Stellate", "LSEC", "KC", "Cholangiocyte"]
cluster_ids = selection_cluster_ids
relative_to = "LSEC"
motif_ids = [motifscan.select_motif("LHX2")]
region_selection = "all"
n = 6
# label_y = False

# design_name = "custom"
# n = 99

# %%
# diffexp.loc["Erythroblast"].join(transcriptome.var[["symbol"]]).head(20)#.loc[transcriptome.var.index[transcriptome.var.symbol.str.contains("IRF")]]

# %%
# select which genes to use

if design_name in ["GATA1_Erythroblast", "GATA1_Erythroblast_full"]:
    regions_oi = fragments.regions.coordinates.loc[
        transcriptome.gene_id(
            [
                "GATA1",
                "KLF1",
                "H1FX",
                "CALR",
                "KLF3",
                # "MEF2C",
                # "IRF6",
                "CD82",
                # "GATA2",
                "HBB",
                "IKZF2",
                # "CSF1",
                # "IL9R",
                # "LEF1",
                # "SLC14A1",
                # "ZNF385D",
                "GALNT10",
                "ITGA2B",
                "ITGA4",
                "APOC1",
                # "CSF2RB",
            ]
            # ["CEBPA", "CEBPD", "DPYD", "FCER1G", "ANXA2", "TNFRSF1B", "LYZ", "AFF3", "CST3", "HDAC9", "GRN", "SAMHD1", "CD74"]
        )
    ]
elif design_name in ["GATA2_GranPrec"]:
    regions_oi = fragments.regions.coordinates.loc[
        transcriptome.gene_id(
            [
                "ENPP3",
                "ITGB8",
                "GRAP2",
                "HPGD",
                "Z82206.1",
                "HES1",
            ]
        )
    ]
elif design_name in ["HSC_LHX2"]:
    regions_oi = fragments.regions.coordinates.loc[transcriptome.gene_id(["Lhx2"])]
else:
    regions_oi = fragments.regions.coordinates.loc[
        diffexp.loc[selection_cluster_ids]
        .groupby("gene")
        .mean(numeric_only=True)
        .sort_values("logfoldchanges", ascending=False)
        .head(1000)
        .index
    ]
    regions_oi = pd.concat([
        fragments.regions.coordinates.loc[[transcriptome.gene_id("HES1")]],
        regions_oi
    ])

# %%
import pybedtools
x = slicescores.query("cluster in @selection_cluster_ids")[["region_ix", "start", "end"]]
x["start"] = x["start"] - fragments.regions.window[0]
x["end"] = x["end"] - fragments.regions.window[0]
x = pybedtools.BedTool.from_dataframe(x).sort().merge()

y = slicescores_peak.query("cluster in @selection_cluster_ids")[["region_ix", "start", "end"]]
y["start"] = y["start"] - fragments.regions.window[0]
y["end"] = y["end"] - fragments.regions.window[0]
y = pybedtools.BedTool.from_dataframe(y).sort().merge()

z = y.subtract(x)
slicescores_unique = z.to_dataframe()
slicescores_unique["region"] = fragments.var.index[slicescores_unique["chrom"]]
slicescores_unique["start"] = slicescores_unique["start"] + fragments.regions.window[0]
slicescores_unique["end"] = slicescores_unique["end"] + fragments.regions.window[0]
slicescores_unique["cluster"] = selection_cluster_ids[0]

# count
slicescores_unique["slice"] = pd.Categorical(slicescores_unique["region"].astype(str) + ":" + slicescores_unique["start"].astype(str) + "-" + slicescores_unique["end"].astype(str))
slices_unique = slicescores_unique.groupby("slice")[["region", "start", "end"]].first()
slicecounts_unique = motifscan.count_slices(slices_unique)

# %%
# determine slices of interest, i.e. those with a motif
padding = 400

# get regions from CHD
if region_selection == "all":
    slicescores_oi = slicescores.query("cluster in @selection_cluster_ids").query("region in @regions_oi.index")
    slicescores_oi["region"] = pd.Categorical(slicescores_oi["region"], categories = regions_oi.index.unique())
    slicescores_oi = slicescores_oi.sort_values("region")
    slicescores_oi["start"] = slicescores_oi["start"] - padding
    slicescores_oi["end"] = slicescores_oi["end"] + padding

elif region_selection == "chd":
    slicescores_selected = slicescores.query("cluster in @selection_cluster_ids").query("region in @regions_oi.index")
    slicescores_oi = slicescores_selected.loc[(slicecounts.loc[slicescores_selected["slice"], motif_ids] > 0).any(axis = 1).values].copy()
    slicescores_oi["region"] = pd.Categorical(slicescores_oi["region"], categories = regions_oi.index.unique())
    slicescores_oi = slicescores_oi.sort_values("region")
    slicescores_oi["start"] = slicescores_oi["start"] - padding
    slicescores_oi["end"] = slicescores_oi["end"] + padding

elif region_selection == "peak":
    # get regions from PEAK
    slicescores_selected = slicescores_peak.query("cluster in @selection_cluster_ids").query("region in @regions_oi.index")
    slicescores_oi = slicescores_selected.loc[(slicecounts_peak.loc[slicescores_selected["slice"], motif_ids] > 0).any(axis = 1).values].copy().iloc[:20]
    slicescores_oi["region"] = pd.Categorical(slicescores_oi["region"], categories = regions_oi.index.unique())
    slicescores_oi = slicescores_oi.sort_values("region")
    slicescores_oi["start"] = slicescores_oi["start"] - padding
    slicescores_oi["end"] = slicescores_oi["end"] + padding

elif region_selection == "unique_peak":
    # get regions UNIQUE IN PEAK
    slicescores_selected = slicescores_unique.query("cluster in @selection_cluster_ids").query("region in @regions_oi.index")
    slicescores_oi = slicescores_selected.loc[(slicecounts_unique.loc[slicescores_selected["slice"], motif_ids] > 0).any(axis = 1).values].copy().iloc[:20]
    slicescores_oi["region"] = pd.Categorical(slicescores_oi["region"], categories = regions_oi.index.unique())
    slicescores_oi = slicescores_oi.sort_values("region")
    slicescores_oi["start"] = slicescores_oi["start"] - padding
    slicescores_oi["end"] = slicescores_oi["end"] + padding

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
slicescores_oi = slicescores_oi.iloc[:n]

# %%
breaking = chd.grid.broken.Breaking(slicescores_oi, 0.05, resolution = 3000)
# breaking = chd.grid.broken.Breaking(slicescores_oi, 0.05, resolution = 7500)

# %%
# preload peakcallers
peakcallers = chdm.plotting.peaks.get_peakcallers(chd.get_output() / "peaks" / dataset_name, add_rolling = True)
# peakcallers = peakcallers.loc[["macs2_summits"]]
peakcallers = peakcallers.loc[[
    "macs2_summits", 
    "macs2_leiden_0.1_merged",
    "encode_screen", 
    # "rolling_500"
]]
peakcallers["color"] = chdm.peakcallers.loc[peakcallers.index, "color"]

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
# load peak diffexps
slices_peakcallers = {}
for peakcaller in tqdm.tqdm(peakcallers.index):
    scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / "t-test" / "scoring" / "regionpositional"
    # scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / "t-test-foldchange" / "scoring" / "regionpositional"
    differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

    differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
    differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
    differential_slices_peak.window = fragments.regions.window
    
    # match # of differential within each cluster
    slicescores_peak_full = differential_slices_peak.get_slice_scores(regions = fragments.regions, clustering = clustering)
    slicescores_peak_ = []
    for cluster in selection_cluster_ids:
        peakscores_cluster = slicescores_peak_full.query("cluster == @cluster")
        peakscores_cluster = peakscores_cluster.sort_values("score", ascending=False)
        # peakscores_cluster = peakscores_cluster.sort_values("logfoldchanges", ascending=False)
        n_desired_positions_cluster = n_desired_positions[cluster]

        # peakscores_cluster["cumulative_length"] = peakscores_cluster["length"].cumsum() # at the latest as large
        peakscores_cluster["cumulative_length"] = np.pad(peakscores_cluster["length"].cumsum()[:-1], (1, 0)) # at least as large

        peakscores_cluster = peakscores_cluster.query("cumulative_length <= @n_desired_positions_cluster")
        slicescores_peak_.append(peakscores_cluster)
    slicescores_peak_ = pd.concat(slicescores_peak_)
    slicescores_peak_["slice"] = pd.Categorical(slicescores_peak_["region"].astype(str) + ":" + slicescores_peak_["start"].astype(str) + "-" + slicescores_peak_["end"].astype(str))
    slices_peak = slicescores_peak_.groupby("slice")[["region", "start", "end"]].first()
    slices_peakcallers[peakcaller] = slices_peak

# %%
remap_bed_folder = chd.get_output() / "bed" / "remap2022"
remap_bed_folder.mkdir(exist_ok = True)

# save peaks
files = [
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE60477.ERG.TSU-1621MT_ATRA.bed.gz",
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE49091.ERG.Jurkat.bed.gz",
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE23730.ERG.SKNO-1.bed.gz",
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE23730.ERG.CD34_NR29.bed.gz",
    "https://www.encodeproject.org/files/ENCFF910CYH/@@download/ENCFF910CYH.bed.gz",
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/GSE59087.NFE2.ProEs.bed.gz",
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/ENCSR000FCC.NFE2.K-562.bed.gz",
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/DATASET/ENCSR000DKA.GATA2.K-562.bed.gz",
]
for source_file in files:
    file = (remap_bed_folder / source_file.split("/")[-1]).with_suffix(".bed")
    if not file.exists():
        # !wget {source_file} -O {file.with_suffix(".bed.gz")}
        # !gunzip {file.with_suffix(".bed.gz")}

# %%
# load chipseq data and determine chip-seq+motif design
import pyBigWig
files_all = []
for folder in ["k562_tf_chipseq_bw", "erythroblast_tf_chipseq_bw", "hl60_tf_chipseq_bw", "gm1282_tf_chipseq_bw", "wtc11_tf_chipseq_bw"]:
    bed_folder = chd.get_output() / "bed" / folder
    files = pd.read_csv(bed_folder / "files.csv", index_col=0)
    files["path"] = bed_folder / files["filename"]

    files["n_complaints"] = files["audit_not_compliant"].map(lambda x: x.count(",")+1 if isinstance(x, str) else 0.1)
    files["n_replicated"] = files["technical_replicate(s)"].map(lambda x: x.count(",") + 1 if isinstance(x, str) else 1)
    files["sorter"] = files["n_replicated"] / files["n_complaints"]
    files = files.sort_values("sorter", ascending=False)
    files_all.append(files)
files = pd.concat(files_all)


if design_name == "GATA1_Erythroblast":
    design = pd.DataFrame(
        [
            [
                "GATA1 ChIP-seq\n(Erythroblast)",
                [motifscan.select_motif("GATA1")],
                files.query("(experiment_target == 'GATA1-human') & (biosample_term_name == 'erythroblast')")["path"].iloc[2],
                None,
            ],
        ],
        columns=["label", "motifs", "bigwig", "peaks"],
    )
elif design_name == "GATA1_Erythroblast_full":
    design = pd.DataFrame(
        [
            [
                "GATA1 ChIP-seq\n(Erythroblast)",
                [motifscan.select_motif("GATA1")],
                files.query("(experiment_target == 'GATA1-human') & (biosample_term_name == 'erythroblast')")["path"].iloc[0],
                None,
            ],
            [
                "TAL1 ChIP-seq\n(K562)",
                ["TAL1.H12CORE.1.P.B", motifscan.select_motif("TAL1")],
                files.query("experiment_target == 'TAL1-human'")["path"].iloc[0],
            ],
        ],
        columns=["label", "motifs", "bigwig", "peaks"],
    )
elif design_name in ["GATA2_GranPrec", "GATA2_GranPrec_full"]:
    design = pd.DataFrame(
        [
            [
                "GATA2 ChIP-seq\n(K562)",
                [*motif_ids],
                files.query("(experiment_target == 'GATA2-human')")["path"].iloc[0],
                None,
            ],
        ],
        columns=["label", "motifs", "bigwig", "peaks"],
    )
elif design_name in ["HSPC_ERG", "HSPC_ERG_full"]:
    design = pd.DataFrame(
        [
            [
                "ERG ChIP-seq\n(TSU-1621MT)",
                [motifscan.select_motif("ERG")],
                None,
                remap_bed_folder / pathlib.Path("GSE60477.ERG.TSU-1621MT_ATRA.bed.bed"),
            ],
        ],
        columns=["label", "motifs", "bigwig", "peaks"],
    )
else:
    design = pd.DataFrame(
        [
            # [
            #     "ERG ChIP-seq\n(TSU-1621MT)",
            #     [motifscan.select_motif("ERG")],
            #     None,
            #     pathlib.Path("GSE60477.ERG.TSU-1621MT_ATRA.bed.gz"),
            # ],
        ],
        columns=["label", "motifs", "bigwig", "peaks"],
    )
assert all([path.exists() if path is not None else True for path in design["bigwig"]])

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
# determine whether motif is detected by peakcaller
peakcaller_region_motifdata = {}
for region_id in slicescores_oi["region"].unique():
    for peakcaller in peakcallers.index:
        peakcaller_region_motifdata[(peakcaller, region_id)] = {}

        # gather different motifdatas
        motifdatas = []
        for setting_name, setting in design.iterrows():
            motifs_oi, group_info, motifdata = setting_region_motifdata[(setting_name, region_id)]
            motifdatas.append(motifdata)
        motifdata = pd.concat(motifdatas).copy()

        # determine whether motif is detected by peakcaller
        slices_peak = slices_peakcallers[peakcaller].query("region == @region_id")

        motifdata["captured"] = ((slices_peak["start"].values[None, :] < motifdata["position"].values[:, None]) & (slices_peak["end"].values[None, :] > motifdata["position"].values[:, None])).any(1)

        peakcaller_region_motifdata[(peakcaller, region_id)] = motifdata

# %%
# determine which motif is detected by chd
slices["region"] = transcriptome.var.iloc[slices["region_ix"]].index
for region_id in slicescores_oi["region"].unique():
    peakcaller_region_motifdata[("chd", region_id)] = {}

    # gather different motifdatas
    motifdatas = []
    for setting_name, setting in design.iterrows():
        motifs_oi, group_info, motifdata = setting_region_motifdata[(setting_name, region_id)]
        motifdatas.append(motifdata)
    motifdata = pd.concat(motifdatas).copy()

    # determine whether motif is detected by peakcaller
    slices_ = slices.query("region == @region_id")

    motifdata["captured"] = ((slices_["start"].values[None, :] < motifdata["position"].values[:, None]) & (slices_["end"].values[None, :] > motifdata["position"].values[:, None])).any(1)

    peakcaller_region_motifdata[("chd", region_id)] = motifdata

# %%
# load crispri data
folder = chd.get_output() / "data" / "crispri" / "fulco_2019"
data = pd.read_csv(folder / "data.tsv", sep="\t", index_col = 0)

binwidth = 50
data["bin"] = data["Gene"] + data["chrom"] + (data["start"] // binwidth).astype(str)
data_binned = data.groupby("bin").agg({"HS_LS_logratio":"mean", "Gene":"first", "chrom":"first", "start":"first"})
data_binned["start"] = (data_binned["start"]//binwidth) * binwidth
data_binned["end"] = data_binned["start"] + binwidth

region_crisprdata = {}
for region_id in slicescores_oi["region"].unique():
    region = fragments.regions.coordinates.loc[region_id]
    symbol = transcriptome.var.loc[region_id, "symbol"]
    data_binned_region = data_binned.query("Gene == @symbol").query("chrom == @region.chrom").query("start >= @region.start").query("end <= @region.end")
    data_binned_region = chd.data.peakcounts.plot.center_peaks(data_binned_region, region)
    if len(data_binned_region) > 0:
        region_crisprdata[region_id] = data_binned_region

# %%
from chromatinhd.plot import format_distance
import textwrap

fig = chd.grid.Figure(chd.grid.BrokenGrid(breaking, padding_height = 0.03))

# cluster_info = clustering.cluster_info
cluster_info = clustering.cluster_info.loc[cluster_ids]

norm_atac_diff = mpl.colors.Normalize(np.log(1 / 8), np.log(8.0), clip=True)

for subregion_ix, (_, subregion_info), grid, width in zip(
    range(len(slicescores_oi)), slicescores_oi.iterrows(), fig.main, fig.main.panel_widths
):
    # add upper labelling panel
    panel_labeling = chd.grid.Panel((width, 0.1))
    panel_labeling.ax.set_xlim(subregion_info["start"], subregion_info["end"])
    panel_labeling.ax.axis("off")
    grid.add_under(panel_labeling)
    if subregion_ix == 0:  # we only use the left-most panel for labelling
        ax_labeling = panel_labeling.ax

    region = fragments.regions.coordinates.loc[subregion_info["region"]]
    window = subregion_info[["start", "end"]]
    panel_differential = chd.models.diff.plot.Differential.from_regionpositional(
        subregion_info["region"],
        regionpositional,
        width=width,
        window=window,
        cluster_info=cluster_info,
        panel_height=0.4,
        relative_to=relative_to,
        label_accessibility=False,
        label_cluster=False,
        show_tss=False,
        ymax = 10.,
        norm_atac_diff = norm_atac_diff,
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

    # chd capture
    panel, ax = grid.add_under(chd.grid.Panel((width, 0.1)))
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xlim(subregion_info["start"], subregion_info["end"])
    motifdata = peakcaller_region_motifdata[("chd", subregion_info["region"])]
    motifdata = motifdata.loc[
        (motifdata["position"] >= subregion_info["start"]) & (motifdata["position"] <= subregion_info["end"])
    ]

    motifdata_captured = motifdata.loc[motifdata["captured"]]
    motifdata_notcaptured = motifdata.loc[~motifdata["captured"]]
    ax.scatter(
        motifdata_notcaptured["position"],
        [0.5] * len(motifdata_notcaptured),
        marker="x",
        c="#FF4136",
    )
    ax.scatter(
        motifdata_captured["position"],
        [0.5] * len(motifdata_captured),
        marker="v",
        c="#2ECC40",
    )
    ax.set_xticks([])
    if subregion_ix == 0:
        ax.set_yticks([0.5])
        ax.set_yticklabels(["ChromatinHD"], fontsize=8, rotation=0, va="center", ha="right", color = "#0074D9")
        ax.tick_params(axis="y", which="major", length=0)
    else:
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
    ax.axhline(0., color="#DDD", zorder=10, lw=0.5)

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
    if subregion_ix == 0:
        for tick, peakcaller in zip(panel_peaks.ax.get_yticklabels(), peakcallers.index):
            tick.set_color(peakcallers.loc[peakcaller, "color"])
    grid.add_under(panel_peaks)
    sns.despine(ax=panel_peaks.ax, left=True)

    # peak capture
    ax = panel_peaks.ax
    for y, peakcaller in enumerate(peakcallers.index):
        motifdata = peakcaller_region_motifdata[(peakcaller, subregion_info["region"])]
        motifdata = motifdata.loc[
            (motifdata["position"] >= subregion_info["start"]) & (motifdata["position"] <= subregion_info["end"])
        ]

        motifdata_captured = motifdata.loc[motifdata["captured"]]
        motifdata_notcaptured = motifdata.loc[~motifdata["captured"]]
        ax.scatter(
            motifdata_notcaptured["position"],
            [y + 0.5] * len(motifdata_notcaptured),
            marker="x",
            c="#FF4136",
        )
        ax.scatter(
            motifdata_captured["position"],
            [y + 0.5] * len(motifdata_captured),
            marker="v",
            c="#2ECC40",
        )

    # ChIP-seq + motifs
    if True:
        for setting_id, setting in design.iterrows():
            subregion_info_uncentered = chd.data.peakcounts.plot.uncenter_peaks(subregion_info.copy(), region)

            # preprocess TFBS
            motifs_oi, group_info, motifdata = setting_region_motifdata[(setting_id, subregion_info["region"])]
            plotdata_motifs = motifdata.loc[
                (motifdata["position"] >= subregion_info["start"]) & (motifdata["position"] <= subregion_info["end"])
            ].copy()
            plotdata_motifs["significant"] = False

            do_bigwig = setting["bigwig"] is not None


            panel, ax = grid.add_under(chd.grid.Panel((width, 0.35 if do_bigwig else 0.2)))
            # ax.set_ylabel(setting["label"], rotation = 0, ha = "right", va = "center")

            ax.set_ylim(0, 100)
            ax.set_yscale("symlog", linthresh=10)
            ax.set_yticks([])
            ax.set_yticks([], minor = True)
            ax.set_xticks([])
            ax.set_xlim(subregion_info["start"], subregion_info["end"])
            sns.despine(ax=ax, left=True)


            # chip-seq
            if setting["bigwig"] is not None:
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

                plotdata_motifs["significant"] = (
                    plotdata_chip.set_index("position").reindex(plotdata_motifs["position"])["value"] > 2.0
                ).values
            if setting["peaks"] is not None:
                bed = pybedtools.BedTool(str(setting["peaks"]))

                subregion_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame({"chrom":[subregion_info_uncentered["chrom"]], "start":[int(subregion_info_uncentered["start"])], "end":[int(subregion_info_uncentered["end"])]}))
                intersection = bed.intersect(subregion_bed, wa=True, wb=True).to_dataframe()
                if len(intersection) > 0:
                    intersection = chd.data.peakcounts.plot.center_peaks(intersection, region)
                    intersection["start"] = intersection["start"] - 50
                    intersection["end"] = intersection["end"] + 50

                    for (start, end) in zip(intersection["start"], intersection["end"]):
                        ax.axvspan(start, end, fc="orange", lw=0, alpha=0.3 if do_bigwig else 0.5)
                    plotdata_motifs["significant"] = (
                        plotdata_motifs["position"].apply(lambda x: ((x >= intersection["start"]) & (x <= intersection["end"])).any())
                    )

            # motifs
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
                ax.set_ylabel(setting["label"], rotation=0, ha="right", va="center", fontsize=8)

    # crispr
    if subregion_info["region"] in region_crisprdata:
        panel, ax = grid.add_under(chd.grid.Panel((width, 0.35)))
        ax.set_ylim(np.log(1), np.log(1/16))

        crisprdata = region_crisprdata[subregion_info["region"]]
        crisprdata = crisprdata.loc[
            (crisprdata["start"] >= subregion_info["start"]) & (crisprdata["end"] <= subregion_info["end"])
        ].copy()

        ax.bar(
            crisprdata["start"],
            crisprdata["HS_LS_logratio"],
            width=crisprdata["end"] - crisprdata["start"],
            color="#333",
            lw=0,
        )
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.set_xticks([])
        ax.set_xlim(subregion_info["start"], subregion_info["end"])
        sns.despine(ax=ax, left=True)
        if len(crisprdata) == 0:
            sns.despine(ax=ax, bottom=True, left = True)

        if subregion_ix == 0:
            ax.set_ylabel("CRISPRi low/high\nK562", rotation=0, ha="right", va="center")

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

manuscript.save_figure(fig, "3", f"missed_motifs_{design_name}", dpi = 300)

# %%
manuscript.folder

# %%
# !ls -lh {manuscript.folder}/figure/3

# %%
fig_colorbar = plt.figure(figsize=(2.0, 0.07))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=chd.models.diff.plot.differential.get_norm_atac_diff(),
    cmap=chd.models.diff.plot.differential.get_cmap_atac_diff(),
)
colorbar = plt.colorbar(
    mappable, cax=ax_colorbar, orientation="horizontal", extend="both"
)
colorbar.set_label("Differential\naccessibility", fontsize = 10)
colorbar.set_ticks(np.log([0.25, 0.5, 1, 2, 4]))
colorbar.set_ticklabels(["¼", "½", "1", "2", "4"])
manuscript.save_figure(fig_colorbar, "3", "colorbar_atac_diff")

# %%
# sc.pl.umap(transcriptome.adata, color = ["celltype", transcriptome.gene_id("GATA2")])

# %%
region_oi = chd.data.peakcounts.plot.uncenter_multiple_peaks(slicescores_oi, fragments.regions.coordinates).query("region == @transcriptome.gene_id('PRG2')").iloc[0]
fragments.regions.coordinates.loc[region_oi["region"]]["strand"]

# %%
peaks_path = chd.get_output() / "peaks" / "hspc" / "macs2_summits" / "peaks.bed"
# peaks_path = chd.get_output() / "peaks" / "hspc" / "encode_screen" / "peaks.bed"
# peaks_path = chd.get_output() / "peaks" / "hspc" / "macs2_leiden_0.1_merged" / "peaks.bed"
peaks = pd.read_table(peaks_path, names = ["chrom", "start", "end", "name", "name2", "why"])

# %%
peaks_oi = peaks.query("chrom == @region_oi.chrom").query("~(start >= @region_oi.end_genome)").query("~(end <= @region_oi.start_genome)")

# %%
peaks_oi

# %%
# ENCODE
# peaks.loc[468398, "start"] = 39920217 + 500
# peaks.loc[468398, "end"] = 39920561 + 500
# peaks.loc[630361, "start"] = 131640628 + 5000
# peaks.loc[630361, "end"] = 131640978 + 5000
# peaks.loc[986094, "start"] = 41148615 + 500
# peaks.loc[986094, "end"] = 41148881 + 500
# peaks.loc[986094, "start"] = 41148615 + 500
# peaks.loc[986094, "end"] = 41148881 + 500

# MACS2 SUMMITS
# peaks.loc[26679, "start"] = 716955 + 100
# peaks.loc[26679, "end"] = 717155 + 100
peaks.loc[45140, "start"] = 57363554 + 100
peaks.loc[45140, "end"] = 57363754 + 100

# MACS2 LEIDEN MERGED
# peaks.loc[102576, "start"] = 41148685 + 150
# peaks.loc[102576, "end"] = 41149704 + 0

# %%
peaks.to_csv(peaks_path, sep = "\t", header = False, index = False)

# %%
fragments.regions.coordinates.loc[transcriptome.gene_id('GATA1')]

# %% [markdown]
# ### Simple example

# %%
regions_oi = fragments.regions.coordinates.loc[transcriptome.gene_id(["Lhx2"])]

# %%
slicescores_oi = slicescores.query("cluster in @selection_cluster_ids").query("region in @regions_oi.index")
slicescores_oi["region"] = pd.Categorical(slicescores_oi["region"], categories = regions_oi.index.unique())
slicescores_oi = slicescores_oi.sort_values("region")
slicescores_oi["start"] = slicescores_oi["start"] - padding
slicescores_oi["end"] = slicescores_oi["end"] + padding

# %%
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
slicescores_oi = slicescores_oi.iloc[:n]

# %%
breaking = chd.grid.broken.Breaking(slicescores_oi, 0.05, resolution = 3000)

# %%
fig = chd.grid.Figure(chd.grid.Grid())

region = fragments.regions.coordinates.loc[transcriptome.gene_id("Lhx2")]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
    region,
    breaking,
    genome="mm10",
    label_positions=True,
)
fig.main.add_under(panel_genes)

panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
    region_id, regionpositional, breaking, panel_height=0.4, cluster_info=cluster_info, label_accessibility=False, relative_to = relative_to
)
fig.main.add_under(panel_differential)

# motifs_oi = motifscan.motifs.loc[enrichment.loc["Stellate"].query("q_value < 0.05").sort_values("odds", ascending = False).head(20).index]
# motifs_oi = motifscan.motifs.loc[enrichment.loc["Stellate"].query("q_value < 0.05").sort_values("odds", ascending = False).query("logfoldchanges > 0.5").head(20).index]
motifs_oi = motifscan.motifs.loc[enrichment.loc["Stellate"].query("symbol == 'Wt1'").index]
if len(motifs_oi) > 0:
    panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(motifscan, region.name, motifs_oi = motifs_oi, breaking = breaking)
    fig.main.add_under(panel_motifs)

fig.plot()

# %%
sc.pp.highly_variable_genes(adata)

# %%
adata.obs["n_counts"] = np.array(adata.layers["counts"].sum(1))

# %%
sc.pl.umap(adata, color = "n_counts")

# %%
adata = transcriptome.adata[transcriptome.adata.obs["celltype"] == "KC"]
sc.pp.pca(adata)

gene_ids = transcriptome.gene_id(["Hdac9", "Cadm1", "Zeb2", "Lyz2", "Cd74", "C1qa", "H2-Ab1", "Tmsb4x", "Apoe", "Cd5l"])
sc.pl.umap(adata, color = gene_ids, title = transcriptome.symbol(gene_ids))

# %%
pd.Series(adata.varm["PCs"][:, 0], adata.var["symbol"]).sort_values().head(20)

# %%
enrichment.loc["Stellate"].sort_values("logfoldchanges", ascending = False).dropna().head(10)

# %%
enrichment.loc["Stellate"].query("q_value < 0.05").sort_values("odds", ascending = False).query("logfoldchanges > 0.5")

# %%
enrichment.loc["Stellate"].query("q_value < 0.05").sort_values("odds", ascending = False).head(20).index

# %% [markdown]
# ## Motifs optimize

# %% [markdown]
# ### Get slice scores

# %%
selected_slices = regionpositional.calculate_slices(-1, step = 5)

# %%
probs_mean = selected_slices.data.mean(1, keepdims = True)
actual = selected_slices.data
diff = selected_slices.data - probs_mean

# %%
actual_bins = np.linspace(-3, 2., 20)
diff_bins = np.linspace(0., 2., 20)

# %%
plotdata = pd.DataFrame({
    "actual": actual.flatten()[:500000],
    "diff": diff.flatten()[:500000],
})
plotdata = plotdata.loc[plotdata["diff"] > 0.]
plotdata["actual_bin"] = pd.cut(plotdata["actual"], actual_bins, labels = False)
plotdata["diff_bin"] = pd.cut(plotdata["diff"], diff_bins, labels = False)
plotdata = plotdata.dropna()
plotdata["actual_bin"] = actual_bins[plotdata["actual_bin"].fillna(0.).values.astype(int)]
plotdata["diff_bin"] = diff_bins[plotdata["diff_bin"].fillna(0.).values.astype(int)]

# %%
x1, y1 = (np.log(3.), -3)
x2, y2 = (np.log(2.), 1)

# x1, y1 = (np.log(3.), -3)
# x2, y2 = (np.log(3.), 1)

# %%
counts = plotdata.groupby(["actual_bin", "diff_bin"]).size().unstack().fillna(0).astype(int)
fig, ax = plt.subplots()
ax.matshow(counts, aspect = "auto", extent = [diff_bins[0], diff_bins[-1], actual_bins[-1], actual_bins[0]], cmap = "Blues")
ax.axline((x1, y1), slope = slope, color="black")

# %%
X = diff
Y = actual

slope = -7
target = 10e6
def fun(x1):
    x1, y1 = x1, 0
    x2, y2 = x1 + 1e-4, 0 + slope * 1e-4

    above = ((x2 - x1)*(Y - y1) - (y2 - y1)*(X - x1)) > 0
    score = abs((above.sum() - target))
    print(score)
    return score

import scipy.optimize
res = scipy.optimize.minimize_scalar(fun, bounds = (np.log(1.), np.log(10.)), method = "bounded")

# %%
x1, y1 = res.x, 0
x2, y2 = res.x+1e-4, 0+slope*1e-4

# %%
X = diff
Y = actual

above = ((x2 - x1)*(Y - y1) - (y2 - y1)*(X - x1)) > 0
print(above.sum())

plotdata = pd.DataFrame({
    "actual": actual[above],
    "diff": diff[above],
})
plotdata["actual_bin"] = np.searchsorted(actual_bins, plotdata["actual"])
plotdata["diff_bin"] = np.searchsorted(diff_bins, plotdata["diff"])
# plotdata["actual_bin"] = actual_bins[plotdata["actual_bin"].fillna(0.).values.astype(int)]
# plotdata["diff_bin"] = diff_bins[plotdata["diff_bin"].fillna(0.).values.astype(int)]

# %%
counts = plotdata.groupby(["actual_bin", "diff_bin"]).size().unstack().fillna(0).astype(int)
fig, ax = plt.subplots()
ax.matshow(counts, aspect = "auto", extent = [diff_bins[0], diff_bins[-1], actual_bins[-1], actual_bins[0]], cmap = "Blues")
# ax.axline((x1, y1), slope = slope, color="black")

# %%
# differential_slices = regionpositional.calculate_differential_slices(selected_slices, fc_cutoff = 8., score = "diff2")
# differential_slices = regionpositional.calculate_differential_slices(selected_slices, fc_cutoff = 1.5, score = "diff")
# differential_slices = regionpositional.calculate_differential_slices(selected_slices, fc_cutoff = 3., score = "diff")
# differential_slices = regionpositional.calculate_differential_slices(selected_slices, fc_cutoff = 1., score = "diff3", a = (x1, y1), b = (x2, y2), n = 1e5)
differential_slices = regionpositional.calculate_differential_slices(selected_slices, fc_cutoff = 1., score = "diff3", b = (x1, y1), a = (x2, y2))

# %% [markdown]
# ## Motif mean vs variance

# %%
motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.MotifscanView(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
step = 25
desired_x = np.arange(*fragments.regions.window, step=step) - fragments.regions.window[0]


# %%
def extract_gc(fasta, chrom, start, end, strand, window_size = 20):
    """
    Extract GC content in a region
    """

    convolution = np.ones(window_size) / window_size

    actual_start = start
    if actual_start < 0:
        actual_start = 0
    seq = fasta.fetch(chrom, actual_start, end)
    if strand == -1:
        seq = seq[::-1]

    gc = np.isin(np.array(list(seq)), ["c", "g", "C", "G"])
    gccontent = np.convolve(gc, convolution, mode="same")
    if start > actual_start:
        gccontent = np.pad(gccontent, (actual_start - start, 0))
    if len(gccontent) != end - start:
        gccontent = np.pad(gccontent, (0, end - start - len(gccontent)))

    # fig, ax = plt.subplots()
    # ax.plot(gccontent[:50])
    # for i, x in enumerate(gc[:50]):
    #     if x:
    #         ax.axvline(i, color="red")

    return gccontent


# %%
# region = fragments.regions.coordinates.iloc[0]

# extract_gc(fasta, region["chrom"], region["start"], region["end"], region["strand"], 5)[::int(region["strand"])][:50]

# %%
if not (chd.get_output() / "datasets"/ dataset_name / "gcs.pkl").exists():
    if dataset_name == "liver":
        genome = "mm10"
    else:
        genome = "GRCh38"
    import pysam
    fasta_file = f"/data/genome/{genome}/{genome}.fa"

    fasta = pysam.FastaFile(fasta_file)

    gcs = {}
    for region_id, region in tqdm.tqdm(fragments.regions.coordinates.iterrows()):
        gcs[region_id] = extract_gc(fasta, region["chrom"], region["start"], region["end"], region["strand"])
    pickle.dump(gcs, open(chd.get_output() / "datasets"/ dataset_name / "gcs.pkl", "wb"))

gcs = pickle.load(open(chd.get_output() / "datasets"/ dataset_name / "gcs.pkl", "rb"))


# %%
def select_motif(str):
    return motifscan.motifs.index[motifscan.motifs.index.str.contains(str)][0]


# %%
import sklearn.naive_bayes
class Classifier():
    def __init__(self):
        self.classifier = sklearn.naive_bayes.GaussianNB()

    def fit(self, X, y, imbalance = 100):
        selected_true = np.where(y)[0]
        false = np.where(~y)[0]
        selected_false = false[:len(selected_true)*imbalance]
        # selected_false = np.random.choice(false, len(selected_true)*imbalance, replace=False)
        X = np.concatenate([X[selected_true], X[selected_false]])
        y = np.concatenate([y[selected_true], y[selected_false]])
        self.correction = len(selected_true)/len(false)*imbalance

        if y.sum() == 0:
            print("No true labels")
            self.classifier = None
            return

        self.classifier.fit(X, y)

    def partial_fit(self, X, y, imbalance = 100):
        selected_true = np.where(y)[0]
        false = np.where(~y)[0]
        selected_false = false[:len(selected_true)*imbalance]
        # selected_false = np.random.choice(false, len(selected_true)*imbalance, replace=False)
        X = np.concatenate([X[selected_true], X[selected_false]])
        y = np.concatenate([y[selected_true], y[selected_false]])
        self.correction = len(selected_true)/len(false)*imbalance

        self.classifier.partial_fit(X, y, classes=[0, 1])

    def predict(self, X):
        if self.classifier is None:
            return np.zeros(len(X), dtype=bool)
        return self.classifier.predict_proba(X)[:, 1] * self.correction

class Classifier():
    def __init__(self, step_size = 0.05):
        self.step_size = step_size
        self.bins = np.arange(0, 1+0.000001, step_size)[1:]
        self.cuts = self.bins[:-1]

        self.total = np.zeros(len(self.cuts) + 1)
        self.counts = np.zeros(len(self.cuts) + 1)


    def fit(self, X, y):
        raise NotImplementedError()

    def partial_fit(self, X, y, imbalance = 100):
        bin = np.searchsorted(self.cuts, X[:, 0])

        self.total += np.bincount(bin[:1000], minlength=len(self.cuts) + 1) * (len(bin) / 1000)
        self.counts += np.bincount(bin[y], minlength=len(self.cuts) + 1)

    def predict(self, X):
        bin = np.searchsorted(self.cuts, X[:, 0])
        return self.counts[bin] / self.total[bin]



class GCExpectation():
    def __init__(self):
        self.bins = np.linspace(0, 1, 20)[1:]
        self.cuts = self.bins[:-1]

    def fit(self, motifscan, gcs):
        regions = motifscan.regions

        self.bincounts = np.zeros((motifscan.n_motifs, len(self.cuts) + 1))
        self.bincounts_oi = np.zeros((motifscan.n_motifs, len(self.cuts) + 1))

        self.features = []
        self.labels = {motif_ix:[] for motif_ix in range(motifscan.n_motifs)}

        self.classifiers = []
        for motif_ix in range(motifscan.n_motifs):
            self.classifiers.append(Classifier())

        for region_ix, (region_id, region) in tqdm.tqdm(enumerate(regions.coordinates.iterrows())):
            gc_oi = gcs[region_id]
            positions, indices = motifscan.get_slice(region_ix = region_ix, return_scores = False, return_strands = False)
            positions = positions - regions.window[0]
            positions, indices = positions[positions < len(gc_oi)], indices[positions < len(gc_oi)]

            self.features.append(gc_oi)
            self.bincounts[:] += np.bincount(np.searchsorted(self.cuts, gc_oi), minlength=len(self.cuts) + 1)[None, :]

            for motif_ix in range(motifscan.n_motifs):
                positions_oi = (positions[indices == motif_ix]).astype(int)
                self.bincounts_oi[motif_ix] += np.bincount(np.searchsorted(self.cuts, gc_oi[positions_oi]), minlength=len(self.cuts) + 1)

                labels = np.zeros(len(gc_oi), dtype=bool)
                labels[positions_oi] = True
                self.labels[motif_ix].append(labels)

                # if len(positions_oi) > 0:
                #     self.classifiers[motif_ix].partial_fit(gc_oi[:, None], labels)

            if region_ix > 50:
                break

        # for motif_ix in range(motifscan.n_motifs):
        #     self.classifiers[motif_ix].fit(np.concatenate(self.features)[:, None], np.concatenate(self.labels[motif_ix]))

    def get_expectation(self, gc, motif_ix = None):
        bin = np.searchsorted(self.cuts, gc)
        if motif_ix is not None:
            return (self.bincounts_oi[motif_ix] / self.bincounts[motif_ix])[bin]
        return (self.bincounts_oi / self.bincounts)[:, bin]


# %%
background = GCExpectation()
background.fit(motifscan, gcs)

# %%
motif_oi = select_motif("SPI1")
motif_oi
motif_ix = motifscan.motifs.index.get_indexer([motif_oi])[0]

classifier = background.classifiers[motif_ix]

X_test = np.linspace(0, 1, 100)[:, None]
y_test = classifier.predict(X_test)
plt.plot(X_test[:, 0], y_test)
plt.plot(X_test[:, 0], background.get_expectation(X_test[:, 0], motif_ix))

# %%
motifs = motifscan.motifs

# manual
# motifclustermapping = chdm.motifclustermapping.get_motifclustermapping(dataset_name, motifscan, clustering)

# automatic
enrichment["n_found"] = [x[1, 1] for x in enrichment["contingency"]]
# motifclustermapping = enrichment.query("q_value < 0.05").query("n_found > 20").sort_values("odds", ascending = False).groupby("cluster").head(5).reset_index()
motifclustermapping = enrichment.query("q_value < 0.05").query("n_found > 20").query("odds > 1").sort_values("aggregate", ascending = False).groupby("cluster").head(5).reset_index()
motifclustermapping["motif_ix"] = motifscan.motifs.index.get_indexer(motifclustermapping["motif"])
motifclustermapping["cluster_ix"] = clustering.var.index.get_indexer(motifclustermapping["cluster"])

motifclustermapping = motifclustermapping.loc[~motifclustermapping["cluster"].isin(["Unknown 1", "Unknown 2"])]

# %%
import torch_scatter

# %%
probs_mean_bins = pd.DataFrame(
    {"cut": np.array([-3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2.,2.5,3., np.inf])}
)
probs_mean_bins["label"] = probs_mean_bins["cut"].astype(str)

cuts_left = np.array([2/8, 3/8, 4/8, 5/8, 6/8, 7/8])
clusterprobs_diff_bins = pd.DataFrame(
    {"cut": np.log(np.array([*cuts_left, *(1/cuts_left[::-1]), np.inf]))}
)
clusterprobs_diff_bins["label"] = clusterprobs_diff_bins["cut"].astype(str)
clusterprobs_diff_bins["label"] = ""
clusterprobs_diff_bins.loc[0, "label"] = "<¼"
clusterprobs_diff_bins.loc[2, "label"] = "<½"
clusterprobs_diff_bins.loc[6, "label"] = "≈1"
clusterprobs_diff_bins.loc[10, "label"] = "≥2"
clusterprobs_diff_bins.loc[12, "label"] = "≥4"
# clusterprobs_diff_bins["label"] = ["<1/4", "<1/2", "<2/3", "2/3-4/3", ">4/3", ">2", ">4"]

# %%
probs_mean_bins = pd.DataFrame(
    {"cut_exp":[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2., 4, 8, np.inf]}
)
probs_mean_bins["cut"] = np.log(probs_mean_bins["cut_exp"])
probs_mean_bins["label"] = ["<" + str(probs_mean_bins["cut_exp"][0])] + ["≥" + str(x) for x in probs_mean_bins["cut_exp"].astype(str)[:-1]]

clusterprobs_diff_bins = pd.DataFrame(
    {"cut": list(np.log(np.round(np.logspace(np.log(1.25), np.log(2.5), 7, base = np.e), 5))) + [np.inf]}
)
clusterprobs_diff_bins["cut_exp"] = np.exp(clusterprobs_diff_bins["cut"])
clusterprobs_diff_bins["label"] = ["<" + str(np.round(clusterprobs_diff_bins["cut_exp"][0], 1))] + ["≥" + str(x) for x in np.round(clusterprobs_diff_bins["cut_exp"], 1).astype(str)[:-1]]
clusterprobs_diff_bins["do_label"] = True
clusterprobs_diff_bins

# %%
found = np.zeros((len(motifclustermapping), len(clusterprobs_diff_bins) * len(probs_mean_bins)), dtype=int)
tot = np.zeros((len(motifclustermapping), len(clusterprobs_diff_bins) * len(probs_mean_bins)), dtype=int)
expected = np.zeros((len(motifclustermapping), len(clusterprobs_diff_bins) * len(probs_mean_bins)), dtype=float)
gc = np.zeros((len(motifclustermapping), len(clusterprobs_diff_bins) * len(probs_mean_bins)), dtype=float)
for region_id in tqdm.tqdm(fragments.var.index):
    probs = regionpositional.probs[region_id]
    region = fragments.regions.coordinates.loc[region_id]
    region_ix = fragments.regions.coordinates.index.get_indexer([region_id])[0]

    x_raw = probs.coords["coord"].values - fragments.regions.window[0]
    y_raw = probs.values

    y = chd.utils.interpolate_1d(torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)).numpy()
    ymean = y.mean(0)
    ymax = y.max(0)

    z = y - ymean

    ybin = np.searchsorted(probs_mean_bins["cut"].values, ymean)
    zbin = np.searchsorted(clusterprobs_diff_bins["cut"].values, z)
    bin = ybin * len(clusterprobs_diff_bins) + zbin
    nbin = len(clusterprobs_diff_bins) * len(probs_mean_bins)

    # get association
    positions, indices = motifscan.get_slice(
        region_ix=region_ix,
        return_scores=False,
        return_strands=False,
        motif_ixs=np.unique(motifclustermapping["motif_ix"].values),
    )

    # get gc
    gc_oi = gcs[region_id][::step]
    expected_motif_counts = background.get_expectation(gc_oi)

    # get motifs
    for mapping_ix, (motif_ix, cluster_ix) in enumerate(motifclustermapping[["motif_ix", "cluster_ix"]].values):
        positions_oi = positions[indices == motif_ix] - fragments.regions.window[0]
        ixs = np.clip((positions_oi // step).astype(int), 0, len(ybin) - 1)
        tot[mapping_ix] += np.bincount(bin[cluster_ix], minlength=nbin)
        found[mapping_ix] += np.bincount(bin[cluster_ix][ixs], minlength=nbin)

        expected[mapping_ix] += (
            torch_scatter.scatter_sum(
                torch.from_numpy(expected_motif_counts[motif_ix]), torch.from_numpy(bin[cluster_ix]), dim_size=nbin
            )
            * step
        ).numpy()

        # get gc
        # gc[mapping_ix] += torch_scatter.scatter_sum(torch.from_numpy(gc_oi).float(), torch.from_numpy(bin[cluster_ix]).long(), dim_size = nbin).numpy()

# %%
tot_reshaped = tot.reshape(-1, len(probs_mean_bins), len(clusterprobs_diff_bins))
found_reshaped = found.reshape(-1, len(probs_mean_bins), len(clusterprobs_diff_bins))
expected_reshaped = expected.reshape(-1, len(probs_mean_bins), len(clusterprobs_diff_bins))
gc_reshaped = gc.reshape(-1, len(probs_mean_bins), len(clusterprobs_diff_bins))

# %%
fig, ax = plt.subplots()
ax.plot(ymean)
ax.plot(z[cluster_ix])
ax.plot(y[cluster_ix])
for cut in probs_mean_bins["cut"].values:
    ax.axhline(cut, color="k", linestyle="--")

# %%
plotdata = pd.DataFrame(tot_reshaped[0], columns = clusterprobs_diff_bins["label"], index = probs_mean_bins["label"])
sns.heatmap(np.log1p(plotdata))
# sns.heatmap(np.log(plotdata))

# %%
sns.heatmap(gc_reshaped[0] / tot_reshaped[0])

# %%
d = gc_reshaped
c = gc_reshaped.sum(-1, keepdims=True).sum(-2, keepdims=True) - gc_reshaped
b = tot_reshaped - gc_reshaped
a = tot_reshaped.sum(-1, keepdims=True).sum(-2, keepdims=True) - b - c - d
odds = ((a*d)/(b*c))

odds = (found_reshaped / expected_reshaped) / (found_reshaped.sum()/expected_reshaped.sum())
odds[expected_reshaped < 1] = 1

d = found_reshaped
c = found_reshaped.sum(-1, keepdims=True).sum(-2, keepdims=True) - found_reshaped
b = tot_reshaped - found_reshaped
a = tot_reshaped.sum(-1, keepdims=True).sum(-2, keepdims=True) - b - c - d
odds = ((a*d)/(b*c))

odds = found_reshaped / expected_reshaped

# %%
odds[found_reshaped < 10] = 1

# %%
clustering.var.loc["Megakaryocyte-erythrocyte gradient", "label"] = "M/E gradient"
clustering.var.loc["Erythrocyte precursors", "label"] = "Erythro. prec."
clustering.var.loc["Granulocyte precursor", "label"] = "Granulo. prec."
clustering.var.loc["Megakaryocyte progenitors", "label"] = "Mega. prec."

# %%
fig = chd.grid.Figure(chd.grid.Wrap(padding_width=0.1, padding_height=0.0, ncol = 12))

cmap = mpl.cm.PiYG
norm = mpl.colors.Normalize(vmin=np.log(0.125), vmax=np.log(8.0))

motifclustermapping["ix"] = np.arange(len(motifclustermapping))
motifclustermapping_oi = motifclustermapping
motifclustermapping_oi = motifclustermapping.loc[
    (~motifclustermapping["symbol"].str.contains("RFX3")) &
    (~motifclustermapping["symbol"].str.contains("TCF4")) &
    (~motifclustermapping["symbol"].str.contains("MEF2C"))
]
# motifclustermapping_oi = motifclustermapping.groupby("cluster").head(2).sort_values("cluster")

size = 0.8
for (_, motif_cluster), odds_motifcluster in zip(
    motifclustermapping_oi.iterrows(), odds[motifclustermapping_oi["ix"]]
):
    cluster = motif_cluster["cluster"]
    motif = motif_cluster["motif"]

    odds_max = np.nanmax(odds_motifcluster)

    norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max))

    panel, ax = fig.main.add(chd.grid.Panel((size, size)))
    ax.matshow(np.log(odds_motifcluster), cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    symbol = motifscan.motifs.loc[motif]["symbol"]

    ax.set_title(f"{clustering.var.loc[cluster, 'label']}\n{symbol}", fontsize=8)

# set ticks for bottom left
panel, ax = fig.main.get_bottom_left_corner()
ax.set_ylabel("Mean accessibility")
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"], fontsize = 7)

ax.tick_params(
    axis="x", rotation=90, bottom=True, top=False, labelbottom=True, labeltop=False
)
ax.set_xlabel("Fold accessibility change")
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"], fontsize = 7)

fig.plot()

if dataset_name == "hspc":
    manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment_examples")

# %%
fig_colorbar = plt.figure(figsize=(2.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=norm, cmap=cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal", ticks=[np.log(1/odds_max), 0, np.log(odds_max)], extend = "both")
ax_colorbar.set_xticklabels([r"1/max", "1", f"max"])
colorbar.set_label("Odds ratio")
manuscript.save_figure(fig_colorbar, "4", "colorbar_odds_motif_examples")

# %%
# weight so that each motif contributes equally
weights = 1/found_reshaped.sum(-1).sum(-1)
weights = weights / weights.sum()

# calculate total contingency tables
found_all = (found_reshaped)[motifclustermapping_oi["ix"]].sum(0)
tot_all = (tot_reshaped)[motifclustermapping_oi["ix"]].sum(0)
# found_all = (found_reshaped * weights[:, None, None])[motifclustermapping_oi["ix"]].sum(0)
# tot_all = (tot_reshaped * weights[:, None, None])[motifclustermapping_oi["ix"]].sum(0)

odds_all = ((found_all)/(tot_all+1) / ((found.sum()+1)/(tot.sum()+1)))
odds_all[found_reshaped.sum(0) < 100] = np.nan

plotdata = np.log2(pd.DataFrame(odds_all, columns = clusterprobs_diff_bins["label"], index = probs_mean_bins["label"]))

# %%
fig, ax = plt.subplots(figsize=(2.5, 2))

cmap = mpl.cm.PiYG
odds_max = 8.
norm = mpl.colors.Normalize(vmin=np.log(1/odds_max), vmax=np.log(odds_max), clip=True)

ax.matshow(plotdata, cmap=cmap, norm=norm)
ax.set_ylabel("Mean accessibility")
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])

ax.set_ylabel("")
ax.set_yticks(np.arange(1, len(probs_mean_bins))-0.5)
ax.set_yticklabels([])

ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, len(clusterprobs_diff_bins)) - 0.5, minor=True)
ax.set_xticklabels(np.round(clusterprobs_diff_bins["cut_exp"], 1).astype(str)[:-1], minor=True, rotation = 90)
ax.set_xticks([])
ax.set_xticklabels([])

sns.despine(fig, ax)

if dataset_name == "hspc":
    manuscript.save_figure(fig, "4", "mean_vs_variance_motif_enrichment")
else:
    manuscript.save_figure(fig, "4", f"mean_vs_variance_motif_enrichment_{dataset_name}")

# %%
fig_colorbar = plt.figure(figsize=(2.0, 0.1))
ax_colorbar = fig_colorbar.add_axes([0.05, 0.05, 0.5, 0.9])
mappable = mpl.cm.ScalarMappable(
    norm=norm, cmap=cmap
)
colorbar = plt.colorbar(mappable, cax=ax_colorbar, orientation="horizontal", ticks=[np.log(1/odds_max), 0, np.log(odds_max)], extend = "both")
ax_colorbar.set_xticklabels([f"1/{odds_max:.0f}", "1", f"{odds_max:.0f}"])
colorbar.set_label("Odds ratio")
manuscript.save_figure(fig_colorbar, "4", "colorbar_odds_motif")

# %%
# weight so that each motif contributes equally
weights = 1/found_reshaped.sum(-1).sum(-1)
weights = weights / weights.sum()

# calculate total contingency tables
found_all = (found_reshaped * weights[:, None, None])[motifclustermapping_oi["ix"]].sum(0)
expected_all = (expected_reshaped * weights[:, None, None])[motifclustermapping_oi["ix"]].sum(0)

odds_all = (found_all)/(expected_all)

plotdata = pd.DataFrame(odds_all, columns = clusterprobs_diff_bins["label"], index = probs_mean_bins["label"])

fig, ax = plt.subplots(figsize = (2, 2))
cmap = mpl.cm.PiYG

max_odds = 8
norm = mpl.colors.Normalize(vmin=np.log2(1/max_odds), vmax=np.log2(max_odds))
ax.matshow(np.log2(plotdata), cmap=cmap, norm=norm)
ax.set_xticks(np.arange(len(clusterprobs_diff_bins)))
ax.set_xticklabels(clusterprobs_diff_bins["label"])
ax.set_yticks(np.arange(len(probs_mean_bins)))
ax.set_yticklabels(probs_mean_bins["label"])
ax.tick_params(axis="x", rotation=90, bottom=True, top=False, labelbottom=True, labeltop=False)
ax.set_xlabel("Fold accessibility change")
ax.set_ylabel("Mean\naccessibility", rotation=0, ha="right", va="center")

# %% [markdown]
# ## Cell-type specific eQTL

# %%
import chromatinhd.data.associations
import chromatinhd.data.associations.plot

# %%
motifscan_name = "onek1k_gwas_specific"

associations = chd.data.associations.Associations(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
step = 25
desired_x = np.arange(*fragments.regions.window, step=step) - fragments.regions.window[0]

# %%
association = associations.association

# %%
group_celltype_mapping = {
    "CD4 T":["CD4 memory T", "CD4 naive T"],
    "CD8 T":["CD8 activated T", "CD8 naive T"],
    "NK":["NK"],
    "B":["memory B", "naive B", "Plasma"],
    "Monocyte":["CD14+ Monocytes", "FCGR3A+ Monocytes"],
    "cDC":["cDCs"],
}
assert all([x in clustering.var.index for v in group_celltype_mapping.values() for x in v])
assert all([x in group_celltype_mapping.keys() for x in association["disease/trait"].unique()])

group_celltype_ix_mapping = {associations.motifs.index.get_loc(k):[clustering.var.index.get_indexer([x])[0] for x in v] for i, (k, v) in enumerate(group_celltype_mapping.items())}
groups = list(group_celltype_mapping.keys())
n_groups = len(group_celltype_mapping)

association["group_ix"] = [groups.index(x) for x in association["disease/trait"]]

# %%
ymean = y2.mean(0)
z = y - ymean

# %%
scores = []
for region_ix, region_id in tqdm.tqdm(zip(np.arange(len(fragments.var)), fragments.var.index)):
    probs = regionpositional.probs[region_id]
    region_ix = fragments.regions.coordinates.index.get_indexer([region_id])[0]

    positions, indices = associations.get_slice(region_ix = region_ix, return_scores = False, return_strands = False)
    if len(positions) == 0:
        continue

    x_raw = probs.coords["coord"].values - fragments.regions.window[0]
    y_raw = probs.values

    y = chd.utils.interpolate_1d(
        torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
    ).numpy()

    y2 = np.zeros((n_groups, len(desired_x)))
    for group_ix, celltype_ixs in group_celltype_ix_mapping.items():
        y2[group_ix] = y[celltype_ixs].mean(0)

    ymean = y2.mean(0)
    z = y2 - ymean

    positions = (positions - fragments.regions.window[0]) // step

    scores.append((np.argmax(y2[:, positions], 0) == indices).mean())

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))

window = [20000, 28000]
width = (window[1] - window[0]) / 2000

region = fragments.regions.coordinates.loc[region_id]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window, genome = "mm10" if dataset_name == "liver" else "GRCh38")
fig.main.add_under(panel_genes)

cluster_info = clustering.cluster_info
plotdata, plotdata_mean = regionpositional.get_plotdata(region_id, clusters = cluster_info.index)

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome=transcriptome, clustering=clustering, gene=region_id, panel_height=0.4, order = True, cluster_info = cluster_info
)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=cluster_info, panel_height=0.4, width=width, window = window, order = panel_expression.order, ymax = 5
)

fig.main.add_under(panel_differential)
fig.main.add_right(panel_expression, row=panel_differential)

# panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, region_id, width = width, window = window)
# fig.main.add_under(panel_motifs)

panel_association = chd.data.associations.plot.Associations(associations, region_id, width = width, window = window, show_ld = False)
fig.main.add_under(panel_association)

import chromatinhd_manuscript as chdm
panel_peaks = chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = window, width = width)
fig.main.add_under(panel_peaks)

fig.plot()

# %%
positions

# %%
indices

# %%
associations.motifs

# %%
sns.heatmap(y2[:, positions])

# %%
np.mean(scores), 1/associations.motifs.shape[0]

# %%
np.argmax(y2[:, positions], 0)
