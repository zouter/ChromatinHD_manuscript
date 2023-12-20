import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile

import chromatinhd.data.associations
import chromatinhd.data.associations.plot

dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
# dataset_name = "liver"
regions_name = "100k100k"
# regions_name = "10k10k"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
if dataset_name == "pbmc10k/subsets/top250":
    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

models = chd.models.diff.model.binary.Models(chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31")
regionpositional = chd.models.diff.interpret.RegionPositional(models.path / "scoring" / "regionpositional")

regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering


def uncenter_peaks(slices, coordinates):
    if "region_ix" not in slices.columns:
        slices["region_ix"] = coordinates.index.get_indexer(slices["region"])
    coordinates_oi = coordinates.iloc[slices["region_ix"]].copy()

    slices["chrom"] = coordinates_oi["chrom"].values

    slices["start_genome"] = np.where(
        coordinates_oi["strand"] == 1,
        (slices["start"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
        (slices["end"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
    )
    slices["end_genome"] = np.where(
        coordinates_oi["strand"] == 1,
        (slices["end"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
        (slices["start"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
    )
    return slices


coordinates = fragments.regions.coordinates

# load peaks
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_100" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "rolling_500" / "t-test" / "scoring" / "regionpositional"
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test" / "scoring" / "regionpositional"
scoring_folder = (
    chd.get_output()
    / "diff"
    / dataset_name
    / regions_name
    / "encode_screen"
    / "t-test"
    / "scoring"
    / "regionpositional"
)
# scoring_folder = chd.get_output() / "diff" / dataset_name / regions_name / "macs2_leiden_0.1_merged" / "t-test-foldchange" / "scoring" / "regionpositional"
differential_slices_peak = pickle.load(open(scoring_folder / "differential_slices.pkl", "rb"))

differential_slices_peak.start_position_ixs = differential_slices_peak.start_position_ixs - fragments.regions.window[0]
differential_slices_peak.end_position_ixs = differential_slices_peak.end_position_ixs - fragments.regions.window[0]
differential_slices_peak.window = fragments.regions.window

# load associations
motifscan_name = "gwas_immune"
# motifscan_name = "causaldb_immune"
# motifscan_name = "gwas_liver"
# motifscan_name = "gwas_lymphoma"
# motifscan_name = "causaldb_lymphoma"
associations = chd.data.associations.Associations(
    chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name
)

association = associations.association
haplotypes = {}
for (haplotype, trait), snps in association.groupby(["snp_main", "disease/trait"]):
    snps["start"] = (snps["pos"] - 1).astype(int)
    snps["end"] = snps["pos"].astype(int)

    # filter only on main SNP
    snps = snps.loc[snps["snp"] == snps["snp_main"]]
    if len(snps) == 0:
        continue

    haplotypes[haplotype] = {"chrom": snps["chr"].values[0], "trait": trait, "snps": snps}

# design
design = chd.utils.crossing(
    pd.DataFrame({"overall_cutoff": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]}),
    # pd.DataFrame({"differential_cutoff": [2.5]}),
    pd.DataFrame({"differential_cutoff": [1.1, 1.2, 1.5, 1.75, 2.0, 2.25, 2.5]}),
)

for row_ix, row in design.iterrows():
    slices_file = regionpositional.path / "slices_{:.2f}.pkl".format(row["overall_cutoff"])
    print(slices_file)
    print(row["overall_cutoff"], row["differential_cutoff"])

    if slices_file.exists():
        slices = pickle.load(open(slices_file, "rb"))
    else:
        slices = regionpositional.calculate_slices(row["overall_cutoff"], step=25)
        pickle.dump(slices, open(slices_file, "wb"))

    top_slices = regionpositional.calculate_top_slices(slices, row["differential_cutoff"])

    slicescores = top_slices.get_slice_scores(regions=fragments.regions)
    coordinates = fragments.regions.coordinates
    slices = uncenter_peaks(slicescores, fragments.regions.coordinates)
    slicescores["slice"] = pd.Categorical(
        slicescores["chrom"].astype(str)
        + ":"
        + slicescores["start_genome"].astype(str)
        + "-"
        + slicescores["end_genome"].astype(str)
    )
    slices = slicescores.groupby("slice")[["region_ix", "start", "end", "chrom", "start_genome", "end_genome"]].first()

    import pyranges

    pr = pyranges.PyRanges(
        slices[["chrom", "start_genome", "end_genome"]].rename(
            columns={"chrom": "Chromosome", "start_genome": "Start", "end_genome": "End"}
        )
    ).merge()
    pr = pr.sort()

    n_desired_positions = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
    n_desired_positions

    slicescores_peak = (
        differential_slices_peak.get_slice_scores(regions=fragments.regions)
        .set_index(["region_ix", "start", "end", "cluster_ix"])["score"]
        .unstack()
        .max(1)
        .to_frame("score")
        .reset_index()
    )

    slicescores_peak = uncenter_peaks(slicescores_peak, fragments.regions.coordinates)
    slicescores_peak["slice"] = pd.Categorical(
        slicescores_peak["chrom"].astype(str)
        + ":"
        + slicescores_peak["start_genome"].astype(str)
        + "-"
        + slicescores_peak["end_genome"].astype(str)
    )
    slicescores_peak = slicescores_peak.groupby("slice")[
        ["region_ix", "start", "end", "chrom", "start_genome", "end_genome", "score"]
    ].first()

    slicescores_peak = slicescores_peak.sort_values("score", ascending=False)
    slicescores_peak["length"] = slicescores_peak["end"] - slicescores_peak["start"]
    slicescores_peak["cum_length"] = slicescores_peak["length"].cumsum()
    slices_peak = slicescores_peak[slicescores_peak["cum_length"] <= n_desired_positions].reset_index(drop=True)

    import pyranges

    pr_peak = pyranges.PyRanges(
        slices_peak[["chrom", "start_genome", "end_genome"]].rename(
            columns={"chrom": "Chromosome", "start_genome": "Start", "end_genome": "End"}
        )
    ).merge()
    pr_peak = pr_peak.sort()

    haplotype_scores = []

    overlaps = []
    for haplotype, haplotype_info in haplotypes.items():
        pr_snps = pyranges.PyRanges(
            haplotype_info["snps"][["chr", "start", "end"]].rename(
                columns={"chr": "Chromosome", "start": "Start", "end": "End"}
            )
        )
        overlap = pr_snps.intersect(pr)
        overlaps.append(overlap)
        haplotype_scores.append(
            {
                "haplotype": haplotype,
                "trait": haplotype_info["trait"],
                "n_matched": len(overlap),
                "n_total": len(pr_snps),
            }
        )
    haplotype_scores = pd.DataFrame(haplotype_scores).set_index(["haplotype", "trait"])

    matched = haplotype_scores["n_matched"].sum()
    total_snps = haplotype_scores["n_total"].sum()
    total_diff = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
    total_positions = fragments.regions.width * fragments.n_regions

    contingency = pd.DataFrame(
        [[matched, total_snps - matched], [total_diff - matched, total_positions - total_snps - total_diff + matched]],
        index=["SNP", "Not SNP"],
        columns=["In slice", "Not in slice"],
    )
    contingency

    from scipy.stats import fisher_exact

    fisher = fisher_exact(contingency)
    print(fisher.statistic)
    design.loc[row_ix, "odds"] = fisher.statistic

    ##
    haplotype_scores_peak = []

    overlaps = []
    for haplotype, haplotype_info in haplotypes.items():
        pr_snps = pyranges.PyRanges(
            haplotype_info["snps"][["chr", "start", "end"]].rename(
                columns={"chr": "Chromosome", "start": "Start", "end": "End"}
            )
        )
        overlap = pr_snps.intersect(pr_peak)
        overlaps.append(overlap)
        haplotype_scores_peak.append(
            {
                "haplotype": haplotype,
                "trait": haplotype_info["trait"],
                "n_matched": len(overlap),
                "n_total": len(pr_snps),
            }
        )
    haplotype_scores_peak = pd.DataFrame(haplotype_scores_peak).set_index(["haplotype", "trait"])

    matched = haplotype_scores_peak["n_matched"].sum()
    total_snps = haplotype_scores_peak["n_total"].sum()
    total_diff = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
    total_positions = fragments.regions.width * fragments.n_regions

    contingency = pd.DataFrame(
        [[matched, total_snps - matched], [total_diff - matched, total_positions - total_snps - total_diff + matched]],
        index=["SNP", "Not SNP"],
        columns=["In slice", "Not in slice"],
    )
    contingency

    from scipy.stats import fisher_exact

    fisher = fisher_exact(contingency)
    print(fisher.statistic)
    design.loc[row_ix, "odds_peak"] = fisher.statistic

    print(
        (
            haplotype_scores.join(haplotype_scores_peak, rsuffix="_peak")
            .query("(n_matched > 0) & (n_matched_peak == 0)")
            .shape[0]
            + 1e-3
        )
        / (
            haplotype_scores.join(haplotype_scores_peak, rsuffix="_peak")
            .query("(n_matched == 0) & (n_matched_peak > 0)")
            .shape[0]
            + 1e-3
        )
    )


print(design)
