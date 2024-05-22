import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import tqdm.auto as tqdm
import pyranges

import pickle

import chromatinhd.data.associations

from chromatinhd_manuscript.designs_diff import (
    dataset_latent_method_combinations as design,
)
from chromatinhd_manuscript.designs_diff import dataset_latent_peakcaller_diffexp_combinations as design_diffexp
from chromatinhd_manuscript.designs_qtl import design as design_qtl

design = design.merge(design_diffexp)
design = design.merge(design_qtl)

design = design.query("splitter == '5x1'")
design = design.query("method == 'v31'")
design = design.query("diffexp == 'snap'")
# design = design.query("dataset == 'pbmc20k'")
# design = design.query("regions == '100k100k'")

print(design)

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True
for (dataset_name, regions_name, splitter, latent, method_name), subdesign in design.groupby(
    ["dataset", "regions", "splitter", "latent", "method"]
):
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    try:
        fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
        transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")
        clustering = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "latent" / latent)
    except FileNotFoundError:
        print("!! fragments/transcriptome/clustering not found", dataset_name, latent)
        continue

    assert method_name == "v31"
    assert splitter == "5x1"

    regionpositional = chd.models.diff.interpret.RegionPositional(
        chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional"
    )
    regionpositional.regions = fragments.regions

    slices_folder = regionpositional.path / "top" / "-1-1.5"

    for (peakcaller, diffexp), subdesign in subdesign.groupby(["peakcaller", "diffexp"]):
        slices_cre_folder = (
            chd.get_output()
            / "diff"
            / dataset_name
            / regions_name
            / peakcaller
            / diffexp
            / "scoring"
            / "regionpositional"
        )

        if not (slices_cre_folder / "differential_slices.pkl").exists():
            print("!! no differential slices found", dataset_name, regions_name, peakcaller, diffexp)
            continue

        for motifscan_name, subdesign in subdesign.groupby("motifscan"):
            enrichment_folder = slices_folder / "enrichment" / motifscan_name / peakcaller / diffexp
            enrichment_folder.mkdir(exist_ok=True, parents=True)

            force = subdesign["force"].iloc[0]
            if not (enrichment_folder / "scores.pkl").exists():
                force = True

            if force:
                print(subdesign)

                # load differential slices from reference
                if not (slices_folder / "top_slices.pkl").exists():
                    print("!! no top slices found", dataset_name, regions_name, method_name)
                    continue
                top_slices = pickle.load(open(slices_folder / "top_slices.pkl", "rb"))

                slicescores = top_slices.get_slice_scores(regions=fragments.regions)
                coordinates = fragments.regions.coordinates
                slices = chd.data.peakcounts.plot.uncenter_multiple_peaks(slicescores, fragments.regions.coordinates)
                slicescores["slice"] = pd.Categorical(
                    slicescores["chrom"].astype(str)
                    + ":"
                    + slicescores["start_genome"].astype(str)
                    + "-"
                    + slicescores["end_genome"].astype(str)
                )
                slices = slicescores.groupby("slice")[
                    ["region_ix", "start", "end", "chrom", "start_genome", "end_genome"]
                ].first()

                associations = chd.data.associations.Associations(
                    chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name
                )
                association = associations.association

                association["start"] = (association["pos"]).astype(int)
                association["end"] = (association["pos"] + 1).astype(int)

                # determine number of desired positions
                pr = pyranges.PyRanges(
                    slices[["chrom", "start_genome", "end_genome"]].rename(
                        columns={"chrom": "Chromosome", "start_genome": "Start", "end_genome": "End"}
                    )
                ).merge()
                pr = pr.sort()

                n_desired_positions = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
                n_desired_positions

                # load differential slices from CRE
                differential_slices_peak = pickle.load(open(slices_cre_folder / "differential_slices.pkl", "rb"))

                differential_slices_peak.start_position_ixs = (
                    differential_slices_peak.start_position_ixs - fragments.regions.window[0]
                )  # small fix
                differential_slices_peak.end_position_ixs = (
                    differential_slices_peak.end_position_ixs - fragments.regions.window[0]
                )  # small fix
                differential_slices_peak.window = fragments.regions.window  # small fix

                print(
                    differential_slices_peak.get_slice_scores(regions=fragments.regions)
                    .set_index(["region_ix", "start", "end", "cluster_ix"])["score"]
                    .drop_duplicates(keep="first")
                )

                # select top slices from CRE
                slicescores_peak = (
                    differential_slices_peak.get_slice_scores(regions=fragments.regions)
                    .set_index(["region_ix", "start", "end", "cluster_ix"])["score"]
                    .drop_duplicates(keep="first")
                    .unstack()
                    .max(1)
                    .to_frame("score")
                    .reset_index()
                )

                slicescores_peak = chd.data.peakcounts.plot.uncenter_multiple_peaks(
                    slicescores_peak, fragments.regions.coordinates
                )
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
                slices_peak = slicescores_peak[slicescores_peak["cum_length"] <= n_desired_positions].reset_index(
                    drop=True
                )

                import pyranges

                pr_peak = pyranges.PyRanges(
                    slices_peak[["chrom", "start_genome", "end_genome"]].rename(
                        columns={"chrom": "Chromosome", "start_genome": "Start", "end_genome": "End"}
                    )
                ).merge()
                pr_peak = pr_peak.sort()

                # do actual enrichment
                pr_snps = pyranges.PyRanges(
                    association.reset_index()[["chr", "start", "end", "index"]].rename(
                        columns={"chr": "Chromosome", "start": "Start", "end": "End"}
                    )
                )
                overlap = pr_snps.intersect(pr_peak)

                haplotype_scores_peak = association[["snp", "disease/trait"]].copy()
                haplotype_scores_peak["n_matched"] = (
                    haplotype_scores_peak.index.isin(overlap.as_df()["index"])
                ).astype(int)
                haplotype_scores_peak["n_total"] = 1

                matched = haplotype_scores_peak["n_matched"].sum()
                total_snps = haplotype_scores_peak["n_total"].sum()
                total_diff = (pr.as_df()["End"] - pr.as_df()["Start"]).sum()
                total_positions = fragments.regions.width * fragments.n_regions

                contingency = pd.DataFrame(
                    [
                        [matched, total_snps - matched],
                        [total_diff - matched, total_positions - total_snps - total_diff + matched],
                    ],
                    index=["SNP", "Not SNP"],
                    columns=["In slice", "Not in slice"],
                )
                contingency

                from scipy.stats import fisher_exact

                test = fisher_exact(contingency)

                scores = {
                    "matched": matched,
                    "total_snps": total_snps,
                    "total_diff": total_diff,
                    "total_positions": total_positions,
                    "odds": test[0],
                    "pvalue": test[1],
                }
                pickle.dump(scores, open(enrichment_folder / "scores.pkl", "wb"))
