import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import tqdm.auto as tqdm

import pickle

from chromatinhd_manuscript.designs_diff import (
    dataset_latent_method_combinations as design,
)
from chromatinhd_manuscript.diff_params import params
from chromatinhd_manuscript.designs_qtl import design as design_qtl

import chromatinhd.data.associations

design = design.merge(design_qtl, on=["dataset", "regions"], how="inner")

design = design.query("splitter == '5x1'")
design = design.query("method == 'v31'")
# design = design.query("dataset == 'e18brain'")

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

for (dataset_name, regions_name, splitter, latent, method_name), subdesign in design.groupby(
    ["dataset", "regions", "splitter", "latent", "method"]
):
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")
    clustering = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    assert method_name == "v31"
    assert splitter == "5x1"

    regionpositional = chd.models.diff.interpret.RegionPositional(
        chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional"
    )
    regionpositional.regions = fragments.regions

    slices_folder = regionpositional.path / "top" / "-1-1.5"

    for motifscan_name, subdesign in subdesign.groupby("motifscan"):
        enrichment_folder = slices_folder / "enrichment" / motifscan_name
        enrichment_folder.mkdir(exist_ok=True, parents=True)

        force = subdesign["force"].iloc[0]

        desired_outputs = [enrichment_folder / "scores.pkl"]
        if not all([desired_output.exists() for desired_output in desired_outputs]):
            force = True

        if force:
            print(subdesign)
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

            import pyranges

            pr = pyranges.PyRanges(
                slices[["chrom", "start_genome", "end_genome"]].rename(
                    columns={"chrom": "Chromosome", "start_genome": "Start", "end_genome": "End"}
                )
            ).merge()
            pr = pr.sort()

            associations = chd.data.associations.Associations(
                chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name
            )
            association = associations.association

            association["start"] = (association["pos"]).astype(int)
            association["end"] = (association["pos"] + 1).astype(int)

            pr_snps = pyranges.PyRanges(
                association.reset_index()[["chr", "start", "end", "index"]].rename(
                    columns={"chr": "Chromosome", "start": "Start", "end": "End"}
                )
            )
            overlap = pr_snps.intersect(pr)

            haplotype_scores = association[["snp", "disease/trait"]].copy()
            haplotype_scores["n_matched"] = (haplotype_scores.index.isin(overlap.as_df()["index"])).astype(int)
            haplotype_scores["n_total"] = 1

            matched = haplotype_scores["n_matched"].sum()
            total_snps = haplotype_scores["n_total"].sum()
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
            print(test[0])
            pickle.dump(scores, open(enrichment_folder / "scores.pkl", "wb"))
