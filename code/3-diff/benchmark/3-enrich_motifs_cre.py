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
from chromatinhd_manuscript.designs_diff import dataset_latent_peakcaller_diffexp_combinations as design_diffexp
from chromatinhd_manuscript.designs_motif import design as design_motif

design = design.merge(design_diffexp)
design = design.merge(design_motif)

design = design.query("splitter == '5x1'")
design = design.query("method == 'v31'")
design = design.query("regions == '100k100k'")
design = design.query("diffexp == 'snap'")
# design = design.query("dataset == 'lymphoma'")
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

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")
    clustering = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    assert method_name == "v31"
    assert splitter == "5x1"

    regionpositional = chd.models.diff.interpret.RegionPositional(
        chd.get_output() / "diff" / dataset_name / regions_name / "5x1" / "v31" / "scoring" / "regionpositional"
    )
    regionpositional.regions = fragments.regions

    # slices_folder = regionpositional.path / "differential" / "0-1.5"
    slices_folder = regionpositional.path / "differential" / "-1-3"

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
            print("!! no cre differential slices found", dataset_name, regions_name, peakcaller, diffexp)
            continue

        for motifscan_name, subdesign in subdesign.groupby("motifscan"):
            enrichment_folder = slices_folder / "enrichment" / motifscan_name / peakcaller / diffexp
            enrichment_folder.mkdir(exist_ok=True, parents=True)

            force = subdesign["force"].iloc[0]
            if not (enrichment_folder / "enrichment.pkl").exists():
                force = True

            if force:
                print(subdesign)

                # load differential slices from reference
                if not (slices_folder / "differential_slices.pkl").exists():
                    print("!! no differential slices found", dataset_name, regions_name, method_name)
                    continue
                differential_slices = pickle.load(open(slices_folder / "differential_slices.pkl", "rb"))
                slicescores = differential_slices.get_slice_scores(regions=fragments.regions, clustering=clustering)
                n_desired_positions = slicescores.groupby("cluster")["length"].sum()

                # load differential slices from CRE
                differential_slices_peak = pickle.load(open(slices_cre_folder / "differential_slices.pkl", "rb"))

                differential_slices_peak.start_position_ixs = (
                    differential_slices_peak.start_position_ixs - fragments.regions.window[0]
                )  # small fix
                differential_slices_peak.end_position_ixs = (
                    differential_slices_peak.end_position_ixs - fragments.regions.window[0]
                )  # small fix
                differential_slices_peak.window = fragments.regions.window

                # match # of differential within each cluster
                slicescores_peak_full = differential_slices_peak.get_slice_scores(
                    regions=fragments.regions, clustering=clustering
                )

                slicescores_peak = []
                for cluster in n_desired_positions.index:
                    peakscores_cluster = slicescores_peak_full.query("cluster == @cluster")
                    peakscores_cluster = peakscores_cluster.sort_values("score", ascending=False)
                    # peakscores_cluster = peakscores_cluster.sort_values("logfoldchanges", ascending=False)
                    n_desired_positions_cluster = n_desired_positions[cluster]

                    # peakscores_cluster["cumulative_length"] = peakscores_cluster["length"].cumsum() # at the latest as large
                    peakscores_cluster["cumulative_length"] = np.pad(
                        peakscores_cluster["length"].cumsum()[:-1], (1, 0)
                    )  # at least as large

                    peakscores_cluster = peakscores_cluster.query("cumulative_length <= @n_desired_positions_cluster")
                    slicescores_peak.append(peakscores_cluster)
                slicescores_peak = pd.concat(slicescores_peak)
                slicescores_peak = slicescores_peak.loc[
                    ~pd.isnull(slicescores_peak["start"]) & ~pd.isnull(slicescores_peak["end"])
                ]
                slicescores_peak["slice"] = pd.Categorical(
                    slicescores_peak["region"].astype(str)
                    + ":"
                    + slicescores_peak["start"].astype(str)
                    + "-"
                    + slicescores_peak["end"].astype(str)
                )
                slices_peak = slicescores_peak.groupby("slice")[["region", "start", "end"]].first()

                # do actual enrichment
                motifscan = chd.data.motifscan.MotifscanView(
                    chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name
                )

                slicecounts_peak = motifscan.count_slices(slices_peak)
                enrichment_peak = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
                    slicescores_peak, slicecounts_peak
                )
                enrichment_peak["log_odds"] = np.log(enrichment_peak["odds"])

                # store enrichment
                pickle.dump(enrichment_peak, open(enrichment_folder / "enrichment.pkl", "wb"))
