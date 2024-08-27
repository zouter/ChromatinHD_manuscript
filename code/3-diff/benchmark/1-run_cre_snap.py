import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import scanpy as sc

import tqdm.auto as tqdm

import pickle
import pathlib
import time

from chromatinhd_manuscript.designs_diff import (
    dataset_latent_peakcaller_diffexp_combinations as design,
)

# !!
design = design.query("diffexp == 'snap'")

# design = design.query("dataset in ['lymphoma', 'pbmc10kx', 'pbmc10k', 'liver']")
design = design.query("dataset == 'pbmc10k'")
# design = design.query("dataset == 'hspc'")
# design = design.query("dataset == 'liver'")
# design = design.query("dataset == 'lymphoma'")
# design = design.query("dataset == 'e18brain'")
# design = design.query("dataset == 'hspc'")
design = design.query("regions == '100k100k'")
# design = design.query("peakcaller != 'gene_body'")
design = design.query("peakcaller == 'macs2_summits'")
# design = design.query("peakcaller == 'cellranger'")
# design = design.query("peakcaller in ['encode_screen', 'macs2_leiden_0.1_merged', 'macs2_summits']")
# design = design.query("peakcaller in ['encode_screen', 'rolling_500']")

if design.shape[0] == 0:
    raise ValueError("No designs")

design = design.copy()
dry_run = False
design["force"] = False
design["force"] = True
# dry_run = True

R_location = "/data/peak_free_atac/software/R-4.2.2/bin/"
snap_script_location = chd.get_code() / "1-preprocessing" / "peaks" / "run_snap.R"

for _, design_row in design.iterrows():
    dataset_name, regions_name, peakcaller, diffexp, latent = design_row[
        ["dataset", "regions", "peakcaller", "diffexp", "latent"]
    ]

    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
    fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
    clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    scoring_folder = (
        chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / diffexp / "scoring" / "regionpositional"
    )

    force = design_row["force"]
    if not (scoring_folder / "differential_slices.pkl").exists():
        force = True

    if force and (scoring_folder / "differential_slices.pkl").exists():
        (scoring_folder / "differential_slices.pkl").unlink()

    if not force:
        continue

    print(design_row)

    start = time.time()

    try:
        peakcounts = chd.flow.Flow.from_path(
            chd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / regions_name
        )
    except FileNotFoundError:
        print("peakcounts not found", dataset_name, regions_name, peakcaller)
        continue

    if not peakcounts.counted:
        print("peakcounts not counted", dataset_name, regions_name, peakcaller)
        continue

    peakscores = []
    obs = pd.DataFrame({"cluster": pd.Categorical(clustering.labels)}, index=fragments.obs.index)

    var, counts = peakcounts.get_peaks_counts(fragments.var.index)

    counts_cluster = []
    for cluster in obs["cluster"].cat.categories:
        counts_cluster.append(counts[obs["cluster"] == cluster].sum(0))
    counts_cluster = pd.DataFrame(np.stack(counts_cluster, 0), index=obs["cluster"].cat.categories)
    counts_cluster.to_csv("/tmp/counts_cluster.csv")

    print(counts_cluster.shape)

    # run R script
    import os

    script_location = snap_script_location

    os.system(f"{R_location}/Rscript {script_location} /tmp/")

    # postprocess
    peakscores = pd.read_csv("/tmp/tb_pos.csv")

    peakscores["peak_ix"] = peakscores["peak_ix"].str[1:].astype(int)

    peakscores["peak_gene"] = var.index[peakscores["peak_ix"]]
    peakscores["region"] = var["region"].iloc[peakscores["peak_ix"]].values

    peakscores[["relative_start", "relative_end"]] = (
        var[["relative_start", "relative_end"]].iloc[peakscores["peak_ix"]].values
    )

    peakscores["region_ix"] = fragments.var.index.get_indexer(peakscores["region"])
    peakscores["cluster_ix"] = clustering.var.index.get_indexer(peakscores["cluster"])

    differential_slices_peak = chd.models.diff.interpret.regionpositional.DifferentialPeaks(
        peakscores["region_ix"].values,
        peakscores["cluster_ix"].values,
        peakscores["relative_start"],
        peakscores["relative_end"],
        data=-peakscores["logFC"],
        n_regions=fragments.regions.n_regions,
    )

    scoring_folder.mkdir(exist_ok=True, parents=True)

    pickle.dump(differential_slices_peak, open(scoring_folder / "differential_slices.pkl", "wb"))

    end = time.time()
    pickle.dump({"total": end - start}, open(scoring_folder / "time.pkl", "wb"))
    print(end - start)
