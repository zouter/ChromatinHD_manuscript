import pandas as pd
import numpy as np
import torch

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import scanpy as sc

import tqdm.auto as tqdm

import pickle

from chromatinhd_manuscript.designs_diff import (
    dataset_latent_peakcaller_diffexp_combinations as design,
)
from chromatinhd_manuscript.diff_params import params

# design = design.query("diffexp == 't-test'")
design = design.query("diffexp == 't-test-foldchange'")
# design = design.query("dataset in ['lymphoma', 'pbmc10kx', 'pbmc10k', 'liver']")
design = design.query("dataset == 'pbmc10k'")
# design = design.query("dataset == 'liver'")
design = design.query("regions == '100k100k'")
design = design.query("peakcaller == 'macs2_leiden_0.1_merged'")
# design = design.query("peakcaller in ['encode_screen', 'rolling_500']")

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

min_score = np.log(1.25)

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

    if not force:
        continue

    print(design_row)

    peakcounts = chd.flow.Flow.from_path(
        chd.get_output() / "datasets" / dataset_name / "peakcounts" / peakcaller / regions_name
    )

    peakscores = []
    obs = pd.DataFrame({"cluster": pd.Categorical(clustering.labels)}, index=fragments.obs.index)

    for region, _ in tqdm.tqdm(fragments.var.iterrows(), total=fragments.var.shape[0], leave=False):
        var, counts = peakcounts.get_peak_counts(region)

        if counts.sum() == 0:
            continue

        adata_atac = sc.AnnData(
            counts.astype(np.float32),
            obs=obs,
            var=pd.DataFrame(index=var.index),
        )
        # adata_atac = adata_atac[(counts > 0).any(1), :].copy()
        sc.pp.normalize_total(adata_atac)
        sc.pp.log1p(adata_atac)

        if diffexp in ["t-test", "t-test-foldchange"]:
            sc.tl.rank_genes_groups(
                adata_atac,
                "cluster",
                method="t-test",
                max_iter=500,
            )

        for cluster_oi in clustering.var.index:
            peakscores_cluster = (
                sc.get.rank_genes_groups_df(adata_atac, group=cluster_oi)
                .rename(columns={"names": "peak", "scores": "score"})
                .set_index("peak")
                .assign(cluster=cluster_oi)
            )
            peakscores_cluster = var.join(peakscores_cluster).sort_values("score", ascending=False)
            peakscores_cluster = peakscores_cluster.query("score > @min_score")
            peakscores.append(peakscores_cluster)
    peakscores = pd.concat(peakscores)
    peakscores["cluster"] = pd.Categorical(peakscores["cluster"], categories=clustering.var.index)
    peakscores["length"] = peakscores["end"] - peakscores["start"]
    if "region" not in peakscores.columns:
        peakscores["region"] = peakscores["gene"]

    peakscores["region_ix"] = fragments.var.index.get_indexer(peakscores["region"])
    peakscores["cluster_ix"] = clustering.var.index.get_indexer(peakscores["cluster"])

    differential_slices_peak = chd.models.diff.interpret.regionpositional.DifferentialSlices(
        peakscores["region_ix"].values,
        peakscores["cluster_ix"].values,
        peakscores["relative_start"],
        peakscores["relative_end"],
        peakscores["score"] if diffexp == "t-test" else peakscores["logfoldchanges"],
        fragments.regions.n_regions,
    )

    scoring_folder.mkdir(exist_ok=True, parents=True)

    pickle.dump(differential_slices_peak, open(scoring_folder / "differential_slices.pkl", "wb"))
