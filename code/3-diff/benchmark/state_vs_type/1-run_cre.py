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


def rank_genes_groups_df(
    adata,
    group,
    *,
    key: str = "rank_genes_groups",
    colnames=("names", "scores", "logfoldchanges", "pvals", "pvals_adj"),
) -> pd.DataFrame:
    """\
    :func:`scanpy.tl.rank_genes_groups` results in the form of a
    :class:`~pandas.DataFrame`.

    Params
    ------
    adata
        Object to get results from.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
        argument) to return results from. Can be a list. All groups are
        returned if groups is `None`.
    key
        Key differential expression groups were stored under.
        

    Example
    -------
    >>> import scanpy as sc
    >>> pbmc = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(pbmc, groupby="louvain", use_raw=True)
    >>> dedf = sc.get.rank_genes_groups_df(pbmc, group="0")
    """
    if isinstance(group, str):
        group = [group]
    if group is None:
        group = list(adata.uns[key]["names"].dtype.names)

    d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
    d = pd.concat(d, axis=1, names=[None, "group"], keys=colnames)
    d = d.stack(level=1).reset_index()
    d["group"] = pd.Categorical(d["group"], categories=group)
    d = d.sort_values(["group", "level_0"]).drop(columns="level_0")

    # remove group column for backward compat if len(group) == 1
    if len(group) == 1:
        d.drop(columns="group", inplace=True)

    return d.reset_index(drop=True)


design = design.query("peakcaller == 'macs2_summits'")
design = design.query("diffexp == 't-test'")
design = design.query("regions == '100k100k'")

from design import dataset_pair_combinations

design = design.merge(dataset_pair_combinations, on="dataset")

print(design)

if design.shape[0] == 0:
    raise ValueError("No designs")

design = design.copy()
dry_run = False
design["force"] = False
# design["force"] = True
# dry_run = True

min_score = np.log(1.25)
min_score = -np.inf

for _, design_row in design.iterrows():
    dataset_name, regions_name, peakcaller, diffexp, latent, celltype_a, celltype_b = design_row[
        ["dataset", "regions", "peakcaller", "diffexp", "latent", "celltype_a", "celltype_b"]
    ]

    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
    fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
    clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

    assert celltype_a in clustering.var.index
    assert celltype_b in clustering.var.index

    scoring_folder = (
        chd.get_output() / "diff" / dataset_name / regions_name / peakcaller / diffexp / f"{celltype_a}_{celltype_b}"
        "scoring" / "regionpositional"
    )

    force = design_row["force"]
    if not (scoring_folder / "differential_slices.pkl").exists():
        force = True

    if not force:
        continue

    print(design_row)

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
    obs = pd.DataFrame({"cluster": clustering.labels.astype(str)}, index=fragments.obs.index)
    cells_oi = obs["cluster"].isin([celltype_a, celltype_b]).values

    for region, _ in tqdm.tqdm(fragments.var.iterrows(), total=fragments.var.shape[0], leave=False):
        var, counts = peakcounts.get_peak_counts(region)

        if counts.sum() == 0:
            print("no counts", dataset_name, regions_name, peakcaller, region)
            continue

        adata_atac = sc.AnnData(
            counts[cells_oi].astype(np.float32),
            obs=obs.loc[cells_oi],
            var=pd.DataFrame(index=var.index),
        )
        # adata_atac = adata_atac[(counts > 0).any(1), :].copy()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.normalize_total(adata_atac)
        sc.pp.log1p(adata_atac)

        if diffexp in ["t-test", "t-test-foldchange"]:
            sc.tl.rank_genes_groups(
                adata_atac,
                "cluster",
                method="t-test",
                max_iter=500,
            )
            columns = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
        elif diffexp == "wilcoxon":
            sc.tl.rank_genes_groups(
                adata_atac,
                "cluster",
                method="wilcoxon",
                max_iter=500,
            )
            columns = ["names", "scores"]
        elif diffexp == "logreg":
            sc.tl.rank_genes_groups(
                adata_atac,
                "cluster",
                method="logreg",
                max_iter=500,
            )
            columns = ["names", "scores"]
        else:
            print("Not supported diffexp ", diffexp)
            continue

        for cluster_oi in [celltype_a, celltype_b]:
            peakscores_cluster = (
                rank_genes_groups_df(adata_atac, group=cluster_oi, colnames=columns)
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

    differential_slices_peak = chd.models.diff.interpret.regionpositional.DifferentialPeaks(
        peakscores["region_ix"].values,
        peakscores["cluster_ix"].values,
        peakscores["relative_start"],
        peakscores["relative_end"],
        data=peakscores["logfoldchanges"] if diffexp.endswith("foldchange") else peakscores["score"],
        n_regions=fragments.regions.n_regions,
    )

    scoring_folder.mkdir(exist_ok=True, parents=True)

    pickle.dump(differential_slices_peak, open(scoring_folder / "differential_slices.pkl", "wb"))
