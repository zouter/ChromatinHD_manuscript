import pandas as pd
import numpy as np

import chromatinhd as chd
import chromatinhd.scorer
import chromatinhd.peakcounts

import pickle

import torch
import scanpy as sc
import tqdm.auto as tqdm

device = "cuda:1"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# fmt: off
import itertools
design = pd.DataFrame.from_records(itertools.chain(
    # itertools.product(
    #     ["lymphoma"], 
    #     ["celltype"], 
    #     ["v4_128-64-32_30_rep"], 
    #     [
    #         # "cellranger",
    #         # "macs2",
    #         "macs2_improved",
    #         # "rolling_500",
    #         # "significant_up"
    #     ], 
    #     [
    #         "cutoff_0001",
    #         "gwas_lymphoma"
    #     ]
    # ),
    itertools.product(
        ["pbmc10k"],
        ["leiden_0.1"],
        ["v4_128-64-32_30_rep"],
        [
            "cellranger",
            "macs2",
            "macs2_improved",
            "rolling_500",
            "significant_up"
        ],
        [
            "cutoff_0001",
            "gwas_immune",
            "onek1k_0.2"
        ]
    ),
    # itertools.product(
    #     ["e18brain"],
    #     ["leiden_0.1"],
    #     ["v4_128-64-32_30_rep"],
    #     [
    #         "cellranger",
    #         "macs2",
    #         "macs2_improved",
    #         "rolling_500",
    #         "significant_up",
    #     ],
    #     ["cutoff_0001"]
    # )
), columns = ["dataset", "latent", "method", "peaks", "motifscan"])
print(design)

# fmt: on

for dataset_name, design_dataset in design.groupby("dataset"):
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    onehot_promoters = pickle.load(
        (folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open(
            "rb"
        )
    ).flatten(0, 1)

    fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
    fragments.window = window

    # create design to run
    from design import get_design, get_folds_inference

    class Prediction(chd.flow.Flow):
        pass

    # folds & minibatching
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    folds = get_folds_inference(fragments, folds)

    for latent_name, design_latent in design_dataset.groupby("latent"):
        latent_folder = folder_data_preproc / "latent"
        latent = pd.read_pickle(latent_folder / (latent_name + ".pkl"))

        fold_slice = slice(0, 1)

        # folds & minibatching
        folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
        folds = get_folds_inference(fragments, folds)

        for method_name, design_method in design_latent.groupby("method"):
            print(f"{dataset_name=} {promoter_name=} {method_name=}")
            prediction = chd.flow.Flow(
                chd.get_output()
                / "prediction_likelihood"
                / dataset_name
                / promoter_name
                / latent_name
                / method_name
            )

            # models = [
            #     pickle.load(
            #         open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")
            #     )
            #     for fold_ix, fold in enumerate(folds[fold_slice])
            # ]
            # model = models[0]

            probs = pickle.load((prediction.path / "ranking_probs.pkl").open("rb"))
            rho_deltas = pickle.load(
                (prediction.path / "ranking_rho_deltas.pkl").open("rb")
            )
            rhos = pickle.load((prediction.path / "ranking_rhos.pkl").open("rb"))
            design = pickle.load((prediction.path / "ranking_design.pkl").open("rb"))

            # calculate the score we're gonna use: how much does the likelihood of a cut in a window change compared to the "mean"?
            probs_diff = (
                probs - probs.mean(-2, keepdims=True) + rho_deltas
            )  # - rho_deltas.mean(-2, keepdims = True)

            # apply a mask to regions with very low likelihood of a cut
            rho_cutoff = np.log(1.0)
            mask = rhos >= rho_cutoff

            probs_diff_masked = probs_diff.copy()
            probs_diff_masked[~mask] = -np.inf

            ## Single base-pair resolution
            # interpolate the scoring from above but now at single base pairs
            # we may have to smooth this in the future, particularly for very detailed models that already look at base pair resolution
            x = (design["coord"].astype(int).values).reshape(
                (
                    len(design["gene_ix"].cat.categories),
                    len(design["active_latent"].cat.categories),
                    len(design["coord"].cat.categories),
                )
            )

            def interpolate(
                x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
            ) -> torch.Tensor:
                a = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1])
                b = fp[..., :-1] - (a.mul(xp[..., :-1]))

                indices = (
                    torch.searchsorted(xp.contiguous(), x.contiguous(), right=False) - 1
                )
                indices = torch.clamp(indices, 0, a.shape[-1] - 1)
                slope = a.index_select(a.ndim - 1, indices)
                intercept = b.index_select(a.ndim - 1, indices)
                return x * slope + intercept

            desired_x = torch.arange(*window)

            probs_diff_interpolated = interpolate(
                desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs_diff)
            ).numpy()
            rhos_interpolated = interpolate(
                desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(rhos)
            ).numpy()

            # again apply a mask
            rho_cutoff = np.log(1.0)
            probs_diff_interpolated_masked = probs_diff_interpolated.copy()
            mask_interpolated = rhos_interpolated >= rho_cutoff
            probs_diff_interpolated_masked[~mask_interpolated] = -np.inf

            basepair_ranking = probs_diff_interpolated_masked

            # basepair_ranking = model.rank(window, latent.shape[1], device=device)

            print(design_method)
            for peaks_name, design_peaks in design_method.groupby("peaks"):
                print(design_peaks)
                # If comparison with peaks is needed:
                if peaks_name != "significant_up":
                    print(
                        f"{dataset_name=} {promoter_name=} {method_name=} {peaks_name=}"
                    )
                    # get differential peaks for each latent dimension
                    peakcounts = chd.peakcounts.FullPeak(
                        folder=chd.get_output()
                        / "peakcounts"
                        / dataset_name
                        / peaks_name
                    )
                    adata_atac = sc.AnnData(
                        peakcounts.counts.astype(np.float32),
                        obs=fragments.obs,
                        var=peakcounts.var,
                    )
                    sc.pp.normalize_total(adata_atac)
                    sc.pp.log1p(adata_atac)

                    adata_atac.obs["cluster"] = pd.Categorical(
                        latent.columns[np.where(latent.values)[1]],
                        categories=latent.columns,
                    )
                    sc.tl.rank_genes_groups(adata_atac, "cluster")

                    for motifscan_name, design_motifscan in design_peaks.groupby(
                        "motifscan"
                    ):
                        print(
                            f"{dataset_name=} {promoter_name=} {method_name=} {peaks_name=} {motifscan_name=}"
                        )
                        motifscan_folder = (
                            chd.get_output()
                            / "motifscans"
                            / dataset_name
                            / promoter_name
                            / motifscan_name
                        )
                        motifscan = chd.data.Motifscan(motifscan_folder)
                        motifs = pickle.load(
                            (motifscan_folder / "motifs.pkl").open("rb")
                        )
                        motifscan.n_motifs = len(motifs)

                        scores = []
                        motifscores_all = []
                        genemotifscores_all = []

                        for cluster_ix, cluster in enumerate(latent.columns):
                            ## enrichment for peaks
                            peakscores = (
                                sc.get.rank_genes_groups_df(adata_atac, group=cluster)
                                .rename(columns={"names": "peak", "scores": "score"})
                                .set_index("peak")
                            )
                            peakscores_joined = peakcounts.peaks.join(
                                peakscores, on="peak"
                            ).sort_values("score", ascending=False)

                            positive = (peakscores_joined["logfoldchanges"] > 1.0) & (
                                peakscores_joined["pvals_adj"] < 0.05
                            )
                            if positive.sum() < 20:
                                positive = peakscores_joined.index.isin(
                                    peakscores_joined.index[:20]
                                )
                            negative = ~positive

                            position_slices = peakscores_joined.loc[
                                positive, ["relative_start", "relative_end"]
                            ].values
                            position_slices = position_slices - window[0]
                            gene_ixs_slices = peakscores_joined.loc[
                                positive, "gene_ix"
                            ].values
                            # gene_ixs_slices = np.random.permutation(gene_ixs_slices)

                            motifscores_peak = (
                                chd.differential.enrichment.enrich_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    onehot_promoters,
                                    n_genes=fragments.n_genes,
                                    window=window,
                                )
                            )
                            genemotifscores_peak = (
                                chd.differential.enrichment.detect_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    fragments.var.index,
                                    window=window,
                                )
                            )

                            # calculate percentage of base pairs in a peak
                            # the same percentage will be used later to select regions
                            n = (position_slices[:, 1] - position_slices[:, 0]).sum()
                            perc = n / (fragments.n_genes * (window[1] - window[0]))

                            ## enrichment for regions
                            cutoff = np.quantile(
                                basepair_ranking[:, cluster_ix], 1 - perc
                            )
                            basepairs_oi = basepair_ranking[:, cluster_ix] >= cutoff

                            # convert chosen positions to slices
                            gene_ixs, positions = np.where(basepairs_oi)
                            groups = np.hstack(
                                [
                                    0,
                                    np.cumsum(
                                        (
                                            np.diff(
                                                gene_ixs * ((window[1] - window[0]) + 1)
                                                + positions
                                            )
                                            != 1
                                        )
                                    ),
                                ]
                            )
                            cuts = np.where(
                                np.hstack([True, (np.diff(groups)[:-1] != 0), True])
                            )[0]

                            position_slices = np.vstack(
                                (positions[cuts[:-1]], positions[cuts[1:] - 1])
                            ).T
                            gene_ixs_slices = gene_ixs[cuts[:-1]]
                            # gene_ixs_slices = np.random.permutation(gene_ixs_slices)

                            motifscores_region = (
                                chd.differential.enrichment.enrich_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    onehot_promoters,
                                    fragments.n_genes,
                                    window,
                                )
                            )
                            genemotifscores_region = (
                                chd.differential.enrichment.detect_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    fragments.var.index,
                                    window,
                                )
                            )

                            # calculate overenrichment
                            motifs_oi = np.repeat(True, motifscores_peak.shape[0])

                            try:
                                import scipy.stats

                                linreg = scipy.stats.linregress(
                                    motifscores_region.loc[motifs_oi, "logodds"],
                                    motifscores_peak.loc[motifs_oi, "logodds"],
                                )
                                slope = linreg.slope
                                intercept = linreg.intercept
                            except ValueError as e:
                                print(e)
                                slope = 1
                                intercept = 0

                            print(1 / slope, cluster)

                            motifscores_all.append(
                                pd.concat(
                                    [
                                        motifscores_peak.rename(
                                            columns=lambda x: x + "_peak"
                                        ),
                                        motifscores_region.rename(
                                            columns=lambda x: x + "_region"
                                        ),
                                    ],
                                    axis=1,
                                ).assign(cluster=cluster)
                            )

                            genemotifscores_all.append(
                                pd.concat(
                                    [
                                        genemotifscores_peak.rename(
                                            columns=lambda x: x + "_peak"
                                        ),
                                        genemotifscores_region.rename(
                                            columns=lambda x: x + "_region"
                                        ),
                                    ],
                                    axis=1,
                                ).assign(cluster=cluster)
                            )

                            scores.append(
                                {
                                    "cluster": cluster,
                                    "slope": 1 / slope,
                                    "n_cells": latent[cluster].sum(),
                                    "n_position": basepairs_oi.sum(),
                                    "cutoff": cutoff,
                                }
                            )

                        scores = pd.DataFrame(scores)
                        motifscores_all = pd.concat(motifscores_all)
                        genemotifscores_all = pd.concat(genemotifscores_all)

                        scores_dir = (
                            prediction.path / "scoring" / peaks_name / motifscan_name
                        )
                        scores_dir.mkdir(parents=True, exist_ok=True)

                        scores.to_pickle(scores_dir / "scores.pkl")
                        motifscores_all.to_pickle(scores_dir / "motifscores_all.pkl")
                        genemotifscores_all.to_pickle(
                            scores_dir / "genemotifscores_all.pkl"
                        )

                else:
                    for motifscan_name, design_motifscan in design_peaks.groupby(
                        "motifscan"
                    ):
                        print(
                            f"{dataset_name=} {promoter_name=} {method_name=} significant_up {motifscan_name=}"
                        )
                        motifscan_folder = (
                            chd.get_output()
                            / "motifscans"
                            / dataset_name
                            / promoter_name
                            / motifscan_name
                        )
                        motifscan = chd.data.Motifscan(motifscan_folder)
                        motifs = pickle.load(
                            (motifscan_folder / "motifs.pkl").open("rb")
                        )
                        motifscan.n_motifs = len(motifs)

                        scores = []
                        motifscores = []

                        for cluster_ix, cluster in enumerate(latent.columns):
                            ## enrichment for regions
                            cutoff = np.log(2)
                            basepairs_oi = basepair_ranking[:, cluster_ix] >= cutoff

                            # convert chosen positions to slices
                            gene_ixs, positions = np.where(basepairs_oi)
                            groups = np.hstack(
                                [
                                    0,
                                    np.cumsum(
                                        (
                                            np.diff(
                                                gene_ixs * ((window[1] - window[0]) + 1)
                                                + positions
                                            )
                                            != 1
                                        )
                                    ),
                                ]
                            )
                            cuts = np.where(
                                np.hstack([True, (np.diff(groups)[:-1] != 0), True])
                            )[0]

                            position_slices = np.vstack(
                                (positions[cuts[:-1]], positions[cuts[1:] - 1])
                            ).T
                            gene_ixs_slices = gene_ixs[cuts[:-1]]
                            # gene_ixs_slices = np.random.permutation(gene_ixs_slices)

                            motifscores_region = enrich_windows(
                                motifscan,
                                position_slices,
                                gene_ixs_slices,
                                onehot_promoters,
                            )

                            motifscores.append(
                                motifscores_region.assign(cluster=cluster)
                            )

                            scores.append(
                                {
                                    "cluster": cluster,
                                    "n_cells": latent[cluster].sum(),
                                    "n_position": basepairs_oi.sum(),
                                    "cutoff": cutoff,
                                }
                            )

                        scores = pd.DataFrame(scores)
                        motifscores = pd.concat(motifscores)

                        scores_dir = (
                            prediction.path
                            / "scoring"
                            / "significant_up"
                            / motifscan_name
                        )
                        print(scores_dir)
                        scores_dir.mkdir(parents=True, exist_ok=True)

                        scores.to_pickle(scores_dir / "scores.pkl")
                        motifscores.to_pickle(scores_dir / "motifscores.pkl")
