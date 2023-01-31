import pandas as pd
import numpy as np

import peakfreeatac as pfa
import peakfreeatac.scorer
import peakfreeatac.peakcounts

import pickle

import torch
import scanpy as sc
import tqdm.auto as tqdm

device = "cuda:1"

folder_root = pfa.get_output()
folder_data = folder_root / "data"


# def count_gc(relative_starts, relative_end, gene_ixs, onehot_promoters):
#     eps = 1e-5
#     gc = []
#     for relative_start, relative_end, gene_ix in (zip(relative_starts, relative_end, gene_ixs)):
#         start_ix = gene_ix * (window[1] - window[0]) + relative_start
#         end_ix = gene_ix * (window[1] - window[0]) + relative_end
#         gc.append(onehot_promoters[start_ix:end_ix, [1, 2]].sum() / (end_ix - start_ix + eps))

#     gc = torch.hstack(gc).numpy()

#     return gc

# def count_motifs(relative_starts, relative_end, gene_ixs, motifscan_indices, motifscan_indptr, n_motifs):
#     motif_indices = []
#     n = 0
#     for relative_start, relative_end, gene_ix in (zip(relative_starts, relative_end, gene_ixs)):
#         start_ix = gene_ix * (window[1] - window[0]) + relative_start
#         end_ix = gene_ix * (window[1] - window[0]) + relative_end
#         motif_indices.append(motifscan_indices[motifscan_indptr[start_ix]:motifscan_indptr[end_ix]])
#         n += relative_end - relative_start
#     motif_indices = np.hstack(motif_indices)
#     motif_counts = np.bincount(motif_indices, minlength = n_motifs)

#     return motif_counts, n


# def count_motifs_genewise(relative_starts, relative_end, gene_ixs, motifscan_indices, motifscan_indptr, n_motifs, n_genes):
#     motif_indices = []
#     for relative_start, relative_end, gene_ix in (zip(relative_starts, relative_end, gene_ixs)):
#         start_ix = gene_ix * (window[1] - window[0]) + relative_start
#         end_ix = gene_ix * (window[1] - window[0]) + relative_end
#         motif_indices.append(motifscan_indices[motifscan_indptr[start_ix]:motifscan_indptr[end_ix]] + gene_ix * n_motifs)
#     motif_indices = np.hstack(motif_indices)
#     motif_counts = np.bincount(motif_indices, minlength = n_motifs * n_genes).reshape((n_genes, n_motifs))

#     return motif_counts

# def select_background(position_slices, gene_ixs_slices, onehot_promoters, n_random = 100, n_select_random = 10, seed = None):
#     window_oi_gc = count_gc(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, onehot_promoters)

#     if seed is not None:
#         rg = np.random.RandomState(seed)
#     else:
#         rg = np.random.RandomState()
#     position_slices_repeated = position_slices.repeat(n_random, 0)
#     random_position_slices = np.zeros_like(position_slices_repeated)
#     random_position_slices[:, 0] = rg.randint(np.ones(position_slices_repeated.shape[0]) * window[0], np.ones(position_slices_repeated.shape[0]) * window[1] - (position_slices_repeated[:, 1] - position_slices_repeated[:, 0]) + 1)
#     random_position_slices[:, 1] = random_position_slices[:, 0] + (position_slices_repeated[:, 1] - position_slices_repeated[:, 0])
#     random_gene_ixs_slices = rg.randint(fragments.n_genes, size = random_position_slices.shape[0])

#     window_random_gc = count_gc(random_position_slices[:, 0], random_position_slices[:, 1], random_gene_ixs_slices, onehot_promoters)

#     random_difference = np.abs((window_random_gc.reshape((position_slices.shape[0], n_random)) - window_oi_gc[:, None]))

#     chosen_background = np.argsort(random_difference, axis = 1)[:, :n_select_random].flatten()
#     chosen_background_idx = np.repeat(np.arange(position_slices.shape[0]), n_select_random) * n_random + chosen_background

#     background_position_slices = random_position_slices[chosen_background_idx]
#     background_gene_ixs_slices = random_gene_ixs_slices[chosen_background_idx]

#     return background_position_slices, background_gene_ixs_slices


# def enrich_windows(motifscan, position_slices, gene_ixs_slices, onehot_promoters):
#     motif_counts, n = count_motifs(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, motifscan.indices, motifscan.indptr, motifscan.n_motifs)

#     background_position_slices, background_gene_ixs_slices = select_background(
#         position_slices, gene_ixs_slices, onehot_promoters, seed = 1
#     )
#     motif_counts2, n2 = count_motifs(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, motifscan.indices, motifscan.indptr,  motifscan.n_motifs)

#     # create contingencies to calculate conditional odds
#     contingencies = np.stack([
#         np.stack([n2 - motif_counts2, motif_counts2]),
#         np.stack([n - motif_counts, motif_counts]),
#     ]).transpose(2, 0, 1)
#     import scipy.stats
#     odds_conditional = []
#     for cont in contingencies:
#         odds_conditional.append(scipy.stats.contingency.odds_ratio(cont + 1, kind='conditional').statistic) # pseudocount = 1

#     n_motifs = np.bincount(motifscan.indices, minlength = motifscan.n_motifs)

#     # create motifscores
#     motifscores = pd.DataFrame({
#         "odds":((motif_counts+1) / (n+1)) / ((motif_counts2+1) / (n2+1)),
#         "odds_conditional":odds_conditional,
#         "motif":motifscan.motifs.index,
#         "in":motif_counts/n_motifs
#     }).set_index("motif")
#     motifscores["logodds"] = np.log(odds_conditional)

#     return motifscores


# def detect_windows(motifscan, position_slices, gene_ixs_slices, gene_ids):
#     motif_counts = count_motifs_genewise(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, motifscan.indices, motifscan.indptr, motifscan.n_motifs, len(gene_ids))
#     gene_counts = pd.DataFrame({"n":position_slices[:, 1]-position_slices[:, 0], "gene_ix":gene_ixs_slices}).groupby("gene_ix")["n"].sum()
#     gene_counts.index = gene_ids[gene_counts.index].values
#     gene_counts = gene_counts.reindex(gene_ids, fill_value=0)

#     motifscores = pd.DataFrame(motif_counts, index = gene_ids, columns = motifscan.motifs.index,).stack().to_frame(name = "n")
#     motifscores["n_positions"] = gene_counts.loc[motifscores.index.get_level_values("gene")].values

#     return motifscores

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

    fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
    fragments.window = window

    # create design to run
    from design import get_design, get_folds_inference

    class Prediction(pfa.flow.Flow):
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
            prediction = Prediction(
                pfa.get_output()
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
            rho_deltas = pickle.load((prediction.path / "ranking_rho_deltas.pkl").open("rb"))
            rhos = pickle.load((prediction.path / "ranking_rhos.pkl").open("rb"))
            design = pickle.load((prediction.path / "ranking_design.pkl").open("rb"))

            # calculate the score we're gonna use: how much does the likelihood of a cut in a window change compared to the "mean"?
            probs_diff = probs - probs.mean(-2, keepdims = True) + rho_deltas# - rho_deltas.mean(-2, keepdims = True)

            # apply a mask to regions with very low likelihood of a cut
            rho_cutoff = np.log(1.)
            mask = rhos >= rho_cutoff

            probs_diff_masked = probs_diff.copy()
            probs_diff_masked[~mask] = -np.inf

            ## Single base-pair resolution
            # interpolate the scoring from above but now at single base pairs
            # we may have to smooth this in the future, particularly for very detailed models that already look at base pair resolution
            x = (design["coord"].astype(int).values).reshape((len(design["gene_ix"].cat.categories), len(design["active_latent"].cat.categories), len(design["coord"].cat.categories)))

            def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
                a = (fp[...,1:] - fp[...,:-1]) / (xp[...,1:] - xp[...,:-1])
                b = fp[..., :-1] - (a.mul(xp[..., :-1]) )

                indices = torch.searchsorted(xp.contiguous(), x.contiguous(), right=False) - 1
                indices = torch.clamp(indices, 0, a.shape[-1] - 1)
                slope = a.index_select(a.ndim-1, indices)
                intercept = b.index_select(a.ndim-1, indices)
                return x * slope + intercept

            desired_x = torch.arange(*window)

            probs_diff_interpolated = interpolate(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs_diff)).numpy()
            rhos_interpolated = interpolate(desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(rhos)).numpy()

            # again apply a mask
            rho_cutoff = np.log(1.)
            probs_diff_interpolated_masked = probs_diff_interpolated.copy()
            mask_interpolated = (rhos_interpolated >= rho_cutoff)
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
                    peakcounts = pfa.peakcounts.FullPeak(
                        folder=pfa.get_output()
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
                            pfa.get_output()
                            / "motifscans"
                            / dataset_name
                            / promoter_name
                            / motifscan_name
                        )
                        motifscan = pfa.data.Motifscan(motifscan_folder)
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
                                pfa.differential.enrichment.enrich_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    onehot_promoters,
                                    n_genes = fragments.n_genes,
                                    window = window,
                                )
                            )
                            genemotifscores_peak = (
                                pfa.differential.enrichment.detect_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    fragments.var.index,
                                    window = window
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
                                pfa.differential.enrichment.enrich_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    onehot_promoters,
                                    fragments.n_genes,
                                    window
                                )
                            )
                            genemotifscores_region = (
                                pfa.differential.enrichment.detect_windows(
                                    motifscan,
                                    position_slices,
                                    gene_ixs_slices,
                                    fragments.var.index,
                                    window
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
                            pfa.get_output()
                            / "motifscans"
                            / dataset_name
                            / promoter_name
                            / motifscan_name
                        )
                        motifscan = pfa.data.Motifscan(motifscan_folder)
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
