import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import scanpy as sc
import chromatinhd_manuscript as chdm
import itertools


class Example:
    def __init__(
        self,
        dataset_name,
        promoter_name,
        latent_name,
        method_name,
        motifscan_name,
        symbol,
        motifs_to_merge=tuple(),
        subset_clusters: list = None,
        **kwargs
    ):
        folder_root = chd.get_output()
        folder_data = folder_root / "data"

        folder_data_preproc = folder_data / dataset_name

        self.promoters = pd.read_csv(
            folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
        )

        if promoter_name == "10k10k":
            window = np.array([-10000, 10000])
        else:
            raise ValueError()

        self.transcriptome = chd.data.Transcriptome(
            folder_data_preproc / "transcriptome"
        )
        self.fragments = chd.data.Fragments(
            folder_data_preproc / "fragments" / promoter_name
        )
        self.fragments.window = window

        self.fragments.create_cut_data()

        self.fragments.obs["lib"] = torch.bincount(
            self.fragments.cut_local_cell_ix, minlength=self.fragments.n_cells
        ).numpy()

        ## Latent space

        # loading
        folder_data_preproc = folder_data / dataset_name
        latent_folder = folder_data_preproc / "latent"
        latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))
        latent_torch = torch.from_numpy(latent.values).to(torch.float)

        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
        cluster_info["color"] = sns.color_palette("husl", latent.shape[1])

        self.fragments.obs["cluster"] = pd.Categorical(
            pd.from_dummies(latent).iloc[:, 0]
        )
        if self.transcriptome is not None:
            self.transcriptome.obs["cluster"] = self.transcriptome.adata.obs[
                "cluster"
            ] = self.fragments.obs["cluster"] = pd.Categorical(
                pd.from_dummies(latent).iloc[:, 0]
            )

        cluster_info["lib"] = self.fragments.obs.groupby("cluster")["lib"].sum().values

        if subset_clusters is not None:
            cluster_info_oi = cluster_info.loc[subset_clusters]
        else:
            cluster_info_oi = cluster_info

        # probs
        prediction = chd.flow.Flow(
            chd.get_output()
            / "prediction_likelihood"
            / dataset_name
            / promoter_name
            / latent_name
            / method_name
        )

        probs = pickle.load((prediction.path / "probs.pkl").open("rb"))
        design = pickle.load((prediction.path / "design.pkl").open("rb"))

        probs_diff = probs - probs.mean(1, keepdims=True)

        # motifs
        motifscan_folder = (
            chd.get_output()
            / "motifscans"
            / dataset_name
            / promoter_name
            / motifscan_name
        )
        motifscan = chd.data.Motifscan(motifscan_folder)
        motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
        motifscan.n_motifs = len(motifs)
        motifs["ix"] = np.arange(motifs.shape[0])

        # gene oi
        gene_id = self.transcriptome.gene_id(symbol)
        gene_oi = self.transcriptome.gene_ix(symbol)

        # motifs oi
        if motifscan_name in ["cutoff_0001", "cutoff_001"]:
            if dataset_name == "pbmc10k":
                motifs_oi = pd.DataFrame(
                    [
                        [
                            motifs.loc[motifs.index.str.contains("SPI1")].index[0],
                            ["B", "Monocytes", "cDCs"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("CEBPB")].index[0],
                            ["Monocytes", "cDCs"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("PEBB")].index[0],
                            ["NK"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("RUNX2")].index[0],
                            ["NK"],
                        ],
                        # [
                        #     motifs.loc[motifs.index.str.contains("TBX21")].index[0],
                        #     ["NK"],
                        # ],
                        [
                            motifs.loc[motifs.index.str.contains("IRF8")].index[0],
                            ["cDCs"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("TFE2")].index[0],
                            ["B"],
                        ],  # TCF3
                        [
                            motifs.loc[motifs.index.str.contains("ITF2")].index[0],
                            ["pDCs"],
                        ],  # TCF4
                        [
                            motifs.loc[motifs.index.str.contains("BHA15")].index[0],
                            ["pDCs"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("BC11A")].index[0],
                            ["pDCs"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("PO2F2")].index[0],
                            ["B"],
                        ],
                        # [
                        #     motifs.loc[motifs.index.str.contains("PAX5")].index[0],
                        #     ["B"],
                        # ],
                        [
                            motifs.loc[motifs.index.str.contains("ATF4")].index[0],
                            ["Monocytes"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("RUNX1")].index[0],
                            ["CD4 T", "CD8 T", "MAIT"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("RUNX3")].index[0],
                            ["CD4 T", "CD8 T", "MAIT"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("GATA3")].index[0],
                            ["CD4 T", "CD8 T", "MAIT"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("TCF7")].index[0],
                            ["CD4 T", "CD8 T", "MAIT"],
                        ],
                    ],
                    columns=["motif", "clusters"],
                ).set_index("motif")
            elif dataset_name == "lymphoma":
                motifs_oi = pd.DataFrame(
                    [
                        # [motifs.loc[motifs.index.str.contains("SPI1")].index[0], ["Monocytes"]],
                        # [motifs.loc[motifs.index.str.contains("RUNX3")].index[0], ["T"]],
                        [
                            motifs.loc[motifs.index.str.contains("PO2F2")].index[0],
                            ["Lymphoma", "Lymphoma cycling"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("PO5F1")].index[0],
                            ["Lymphoma", "Lymphoma cycling"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("TFE2")].index[0],
                            ["Lymphoma", "Lymphoma cycling"],
                        ],
                        [
                            motifs.loc[motifs.index.str.contains("SNAI1")].index[0],
                            ["Lymphoma", "Lymphoma cycling"],
                        ],
                        # [motifs.loc[motifs.index.str.contains("PAX5")].index[0], ["Lymphoma", "Lymphoma cycling"]],
                    ],
                    columns=["motif", "clusters"],
                ).set_index("motif")
            motifs_oi["label"] = motifscan.motifs.loc[motifs_oi.index, "gene_label"]

        elif motifscan_name.startswith("gwas"):
            motifs_oi = pd.DataFrame(
                [
                    [x, [cluster_info.index[i]]]
                    for x, i in zip(
                        motifs.index,
                        itertools.chain(
                            range(len(cluster_info.index)),
                            range(len(cluster_info.index)),
                        ),
                    )
                ],
                columns=["motif", "clusters"],
            ).set_index("motif")

        if subset_clusters is not None:
            motifs_oi["clusters"] = motifs_oi["clusters"].apply(
                lambda x: [y for y in x if y in subset_clusters]
            )
            motifs_oi = motifs_oi.loc[motifs_oi["clusters"].apply(len) > 0]

        motifs_oi["ix"] = motifs.loc[motifs_oi.index, "ix"].values
        assert len(motifs_oi) == len(motifs_oi.index.unique())
        motifs_oi["color"] = sns.color_palette(n_colors=len(motifs_oi))
        if "label" not in motifs_oi:
            motifs_oi["label"] = motifs_oi.index

        # get motif data
        indptr_start = gene_oi * (window[1] - window[0])
        indptr_end = (gene_oi + 1) * (window[1] - window[0])

        motifdata = []
        for motif in motifs_oi.index:
            motif_ix = motifs.loc[motif, "ix"]
            for pos in range(indptr_start, indptr_end):
                pos_indices = motifscan.indices[
                    motifscan.indptr[pos] : motifscan.indptr[pos + 1]
                ]
                if motif_ix in pos_indices:
                    motifdata.append(
                        {"position": pos - indptr_start + window[0], "motif": motif}
                    )
        motifdata = pd.DataFrame(motifdata, columns=["position", "motif"])

        for motif_a, motif_b in motifs_to_merge:
            motifdata.loc[motifdata["motif"] == motif_a, "motif"] = motif_b
            motifs_oi = motifs_oi.loc[motifs_oi.index != motif_a]

        # promoter
        promoter = self.promoters.loc[gene_id]

        # empirical
        self.fragments.create_cut_data()

        plotdata_empirical = []
        for cluster_ix in cluster_info["dimension"]:
            fragments_oi = (
                latent_torch[self.fragments.cut_local_cell_ix, cluster_ix] != 0
            ) & (self.fragments.cut_local_gene_ix == gene_oi)
            cut_coordinates = self.fragments.cut_coordinates[fragments_oi].cpu().numpy()
            cut_coordinates = cut_coordinates * (window[1] - window[0]) + window[0]

            n_bins = 300
            cuts = np.linspace(*window, n_bins + 1)
            bincounts, bins = np.histogram(cut_coordinates, bins=cuts, range=window)
            binmids = bins[:-1] + (bins[:-1] - bins[1:]) / 2
            bindensity = (
                bincounts
                / cluster_info["lib"][cluster_ix]
                * self.fragments.obs["lib"].mean()
                * n_bins
            )

            plotdata_empirical.append(
                pd.DataFrame(
                    {
                        "binright": bins[1:],
                        "binleft": bins[:-1],
                        "count": bincounts,
                        "density": bindensity,
                        "cluster": cluster_ix,
                    }
                )
            )
        plotdata_empirical_bins = pd.concat(plotdata_empirical, ignore_index=True)

        plotdata_empirical = pd.DataFrame(
            {
                "coord": plotdata_empirical_bins[
                    ["binleft", "binright"]
                ].values.flatten(),
                "prob": np.log(plotdata_empirical_bins["density"].values.repeat(2)),
                "cluster": plotdata_empirical_bins["cluster"].values.repeat(2),
            }
        )

        baseline = np.log(
            plotdata_empirical.groupby(["cluster"]).apply(
                lambda x: np.trapz(
                    np.exp(x["prob"]),
                    x["coord"].astype(float) / (window[1] - window[0]),
                )
            )
        )

        plotdata_empirical["prob"] = (
            plotdata_empirical["prob"] - baseline[~np.isinf(baseline)].mean()
        )

        cluster_info["n_cells"] = (
            self.fragments.obs.groupby("cluster")
            .size()[cluster_info["dimension"]]
            .values
        )

        # expression
        plotdata_expression = sc.get.obs_df(
            self.transcriptome.adata, [gene_id, "cluster"]
        ).rename(columns={gene_id: "expression"})
        plotdata_expression_clusters = plotdata_expression.groupby("cluster")[
            "expression"
        ].mean()
        plotdata_diffexpression_clusters = (
            plotdata_expression_clusters - plotdata_expression_clusters.mean()
        )

        # atac (model)
        plotdata_atac = (
            design.query("gene_ix == @gene_oi")
            .copy()
            .rename(columns={"active_latent": "cluster"})
            .set_index(["coord", "cluster"])
            .drop(columns=["batch", "gene_ix"])
        )
        plotdata_atac["prob"] = probs[gene_oi].flatten()
        plotdata_atac["prob_diff"] = probs_diff[gene_oi].flatten()

        plotdata_atac["prob"] = (
            plotdata_atac["prob"]
            - np.log(
                plotdata_atac.reset_index()
                .groupby(["cluster"])
                .apply(
                    lambda x: np.trapz(
                        np.exp(x["prob"]),
                        x["coord"].astype(float) / (window[1] - window[0]),
                    )
                )
            ).mean()
        )
        plotdata_atac_mean = plotdata_atac[["prob"]].groupby("coord").mean()

        plotdata_genome = plotdata_atac
        plotdata_genome_mean = plotdata_atac_mean

        # plot
        main = chd.grid.Grid(3, 3, padding_width=0.1, padding_height=0.1)
        self.fig = chd.grid.Figure(main)

        padding_height = 0.001
        resolution = 0.0003
        panel_width = (window[1] - window[0]) * resolution

        panel_height = 0.5

        # gene annotation
        genome_folder = folder_data_preproc / "genome"
        ax_gene = main[0, 1] = chdm.plotting.Genes(
            promoter,
            genome_folder=genome_folder,
            window=window,
            width=panel_width,
        )

        # differential atac
        wrap_differential = main[1, 1] = chd.differential.plot.Differential(
            plotdata_genome,
            plotdata_genome_mean,
            cluster_info_oi,
            window,
            panel_width,
            panel_height,
            plotdata_empirical=plotdata_empirical,
            padding_height=padding_height,
            ymax=20,
        )

        # highlight motifs
        show_motifs = True
        if show_motifs:
            chd.differential.plot.MotifsHighlighting(
                wrap_differential, motifdata, motifs_oi, cluster_info_oi
            )
            wrap_motiflegend = main[1, 2] = chd.differential.plot.MotifsLegend(
                motifs_oi,
                cluster_info_oi,
                1,
                panel_height,
                padding_height=padding_height,
            )

        # expression
        show_expression = True
        if show_expression:
            wrap_expression = main[1, 0] = chd.differential.plot.DifferentialExpression(
                plotdata_expression,
                plotdata_expression_clusters,
                cluster_info_oi,
                0.3,
                panel_height,
                padding_height=padding_height,
            )

        # peaks
        show_peaks = True
        if show_peaks:
            peaks_folder = chd.get_output() / "peaks" / dataset_name
            peaks_panel = main.add_under(
                chdm.plotting.Peaks(
                    promoter,
                    peaks_folder,
                    window=window,
                    width=panel_width,
                    row_height=0.4,
                ),
                1,
            )
