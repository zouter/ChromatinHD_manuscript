import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import seaborn as sns
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import chromatinhd_manuscript as chdm
import itertools


def center_peaks(peaks, promoter):
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns=["start", "end", "method"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] - promoter["tss"]) * promoter["strand"],
                (peak["end"] - promoter["tss"]) * promoter["strand"],
            ][:: promoter["strand"]]
            for _, peak in peaks.iterrows()
        ]
    return peaks


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
        show_motifs=False,
        **kwargs
    ):
        folder_root = chd.get_output()
        folder_data = folder_root / "data"

        folder_data_preproc = folder_data / dataset_name

        self.dataset_name = dataset_name
        self.symbol = symbol

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

        # gene oi
        gene_id = self.transcriptome.gene_id(symbol)
        gene_oi = self.gene_ix = self.transcriptome.gene_ix(symbol)

        self.promoter = self.promoters.loc[gene_id]

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
            self.cluster_info_oi = cluster_info.loc[subset_clusters].copy()
        else:
            self.cluster_info_oi = cluster_info.copy()
        self.cluster_info_oi["ix"] = np.arange(len(self.cluster_info_oi))

        # prediction
        prediction = chd.flow.Flow(
            chd.get_output()
            / "prediction_positional"
            / dataset_name
            / promoter_name
            / "permutations_5fold5repeat"
            / "v20"
        )
        scores_folder = prediction.path / "scoring" / "multiwindow_gene" / gene_id
        interpolated = pickle.load((scores_folder / "interpolated.pkl").open("rb"))

        plotdata_predictive = pd.DataFrame(
            {
                "deltacor": interpolated["deltacor_test"].mean(0),
                "lost": interpolated["lost"].mean(0),
                "position": pd.Series(np.arange(*window), name="position"),
            }
        )

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
        motifscan = self.motifscan = chd.data.Motifscan(motifscan_folder)
        motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
        motifscan.n_motifs = len(motifs)
        motifs["ix"] = np.arange(motifs.shape[0])

        # motifs oi
        if show_motifs:
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
                                motifs.loc[motifs.index.str.contains("IRF1")].index[0],
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
                            # [
                            #     motifs.loc[motifs.index.str.contains("PO2F2")].index[0],
                            #     ["B"],
                            # ],
                            [
                                motifs.loc[motifs.index.str.contains("COE1")].index[0],
                                ["B"],
                            ],
                            [
                                motifs.loc[motifs.index.str.contains("SNAI1")].index[0],
                                ["B"],
                            ],
                            [
                                motifs.loc[motifs.index.str.contains("PAX5")].index[0],
                                ["B"],
                            ],
                            [
                                motifs.loc[motifs.index.str.contains("IRF4")].index[0],
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

            # merge motifs
            for motif_a, motif_b in motifs_to_merge:
                motifdata.loc[motifdata["motif"] == motif_a, "motif"] = motif_b
                motifs_oi = motifs_oi.loc[motifs_oi.index != motif_a]

            self.motifdata = motifdata

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
        self.panel_width = (window[1] - window[0]) * resolution

        panel_height = 0.5

        # gene annotation
        genome_folder = folder_data_preproc / "genome"
        ax_gene = main[0, 1] = chdm.plotting.Genes(
            self.promoter,
            genome_folder=genome_folder,
            window=window,
            width=self.panel_width,
        )

        # predictive
        predictive_panel = main[1, 1] = chd.predictive.plot.Predictive(
            plotdata_predictive,
            window,
            self.panel_width,
        )

        # differential atac
        self.wrap_differential = main[2, 1] = chd.models.diff.plot.Differential(
            plotdata_genome,
            plotdata_genome_mean,
            self.cluster_info_oi,
            window,
            self.panel_width,
            panel_height,
            # plotdata_empirical=plotdata_empirical,
            padding_height=padding_height,
            ymax=20,
        )

        # highlight motifs
        if show_motifs:
            chd.models.diff.plot.MotifsHighlighting(
                self.wrap_differential, motifdata, motifs_oi, self.cluster_info_oi
            )
            wrap_motiflegend = main[2, 2] = chd.models.diff.plot.MotifsLegend(
                motifs_oi,
                self.cluster_info_oi,
                1,
                panel_height,
                padding_height=padding_height,
            )

        # expression
        show_expression = True
        if show_expression:
            wrap_expression = main[2, 2] = chd.models.diff.plot.DifferentialExpression(
                plotdata_expression,
                plotdata_expression_clusters,
                self.cluster_info_oi,
                0.3,
                panel_height,
                padding_height=padding_height,
                symbol=symbol,
            )

        # peaks
        show_peaks = True
        if show_peaks:
            peaks_folder = chd.get_output() / "peaks" / dataset_name
            peaks_panel = main.add_under(
                chdm.plotting.Peaks(
                    self.promoter,
                    peaks_folder,
                    window=window,
                    width=self.panel_width,
                    row_height=0.4,
                ),
                column=self.wrap_differential,
            )

    def add_bigwig(self, design, show_peaks=False):
        # check design
        design["ix"] = np.arange(len(design))

        if "file_bed" not in design.columns:
            design["file_bed"] = None
            show_peaks = False
        if "file" not in design.columns:
            design["file"] = None
        if "gene" not in design.columns:
            design["gene"] = None

        # add grid for different tracks
        grid = self.fig.main.add_under(
            chd.grid.Grid(padding_width=0.25), column=self.wrap_differential
        )

        window = [
            self.promoter.start - self.promoter.tss,
            self.promoter.end - self.promoter.tss,
        ]

        # get motif data
        indptr_start = self.gene_ix * (window[1] - window[0])
        indptr_end = (self.gene_ix + 1) * (window[1] - window[0])
        motif_indices = self.motifscan.indices[
            self.motifscan.indptr[indptr_start] : self.motifscan.indptr[indptr_end]
        ]
        position_indices = chd.utils.indptr_to_indices(
            self.motifscan.indptr[indptr_start : indptr_end + 1]
        )

        # get motifs oi
        motifs_oi = []
        for motif_identifiers in design["motif_identifiers"]:
            for motif_identifier in motif_identifiers:
                motifs_oi.append(
                    {
                        "motif": self.motifscan.motifs.index[
                            self.motifscan.motifs.index.str.contains(motif_identifier)
                        ][0],
                        "motif_identifier": motif_identifier,
                    }
                )
        motifs_oi = pd.DataFrame(motifs_oi).set_index("motif")

        self.motifscan.motifs["ix"] = np.arange(self.motifscan.motifs.shape[0])
        motifdata = []
        for motif in motifs_oi.index:
            motif_ix = self.motifscan.motifs.loc[motif, "ix"]
            positions_oi = position_indices[motif_indices == motif_ix]
            motifdata.extend(
                [{"position": pos + window[0], "motif": motif} for pos in positions_oi]
            )
        motifdata = self.motifdata = pd.DataFrame(motifdata)

        import pyBigWig

        first = True
        for bigwig_info in design.itertuples():
            panel, ax = grid.add_under(
                chd.grid.Panel((self.panel_width, 0.1)), padding=0
            )

            # keep track of scores of positions, so we can color motifs
            position_scores = pd.DataFrame(
                {
                    "position": np.arange(*window),
                    "value": np.nan,
                }
            )

            ax.set_ylim(0, 15)
            if first:
                if not pd.isnull(bigwig_info.file):
                    ax.set_ylabel(
                        "ChIP-seq\n fold over control",
                        rotation=0,
                        ha="right",
                        va="center",
                    )
                    ax.set_yticks([0, 15])
                else:
                    ax.set_yticks([])
                    ax.set_ylabel("Motifs", rotation=0, ha="right", va="center")
                first = False
            else:
                ax.set_yticks([])
            ax.set_xlim(*window)
            ax.set_xticks([])

            ax.annotate(
                bigwig_info.tf,
                (1.0, 0.5),
                xycoords="axes fraction",
                xytext=(2, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                ha="left",
            )

            # add value (fold-change over control)
            if not pd.isnull(bigwig_info.file):
                panel.dim = tuple([panel.dim[0], 0.2])
                if not bigwig_info.file.exists():
                    raise ValueError(bigwig_info.file, "does not exist")
                bw = pyBigWig.open(str(bigwig_info.file))

                plotdata_bw = pd.DataFrame(
                    {
                        "value": np.array(
                            bw.values(
                                self.promoter.chr,
                                self.promoter.start,
                                self.promoter.end,
                            )
                        )[:: self.promoter.strand],
                        "position": np.arange(*window),
                    }
                )
                ax.plot(
                    plotdata_bw["position"], plotdata_bw["value"], color="#555", lw=0.8
                )
                position_scores["value"] = plotdata_bw["value"].values

            # add peaks
            if not pd.isnull(bigwig_info.file_bed):
                panel.dim = tuple([panel.dim[0], 0.2])
                import pybedtools

                if not bigwig_info.file_bed.exists():
                    raise ValueError(bigwig_info.file_bed, "does not exist")

                bed = pybedtools.BedTool(bigwig_info.file_bed)
                bed_promoter = pybedtools.BedTool(
                    "\t".join(
                        [
                            self.promoter.chr,
                            str(self.promoter.start),
                            str(self.promoter.end),
                        ]
                    ),
                    from_string=True,
                )

                bed_intersect = bed.intersect(
                    bed_promoter,
                    u=True,
                ).to_dataframe()

                bed_intersect = center_peaks(bed_intersect, self.promoter)

                for _, peak in bed_intersect.iterrows():
                    ax.axvspan(peak["start"], peak["end"], color="black", alpha=0.5)

                    position_scores.loc[
                        (position_scores["position"] < peak["end"])
                        & (position_scores["position"] >= peak["start"]),
                        "value",
                    ] = 10

            # add motifs
            if bigwig_info.motif_identifiers is not None:
                motif_ids = motifs_oi.index[
                    motifs_oi["motif_identifier"].isin(bigwig_info.motif_identifiers)
                ]
                plotdata = motifdata.loc[motifdata["motif"].isin(motif_ids)].copy()

                if len(plotdata) > 0:
                    plotdata["value"] = (
                        position_scores.set_index("position")
                        .loc[plotdata["position"], "value"]
                        .values
                    )

                    def determine_color(value):
                        if pd.isnull(value):
                            return "#333"
                        elif value > 2:
                            return "tomato"
                        else:
                            return "grey"

                    plotdata["color"] = plotdata["value"].apply(determine_color)
                    ax.scatter(
                        plotdata["position"],
                        [1] * len(plotdata),
                        transform=mpl.transforms.blended_transform_factory(
                            ax.transData, ax.transAxes
                        ),
                        marker="v",
                        color=plotdata["color"],
                        alpha=1,
                        s=100,
                        zorder=20,
                    )

            if bigwig_info.gene is not None:
                plotdata_expression = sc.get.obs_df(
                    self.transcriptome.adata, [bigwig_info.gene, "cluster"]
                ).rename(columns={bigwig_info.gene: "expression"})
                plotdata_expression_clusters = plotdata_expression.groupby("cluster")[
                    "expression"
                ].mean()
                plotdata_expression_clusters_oi = plotdata_expression_clusters.loc[
                    self.cluster_info_oi.index
                ]

                panel_expression, ax_expression = panel.add_inset(
                    chd.grid.Panel((0.1 * len(plotdata_expression_clusters_oi), 0.1)),
                    pos=(1, 0),
                    offset=(0, 0),
                    anchor=(0, 0.5),
                )

                ax_expression.set_xlim(0, len(plotdata_expression_clusters_oi))
                ax_expression.set_ylim(0, 1)

                cmap_expression = chd.models.diff.plot.get_cmap_rna_diff()
                norm_expression = mpl.colors.Normalize(
                    vmin=0,
                    vmax=plotdata_expression_clusters.max(),
                )

                for cluster, expression in plotdata_expression_clusters_oi.iteritems():
                    x = self.cluster_info_oi.loc[cluster, "ix"] + 0.5
                    circle = mpl.patches.Circle(
                        (x, 0.5),
                        norm_expression(expression) * 0.5,
                        fc=cmap_expression(norm_expression(expression)),
                        lw=1,
                        ec="#333333",
                        zorder=10,
                        clip_on=False,
                    )
                    ax_expression.add_patch(circle)
                    ax_expression.axis("off")
