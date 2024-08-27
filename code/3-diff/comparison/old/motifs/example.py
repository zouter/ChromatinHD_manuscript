import pickle
import numpy as np
import pandas as pd
import chromatinhd as chd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import scanpy as sc
import chromatinhd_manuscript as chdm

import scipy.stats


def calculate_motifscore_expression_correlations(motifscores):
    if motifscores["expression_lfc"].std() == 0:
        slope_peak = 0
        r2_peak = 0
        slope_region = 0
        r2_region = 0
    else:
        linreg_peak = scipy.stats.linregress(
            motifscores["expression_lfc"], motifscores["logodds_peak"]
        )
        slope_peak = linreg_peak.slope
        r2_peak = linreg_peak.rvalue**2

        linreg_region = scipy.stats.linregress(
            motifscores["expression_lfc"], motifscores["logodds_region"]
        )
        slope_region = linreg_region.slope
        r2_region = linreg_region.rvalue**2

    if (r2_peak > 0) and (r2_region > 0):
        r2_diff = r2_region - r2_peak
    elif r2_region > 0:
        r2_diff = r2_region
    elif r2_peak > 0:
        r2_diff = -r2_peak
    else:
        r2_diff = 0.0

    cor_peak = np.corrcoef(motifscores["expression_lfc"], motifscores["logodds_peak"])[
        0, 1
    ]
    cor_region = np.corrcoef(
        motifscores["expression_lfc"], motifscores["logodds_region"]
    )[0, 1]
    cor_diff = cor_region - cor_peak

    #!
    motifscores_oi = motifscores

    contingency_peak = pd.crosstab(
        index=pd.Categorical(motifscores_oi["expression_lfc"] > 0, [False, True]),
        columns=pd.Categorical(motifscores_oi["logodds_peak"] > 0, [False, True]),
        dropna=False,
    )
    contingency_region = pd.crosstab(
        index=pd.Categorical(motifscores_oi["expression_lfc"] > 0, [False, True]),
        columns=pd.Categorical(motifscores_oi["logodds_region"] > 0, [False, True]),
        dropna=False,
    )

    odds_peak = scipy.stats.contingency.odds_ratio(contingency_peak).statistic
    odds_region = scipy.stats.contingency.odds_ratio(contingency_region).statistic

    return {
        "cor_peak": cor_peak,
        "cor_region": cor_region,
        "cor_diff": cor_diff,
        "r2_region": r2_region,
        "r2_diff": r2_diff,
        "slope_region": slope_region,
        "slope_peak": slope_peak,
        "slope_diff": slope_region - slope_peak,
        "logodds_peak": np.log(odds_peak),
        "logodds_region": np.log(odds_region),
        "logodds_difference": np.log(odds_region) - np.log(odds_peak),
    }


def plot_motifscore_expression_correlations(
    motifscores_oi, design_rows, motifs_oi, motif_ids_oi, ref_design_ix=None
):
    assert len(design_rows) > 0

    fig = polyptich.grid.Figure(polyptich.grid.Wrap(ncol=2))

    for design_ix in [*design_rows.index, "region"]:
        panel = fig.main.add(polyptich.grid.Panel((1.5, 1.5)))
        ax = panel.ax
        if design_ix == "region":
            if ref_design_ix is None:
                ref_design_ix = design_rows.index[0]
            motifscores_oi_row = motifscores_oi.query("design_ix == @ref_design_ix")
            title = "ChromatinHD"
            color = chdm.methods.methods.loc["ChromatinHD", "color"]
        else:
            design_row = design_rows.loc[design_ix]
            motifscores_oi_row = motifscores_oi.query("design_ix == @design_ix")
            title = chdm.methods.methods.loc[
                (design_row["peakcaller"] + "_" + design_row["diffexp"]), "label"
            ]
            color = chdm.methods.methods.loc[
                (design_row["peakcaller"] + "_" + design_row["diffexp"]), "color"
            ]
        if len(motifscores_oi_row) == 0:
            continue

        plotdata_scores = calculate_motifscore_expression_correlations(
            motifscores_oi_row
        )

        suffix = "_peak" if design_ix != "region" else "_region"

        ax.scatter(
            motifscores_oi_row["expression_lfc"],
            motifscores_oi_row["logodds" + suffix],
            s=1,
            color="grey",
        )

        plotdata_significant = motifscores_oi_row.query("expression_qval < 0.05")
        ax.scatter(
            plotdata_significant["expression_lfc"],
            plotdata_significant["logodds" + suffix],
            s=2,
            color="black",  # color,
        )

        # plot motifs oi
        plotdata_oi = motifscores_oi_row.query("motif in @motif_ids_oi")
        ax.scatter(
            plotdata_oi["expression_lfc"],
            plotdata_oi["logodds" + suffix],
            color="#333",
            marker="s",
            s=4,
        )
        ax.axvline(0, color="#333", dashes=(2, 2))
        ax.axhline(0, color="#333", dashes=(2, 2))
        cor = plotdata_scores["cor" + suffix]

        odds = np.exp(plotdata_scores["logodds" + suffix])
        ax.text(
            0.05,
            0.95,
            (
                f"r = {cor:.2f}"
                # +  f"\nodds = {odds:.1f}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

        texts = []
        for _, row in plotdata_oi.iterrows():
            label = motifs_oi.loc[row.name]["gene_label"]
            text = ax.text(
                row["expression_lfc"],
                row["logodds" + suffix],
                label,
                fontsize=8,
                ha="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0),
            )
            texts.append(text)

        # adjust_text but with a line between the text and the point
        import adjustText

        adjustText.adjust_text(
            texts, ax=ax, arrowprops=dict(arrowstyle="-", color="black")
        )

        ax.set_title(title, color=color)

    fig.main[0].ax.set_xlabel("Fold-change nuclear gene expression")
    import textwrap

    fig.main[0].ax.set_ylabel(
        "\n".join(
            textwrap.wrap(
                "Motif odds-ratio in\ndifferentially accessible positions", 20
            )
        ),
        rotation=0,
        ha="right",
        va="center",
    )

    return fig


def get_score_folder(x):
    return (
        chd.get_output()
        / "prediction_likelihood"
        / x.dataset
        / x.promoter
        / x.latent
        / str(x.method)
        / "scoring"
        / x.peakcaller
        / x.diffexp
        / x.motifscan
        / x.enricher
    )


class Example:
    def __init__(self, design_rows, cluster_oi, **kwargs):
        motifscores = []
        for design_ix, design_row in design_rows.iterrows():
            score_folder = get_score_folder(design_row)
            try:
                scores_peaks = pd.read_pickle(score_folder / "scores_peaks.pkl")
                scores_regions = pd.read_pickle(
                    score_folder / "scores_regions.pkl"
                ).assign(design_ix="region")

                motifscores.append(
                    pd.merge(
                        scores_peaks,
                        scores_regions,
                        on=["cluster", "motif"],
                        suffixes=("_peak", "_region"),
                        how="outer",
                    ).assign(design_ix=design_ix)
                )
            except FileNotFoundError:
                pass

        motifscores = (
            pd.concat(motifscores).reset_index().set_index(["cluster", "motif"])
        )

        dataset_name = design_row["dataset"]
        latent_name = design_row["latent"]
        motifscan_name = design_row["motifscan"]

        folder_data_preproc = chd.get_output() / "data" / dataset_name
        promoter_name = "10k10k"
        transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

        latent_folder = folder_data_preproc / "latent"
        latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

        cluster_info = pd.read_pickle(latent_folder / (latent_name + "_info.pkl"))
        transcriptome.obs["cluster"] = transcriptome.adata.obs[
            "cluster"
        ] = pd.Categorical(pd.from_dummies(latent).iloc[:, 0])

        motifscan_name = motifscan_name
        motifscan_folder = (
            chd.get_output()
            / "motifscans"
            / dataset_name
            / promoter_name
            / motifscan_name
        )
        motifscan = chd.data.Motifscan(motifscan_folder)

        sc.tl.rank_genes_groups(transcriptome.adata, "cluster", method="t-test")

        diffexp = sc.get.rank_genes_groups_df(transcriptome.adata, cluster_oi)
        diffexp = diffexp.set_index("names")

        motifs_oi = motifscan.motifs.loc[motifscan.motifs["gene"].isin(diffexp.index)]

        motifscores_oi = (
            motifscores.loc[cluster_oi]
            .loc[motifs_oi.index]
            .sort_values("logodds_region", ascending=False)
        )
        motifscores_oi["gene"] = motifs_oi.loc[
            motifscores_oi.index.get_level_values("motif"), "gene"
        ]
        motifscores_oi["expression_lfc"] = np.clip(
            diffexp.loc[motifscores_oi["gene"]]["logfoldchanges"].tolist(),
            -np.log(4),
            np.log(4),
        )
        motifscores_oi["expression_qval"] = diffexp.loc[motifscores_oi["gene"]][
            "pvals_adj"
        ].values

        motifscores_oi["logodds_diff"] = (
            motifscores_oi["logodds_peak"] - motifscores_oi["logodds_region"]
        )

        motifscores_oi.query("qval_region < 0.05").groupby("motif").mean().sort_values(
            "logodds_region", ascending=True
        ).head(20)[
            ["logodds_peak", "logodds_region", "expression_lfc", "logodds_diff"]
        ].reset_index().style.bar()

        # define motif ids oi
        motif_ids_oi = [
            # T lymphoma
            # motifs_oi.index[motifs_oi.index.str.startswith("RUNX3")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("JUNB")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("TCF7")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("MEF2C")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("PAX5")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("KLF12")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("COE1")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("THB")][0],
            # motifs_oi.index[motifs_oi.index.str.startswith("THAP1")][0],
        ]

        # base on logodds_region
        motif_ids_oi = [
            *(
                motifscores_oi.query("qval_region < 0.05")
                .groupby("motif")
                .mean()
                .sort_values("logodds_region", ascending=False)
                .index[:5]
            ),
            *(
                motifscores_oi.query("qval_region < 0.05")
                .groupby("motif")
                .mean()
                .sort_values("logodds_region", ascending=True)
                .index[:5]
            ),
        ]

        # based on multiplication of logodds_region and expression_lfc
        motifscores_oi["multiplication"] = (
            motifscores_oi["logodds_region"] * motifscores_oi["expression_lfc"]
        )
        motif_ids_oi = [
            *(
                motifscores_oi.query("qval_region < 0.05")
                .groupby("motif")
                .mean()
                .query("logodds_region > 0")
                .sort_values("multiplication", ascending=False)
                .index[:5]
            ),
            *(
                motifscores_oi.query("qval_region < 0.05")
                .groupby("motif")
                .mean()
                .query("logodds_region < 0")
                .sort_values("multiplication", ascending=False)
                .index[:5]
            ),
        ]

        # sc.pl.umap(transcriptome.adata, color=["cluster"])
        # sc.pl.umap(
        #     transcriptome.adata,
        #     color=motifs_oi.loc[motif_ids_oi]["gene"],
        #     title=motifs_oi.loc[motif_ids_oi]["gene_label"],
        # )

        motifscores_oi = motifscores_oi.query(
            "(qval_peak < 0.05) | (qval_region < 0.05)"
        )

        self.fig = plot_motifscore_expression_correlations(
            motifscores_oi,
            design_rows.query(
                "peakcaller in ['cellranger', 'rolling_100', 'macs2_leiden_0.1_merged']"
            ),
            ref_design_ix=design_rows.query("peakcaller == 'cellranger'").index[0],
            motifs_oi=motifs_oi,
            motif_ids_oi=motif_ids_oi,
        )
