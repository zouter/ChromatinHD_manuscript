import chromatinhd.plot.genome.genes

import pandas as pd
import tqdm.auto as tqdm
import numpy as np
import scipy.stats


def calculate_joined(regionmultiwindow, data, genes_oi, regions, window_size=100, regionpairwindow=None):
    windows_oi = regionmultiwindow.design.loc[regionmultiwindow.design["window_size"] == window_size].index

    design = regionmultiwindow.design.loc[windows_oi]

    joined_all = []
    for gene_oi in tqdm.tqdm(genes_oi):
        region = regions.coordinates.loc[gene_oi]

        data_oi = data.loc[(data["chrom"] == region["chrom"]) & (data["gene"] == gene_oi)].copy()
        data_oi["start"] = data_oi["start"].astype(int)
        data_oi["end"] = data_oi["end"].astype(int)

        data_oi = data_oi.loc[data_oi["start"] > region["start"]]
        data_oi = data_oi.loc[data_oi["end"] < region["end"]]

        if data_oi.shape[0] > 0:
            data_centered = chromatinhd.plot.genome.genes.center(data_oi, region)
            data_centered["mid"] = (data_centered["start"] + data_centered["end"]) / 2

            data_centered["bin"] = design.index[np.digitize(data_centered["mid"], design["window_mid"]) - 1]
            data_binned = data_centered.groupby("bin").mean(numeric_only=True)

            joined = (
                regionmultiwindow.scores.sel_xr(
                    (gene_oi, slice(None), "test"), variables=["deltacor", "lost", "effect"]
                )
                .sel(window=design.index)
                .mean("fold")
                .to_pandas()
                .join(data_binned, how="left")
            )
            joined["window_mid"] = design.loc[joined.index, "window_mid"]
            joined["gene"] = gene_oi

            if regionpairwindow is not None:
                if gene_oi in regionpairwindow.interaction:
                    joined["interaction"] = 0.0
                    interaction_scores = regionpairwindow.interaction[gene_oi].mean("fold").mean("window1").to_pandas()
                    joined.loc[interaction_scores.index, "interaction"] = interaction_scores.values
                    joined["interaction_abs"] = 0.0
                    interaction_scores = (
                        np.abs(regionpairwindow.interaction[gene_oi].mean("fold")).mean("window1").to_pandas()
                    )
                    joined.loc[interaction_scores.index, "interaction_abs"] = interaction_scores.values
                    joined["deltacor2"] = joined["deltacor"].values
                    joined.loc[interaction_scores.index, "deltacor2"] = 0.0

            joined_all.append(joined)
    joined_all = pd.concat(joined_all)

    joined_all["deltacor_positive"] = np.where(
        (joined_all["effect"] > 0) & (joined_all["deltacor"] < 0.0), joined_all["deltacor"], 0
    )
    return joined_all


def rolling_max(x, w):
    if w == 0:
        return x
    y = []
    for i in range(w * 2):
        y.append(np.pad(x, (w * 2 - i - 1, i), mode="constant", constant_values=np.nan))
    y = np.stack(y)
    y = y[:, w : (y.shape[1] - w + 1)]
    z = np.nanmax(y, 0)
    # z = np.nanmean(y, 0)
    return z
