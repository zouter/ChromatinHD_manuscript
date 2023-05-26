import chromatinhd as chd
import cooler
import numpy as np
import pandas as pd
import itertools


def extract_hic(promoter, c=None, balance="VC_SQRT"):
    promoter_str = f"{promoter.chr}:{promoter.start}-{promoter.end}"
    if c is None:
        c = cooler.Cooler(
            str(chd.get_output() / "4DNFIXP4QG5B.mcool") + "::/resolutions/1000"
        )

    if balance is not False:
        hic = c.matrix(balance=balance, as_pixels=True, join=True).fetch(promoter_str)
    else:
        hic = c.matrix(balance=False, as_pixels=True, join=True).fetch(promoter_str)
        hic["balanced"] = hic["count"]

    # hicmat = (
    #     hic.set_index(["start1", "start2"])["balanced"].unstack().fillna(0)
    #     + hic.set_index(["start1", "start2"])["balanced"].unstack().fillna(0).T.values
    # )

    # fig, ax = plt.subplots()
    # sns.heatmap(np.log1p(hicmat), cmap="viridis", ax=ax)

    if promoter["strand"] == 1:
        hic[["start1", "end1", "start2", "end2"]] = hic[
            ["start1", "end1", "start2", "end2"]
        ].apply(lambda x: (x - promoter.tss) * promoter["strand"])
    else:
        hic[["end1", "start1", "end2", "start2"]] = hic[
            ["start1", "end1", "start2", "end2"]
        ].apply(lambda x: (x - promoter.tss) * promoter["strand"])

    bins_hic = (
        pd.concat(
            [
                (
                    hic.groupby(["start1", "end1"])[[]]
                    .first()
                    .reset_index()
                    .rename(columns={"start1": "start", "end1": "end"})
                ),
                (
                    hic.groupby(["start2", "end2"])[[]]
                    .first()
                    .reset_index()
                    .rename(columns={"start2": "start", "end2": "end"})
                ),
            ]
        )
        .drop_duplicates()
        .sort_index()
    )
    bins_hic["window"] = bins_hic["start"]
    bins_hic = bins_hic.set_index("window")

    hic["window1"] = (
        bins_hic.reset_index().set_index("start").loc[hic["start1"]]["window"].values
    )
    hic["window2"] = (
        bins_hic.reset_index().set_index("start").loc[hic["start2"]]["window"].values
    )

    hic = pd.concat(
        [
            hic.reset_index().set_index(["window1", "window2"]),
            hic.reset_index()
            .rename(columns={"window1": "window2", "window2": "window1"})
            .set_index(["window1", "window2"]),
        ]
    )
    hic = hic.groupby(["window1", "window2"]).first()

    return hic, bins_hic


def match_windows(window, bins_hic):
    window_ref, start, end = (
        bins_hic.index.values,
        bins_hic["start"].values,
        bins_hic["end"].values,
    )
    match = (window[:, None] > start[None, :]) & (window[:, None] <= end[None, :])
    assert (match.sum(1) <= 1).all(), print(match)
    output = window_ref[match.argmax(1)].astype(float)
    # output[match.sum(1) == 0] = np.nan
    return output


def fix_hic(hic, bins_hic):
    if "window1" not in hic.index.names:
        hic["window1"] = hic["start1"].values
        hic["window2"] = hic["start2"].values
        hic = hic.set_index(["window1", "window2"])

    if "window" != bins_hic.index.name:
        bins_hic = bins_hic.set_index("window")

    hic = pd.concat(
        [
            hic.reset_index().set_index(["window1", "window2"]),
            hic.reset_index()
            .rename(columns={"window1": "window2", "window2": "window1"})
            .set_index(["window1", "window2"]),
        ]
    )
    hic = hic.groupby(["window1", "window2"]).first()

    return hic, bins_hic


def create_matching(
    bins_hic,
    scores_chd,
    hic,
):
    matching_hic = hic
    # matching_hic = hic.groupby(["window1", "window2"]).first()

    matching_chd = (
        scores_chd.groupby(["hicwindow1", "hicwindow2"])
        .agg({"cor": lambda x: max(x, key=abs)})
        .reset_index()
        .rename(columns={"hicwindow1": "window1", "hicwindow2": "window2"})
        .set_index(["window1", "window2"])
    )

    matching = (
        pd.DataFrame(
            itertools.product(bins_hic.index, bins_hic.index),
            columns=["window1", "window2"],
        )
        .set_index(["window1", "window2"])
        .groupby(["window1", "window2"])
        .first()
    )
    matching["balanced"] = matching_hic.reindex(matching.index)["balanced"].fillna(0)
    matching["cor"] = matching_chd.reindex(matching.index)["cor"].fillna(0)
    matching["distance"] = np.abs(
        matching.index.get_level_values("window1")
        - matching.index.get_level_values("window2")
    )

    return matching


def pool_prepare_hic(hic, bins_hic, distance_cutoff=1001):
    hic["distance"] = np.abs(
        hic.index.get_level_values("window1").astype(float)
        - hic.index.get_level_values("window2").astype(float)
    )

    x_distance = (
        hic["distance"]
        .unstack()
        .reindex(index=bins_hic.index, columns=bins_hic.index)
        .fillna(0)
    )
    assert (x_distance.index == x_distance.columns).all()

    x = (
        hic["balanced"]
        .unstack()
        .reindex(index=bins_hic.index, columns=bins_hic.index)
        .fillna(0)
    )
    x = x.values + x.T.values
    x[x_distance.values <= distance_cutoff] = 0
    return x


import scipy.ndimage


def maxipool_hic(hic, bins_hic, distance_cutoff=1000, k=1):
    x = pool_prepare_hic(hic, bins_hic, distance_cutoff=distance_cutoff)

    footprint = np.ones((k * 2 + 1, k * 2 + 1))

    x2 = pd.DataFrame(
        scipy.ndimage.maximum_filter(x, footprint=footprint),
        index=bins_hic.index.copy(),
        columns=bins_hic.index.copy(),
    )
    x2.index.name = "window1"
    x2.columns.name = "window2"
    hic2 = x2.stack().to_frame().rename(columns={0: "balanced"})
    hic2["distance"] = np.abs(
        hic2.index.get_level_values("window1").astype(float)
        - hic2.index.get_level_values("window2").astype(float)
    )
    hic2.loc[hic2["distance"] <= 1000, "balanced"] = 0.0
    return hic2


def meanpool_hic(hic, bins_hic, distance_cutoff=1000, k=1):
    x = pool_prepare_hic(hic, bins_hic, distance_cutoff=distance_cutoff)

    footprint = np.ones((k * 2 + 1, k * 2 + 1))

    x2 = pd.DataFrame(
        scipy.ndimage.generic_filter(x, np.mean, footprint=footprint),
        index=bins_hic.index.copy(),
        columns=bins_hic.index.copy(),
    )
    x2.index.name = "window1"
    x2.columns.name = "window2"
    hic2 = x2.stack().to_frame().rename(columns={0: "balanced"})
    hic2["distance"] = np.abs(
        hic2.index.get_level_values("window1").astype(float)
        - hic2.index.get_level_values("window2").astype(float)
    )
    hic2.loc[hic2["distance"] <= 1000, "balanced"] = 0.0
    return hic2
