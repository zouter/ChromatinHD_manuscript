import chromatinhd_manuscript as chdm
from chromatinhd.data.peakcounts.plot import Peaks as PeaksBase
import pandas as pd
import pybedtools
import numpy as np
from chromatinhd_manuscript.peakcallers import peakcallers as peakcallers_all


def Peaks(region, peaks_folder, peakcallers=None, **kwargs):
    peakcallers = get_peakcallers(peaks_folder, peakcallers)

    return PeaksBase.from_bed(region, peakcallers, **kwargs)


def get_peakcallers(peaks_folder, peakcallers=None, add_rolling=False):
    if peakcallers is None:
        peakcaller_names = [
            "cellranger",
            "macs2_improved",
            "macs2_leiden_0.1",
            "macs2_leiden_0.1_merged",
            "macs2_summits",
            # "macs2_summit",
            "genrich",
            "encode_screen",
        ]
        if add_rolling:
            peakcaller_names += ["rolling_500"]
    else:
        peakcaller_names = peakcallers

    peakcallers = pd.DataFrame(
        [
            {"peakcaller": peakcaller, "path": peaks_folder / peakcaller / "peaks.bed"}
            for peakcaller in peakcaller_names
            if (peaks_folder / peakcaller / "peaks.bed").exists()
        ]
    ).set_index("peakcaller")
    peakcallers["label"] = peakcallers_all.loc[peakcallers.index, "label"]
    return peakcallers


# class Peaks(PeaksBase):
#     def __init__(self, region, peaks_folder, peakcallers=None, **kwargs):


#         # fig, ax = plt.subplots(figsize=(2, 0.5))
#         # ax.set_xlim(*window)
#         # for i, (_, peak) in enumerate(peaks.query("peakcaller == @peakcaller").iterrows()):
#         #     ax.plot([peak["start"], peak["end"]], [i, i])
