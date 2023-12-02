import chromatinhd_manuscript as chdm
from chromatinhd.data.peakcounts.plot import Peaks as PeaksBase
import pandas as pd
import pybedtools
import numpy as np


def Peaks(region, peaks_folder, peakcallers=None, **kwargs):
    if peakcallers is None:
        peakcaller_names = [
            "cellranger",
            "macs2_improved",
            "macs2_leiden_0.1",
            "macs2_leiden_0.1_merged",
            "genrich",
            # "rolling_500",
            # "rolling_50",
            "encode_screen",
        ]
    else:
        peakcaller_names = peakcallers

    peakcallers = pd.DataFrame(
        [{"peakcaller": peakcaller, "path": peaks_folder / peakcaller / "peaks.bed"} for peakcaller in peakcaller_names]
    ).set_index("peakcaller")
    peakcallers["label"] = peakcallers.index

    return PeaksBase.from_bed(region, peakcallers, **kwargs)


# class Peaks(PeaksBase):
#     def __init__(self, region, peaks_folder, peakcallers=None, **kwargs):


#         # fig, ax = plt.subplots(figsize=(2, 0.5))
#         # ax.set_xlim(*window)
#         # for i, (_, peak) in enumerate(peaks.query("peakcaller == @peakcaller").iterrows()):
#         #     ax.plot([peak["start"], peak["end"]], [i, i])
