import chromatinhd_manuscript as chdm
from chromatinhd.models.diff.plot import Peaks as PeaksBase
import pandas as pd
import pybedtools
import numpy as np


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


def get_usecols_and_names(peakcaller):
    if peakcaller in ["macs2_leiden_0.1"]:
        usecols = [0, 1, 2, 6]
        names = ["chr", "start", "end", "name"]
    else:
        usecols = [0, 1, 2]
        names = ["chr", "start", "end"]
    return usecols, names


def extract_peaks(peaks_bed, promoter, peakcaller):
    promoter_bed = pybedtools.BedTool.from_dataframe(
        pd.DataFrame(promoter).T[["chr", "start", "end"]]
    )

    usecols, names = get_usecols_and_names(peakcaller)
    peaks = promoter_bed.intersect(peaks_bed, wb=True, nonamecheck=True).to_dataframe(
        usecols=usecols, names=names
    )

    if peakcaller in ["macs2_leiden_0.1"]:
        peaks = peaks.rename(columns={"name": "cluster"})
        peaks["cluster"] = peaks["cluster"].astype(int)

    if len(peaks) > 0:
        peaks["peak"] = (
            peaks["chr"]
            + ":"
            + peaks["start"].astype(str)
            + "-"
            + peaks["end"].astype(str)
        )
        peaks = center_peaks(peaks, promoter)
        peaks = peaks.set_index("peak")
    else:
        peaks = pd.DataFrame({"start": [], "end": [], "method": [], "peak": []})
    return peaks


class Peaks(PeaksBase):
    def __init__(self, promoter, peaks_folder, *args, **kwargs):
        peaks = []

        import pybedtools

        peakcallers = []

        for peakcaller in [
            "cellranger",
            "macs2_improved",
            "macs2_leiden_0.1",
            "macs2_leiden_0.1_merged",
            "genrich",
            # "rolling_500",
            # "rolling_50",
            "encode_screen",
        ]:
            peaks_bed = pybedtools.BedTool(peaks_folder / peakcaller / "peaks.bed")

            peaks.append(
                extract_peaks(peaks_bed, promoter, peakcaller).assign(
                    peakcaller=peakcaller
                )
            )

            peakcallers.append({"peakcaller": peakcaller})

        peaks = pd.concat(peaks).reset_index().set_index(["peakcaller", "peak"])
        peaks["size"] = peaks["end"] - peaks["start"]

        peakcallers = pd.DataFrame(peakcallers).set_index("peakcaller")
        peakcallers = peakcallers.loc[
            peakcallers.index.isin(peaks.index.get_level_values("peakcaller"))
        ]
        peakcallers["ix"] = np.arange(peakcallers.shape[0])
        peakcallers = peakcallers.join(chdm.peakcallers)

        return super().__init__(*args, peaks=peaks, peakcallers=peakcallers, **kwargs)

        # fig, ax = plt.subplots(figsize=(2, 0.5))
        # ax.set_xlim(*window)
        # for i, (_, peak) in enumerate(peaks.query("peakcaller == @peakcaller").iterrows()):
        #     ax.plot([peak["start"], peak["end"]], [i, i])
