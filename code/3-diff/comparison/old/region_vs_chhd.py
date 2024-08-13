import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

method_info = pd.DataFrame(
    [
        ["peak", "#FF4136", f"Unique to differential CREs"],
        ["common", "#B10DC9", f"Common"],
        ["region", "#0074D9", f"Unique to ChromatinHD"],
    ],
    columns=["method", "color", "label"],
).set_index("method")


class RegionVsPeak:
    def __init__(self, regionresult, peakresult, window):
        self.regionresult = regionresult
        self.peakresult = peakresult

        self.window = window
        self.binmids = np.linspace(*window, 200 + 1)
        self.cuts = (self.binmids + (self.binmids[1] - self.binmids[0]) / 2)[:-1]

    def calculate_overlap(self):
        position_chosen_region = self.regionresult.position_chosen
        position_chosen_peak = self.peakresult.position_chosen

        position_region = np.where(position_chosen_region)[0]
        position_peak = np.where(position_chosen_peak)[0]
        position_indices_peak_unique = np.where(
            position_chosen_peak & (~position_chosen_region)
        )[0]
        position_indices_region_unique = np.where(
            position_chosen_region & (~position_chosen_peak)
        )[0]
        position_indices_common = np.where(
            position_chosen_region & position_chosen_peak
        )[0]
        position_indices_intersect = np.where(
            position_chosen_region | position_chosen_peak
        )[0]

        positions_region_unique = (
            position_indices_region_unique % (self.window[1] - self.window[0])
        ) + self.window[0]
        positions_region = (
            position_region % (self.window[1] - self.window[0])
        ) + self.window[0]

        positions_peak_unique = (
            position_indices_peak_unique % (self.window[1] - self.window[0])
        ) + self.window[0]
        positions_peak = (
            position_peak % (self.window[1] - self.window[0])
        ) + self.window[0]

        positions_common = (
            position_indices_common % (self.window[1] - self.window[0])
        ) + self.window[0]

        positions_intersect = (
            position_indices_intersect % (self.window[1] - self.window[0])
        ) + self.window[0]

        self.positions_region_unique_bincounts = np.bincount(
            np.digitize(positions_region_unique, self.cuts, right=True),
            minlength=len(self.cuts) + 1,
        )
        self.positions_region_bincounts = np.bincount(
            np.digitize(positions_region, self.cuts), minlength=len(self.cuts) + 1
        )

        self.positions_peak_unique_bincounts = np.bincount(
            np.digitize(positions_peak_unique, self.cuts), minlength=len(self.cuts) + 1
        )
        self.positions_peak_bincounts = np.bincount(
            np.digitize(positions_peak, self.cuts), minlength=len(self.cuts) + 1
        )

        self.positions_common_bincounts = np.bincount(
            np.digitize(positions_common, self.cuts), minlength=len(self.cuts) + 1
        )

        self.positions_intersect_bincounts = np.bincount(
            np.digitize(positions_intersect, self.cuts), minlength=len(self.cuts) + 1
        )

    def plot_overlap(self, ax=None, annotate=False):
        plotdata = pd.DataFrame(
            {
                "common": self.positions_common_bincounts,
                "peak_unique": self.positions_peak_unique_bincounts,
                "region_unique": self.positions_region_unique_bincounts,
                "intersect": self.positions_intersect_bincounts,
                "position": self.binmids,
            }
        )
        plotdata["peak_unique_density"] = (
            plotdata["peak_unique"] / plotdata["intersect"]
        )
        plotdata["common_density"] = plotdata["common"] / plotdata["intersect"]
        plotdata["region_unique_density"] = (
            plotdata["region_unique"] / plotdata["intersect"]
        )

        plotdata_last = (
            plotdata[["peak_unique_density", "common_density", "region_unique_density"]]
            .iloc[-10]
            .to_frame(name="density")
        )
        plotdata_last["cumulative_density"] = (
            np.cumsum(plotdata_last["density"]) - plotdata_last["density"] / 2
        )
        plotdata_mean = pd.Series(
            {
                "peak_unique_density": self.positions_peak_unique_bincounts.sum()
                / self.positions_intersect_bincounts.sum(),
                "region_unique_density": self.positions_region_unique_bincounts.sum()
                / self.positions_intersect_bincounts.sum(),
                "common_density": self.positions_common_bincounts.sum()
                / self.positions_intersect_bincounts.sum(),
            }
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        else:
            fig = ax.get_figure()
        ax.stackplot(
            self.binmids,
            plotdata["peak_unique_density"],
            plotdata["common_density"],
            plotdata["region_unique_density"],
            baseline="zero",
            colors=method_info["color"],
            lw=0,
            ec="#FFFFFF33",
        )
        ax.set_xlim(*self.window)
        ax.set_ylim(0, 1)

        if annotate:
            transform = mpl.transforms.blended_transform_factory(
                ax.transAxes, ax.transData
            )
            x = 1.02
            ax.text(
                x,
                plotdata_last.loc["peak_unique_density", "cumulative_density"],
                f"{method_info.loc['peak', 'label']}\n{plotdata_mean['peak_unique_density']:.0%}",
                transform=transform,
                ha="left",
                va="center",
                color=method_info.loc["peak", "color"],
            )
            ax.text(
                x,
                plotdata_last.loc["common_density", "cumulative_density"],
                f"{method_info.loc['common', 'label']}\n{plotdata_mean['common_density']:.0%}",
                transform=transform,
                ha="left",
                va="center",
                color=method_info.loc["common", "color"],
            )
            ax.text(
                x,
                plotdata_last.loc["region_unique_density", "cumulative_density"],
                f"{method_info.loc['region', 'label']}\n{plotdata_mean['region_unique_density']:.0%}",
                transform=transform,
                ha="left",
                va="center",
                color=method_info.loc["region", "color"],
            )

        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax.set_xlabel("Distance to TSS")
        return fig
