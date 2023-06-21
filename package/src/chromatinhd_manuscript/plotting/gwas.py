from chromatinhd.grid.broken import Broken, Panel
import pandas as pd
import adjustText
import seaborn as sns


class SNPsBroken(Panel):
    def __init__(
        self, plotdata, regions, width, transform, *args, gap=1, height=0.1, **kwargs
    ):
        super().__init__(
            (width, height),
            *args,
            **kwargs,
        )
        ax = self.ax

        plotdata["position_broken"] = transform(plotdata["position"].values)
        plotdata = plotdata.loc[~pd.isnull(plotdata["position_broken"])]

        snpmain_colors = {
            snpmain: color
            for snpmain, color in zip(
                plotdata["snp_main"].unique(), sns.color_palette("tab10")
            )
        }

        ax.scatter(
            plotdata["position_broken"],
            [0.3] * len(plotdata),
            marker="v",
            s=3,
            c=plotdata["snp_main"].map(snpmain_colors),
        )

        texts = []
        for i, (index, row) in enumerate(plotdata.iterrows()):
            texts.append(
                ax.text(
                    row["position_broken"],
                    0.8,
                    row["rsid"],
                    fontsize=5,
                    ha="center",
                    va="bottom",
                    color=snpmain_colors[row["snp_main"]],
                )
            )
        self.texts = texts
        ax.set_ylim(0, 1)
        ax.set_xlim(regions["cumend"].min(), regions["cumend"].max())
        ax.axis("off")
        ax.axis("off")
