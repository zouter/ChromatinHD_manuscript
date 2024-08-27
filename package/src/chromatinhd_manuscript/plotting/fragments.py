import chromatinhd
import polyptich.grid
import matplotlib as mpl
import matplotlib.collections
import seaborn as sns
import numpy as np
import pandas as pd


length_cmap = mpl.cm.get_cmap("viridis_r")
length_norm = mpl.colors.Normalize(vmin=0, vmax=700)


class Fragments(polyptich.grid.Ax):
    def __init__(
        self,
        coordinates,
        mapping,
        obs,
        window,
        width,
        height,
        connect=True,
    ):
        super().__init__((width, height))

        assert "y" in obs.columns, "y coordinate is missing in obs"

        ax = self.ax

        ax.set_xlim(*window)
        ax.set_ylim(0, obs["y"].max())

        connections = []
        markers = []

        colors = []

        for (start, end, cell_ix) in zip(
            coordinates[:, 0], coordinates[:, 1], mapping[:, 0]
        ):
            if start > window[1] or end < window[0]:
                continue

            connections.append(
                (start, obs.loc[cell_ix, "y"], end, obs.loc[cell_ix, "y"])
            )

            if connect:
                colors.append(length_cmap(length_norm(end - start)))
            else:
                colors.append("#333333AA")

            markers.append((start, obs.loc[cell_ix, "y"]))
            markers.append((end, obs.loc[cell_ix, "y"]))
        connections = np.array(connections)
        connections = connections.reshape(-1, 2, 2)
        markers = np.array(markers)

        if connect:
            ax.add_collection(
                mpl.collections.LineCollection(connections, colors=colors, lw=0.3)
            )
        ax.scatter(*markers.T, s=1, c=np.repeat(colors, 2, 0), zorder=10)
        ax.set_ylabel(f"Cells (n = {len(obs)})")
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.axvline(0, color="#333", linewidth=0.5, dashes=(2, 2))
