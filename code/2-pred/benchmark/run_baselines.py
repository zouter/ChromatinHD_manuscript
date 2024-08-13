import pandas as pd
import numpy as np
import torch
import torch_scatter

import chromatinhd as chd
import chromatinhd.data
import matplotlib.pyplot as plt

import pickle
import xarray as xr

device = "cuda:0"

folder_root = chd.get_output()
folder_data = folder_root / "data"


class Prediction(chd.flow.Flow):
    pass


class Baseline:
    def __init__(self, fragments, transcriptome, selected_transcripts, layer="X"):
        self.transcriptome = transcriptome
        self.fragments = fragments

        gene_bodies = selected_transcripts.loc[transcriptome.var.index]

        gene_bodies["end_relative"] = ((gene_bodies["end"] - gene_bodies["start"]) * (gene_bodies["strand"] == 1)) - (
            (gene_bodies["start"] - gene_bodies["end"]) * (gene_bodies["strand"] == -1)
        )

        self.gene_bodies = gene_bodies

        self.layer = layer

        self.regions_oi = self.transcriptome.var.sort_values("dispersions_norm", ascending=False).index

        self.loader = chd.loaders.transcriptome_fragments.TranscriptomeFragments(
            self.fragments, self.transcriptome, len(fragments.obs) * 1, layer=layer
        )

    def fitted(self, region, fold_ix):
        return True

    def get_prediction(self, region, fold_ix, cell_ixs, return_raw=False, fragments=None, transcriptome=None):
        region_ix = self.fragments.var.index.get_loc(region)
        minibatch = chd.loaders.minibatches.Minibatch(cell_ixs, np.array([region_ix]))
        data = self.loader.load(minibatch)

        predicted = self._predict(data, minibatch)
        expected = data.transcriptome.value.numpy()
        n_fragments = np.zeros_like(predicted)

        if return_raw:
            return predicted, expected, n_fragments
        raise ValueError()

    def _predict(self, data, minibatch):
        raise NotImplementedError()


class V42(Baseline):
    def _predict(self, data, minibatch):
        coord = data.fragments.coordinates.numpy().mean(1)
        dist_tss = coord
        dist_tst = coord - self.gene_bodies["end"].values[data.fragments.regionmapping]
        dist = np.where(abs(dist_tss) < abs(dist_tst), dist_tss, dist_tst)
        within = ~(dist_tss < -5000) & ~(dist_tst > 5000)

        weight = (1 + np.e ** (-1)) * within + (np.exp(-abs(dist / 5000)) + np.e ** (-1)) * ~within

        weight = torch.tensor(weight)

        cellxregion_ix = data.fragments.local_cellxregion_ix
        predicted = (
            torch_scatter.segment_sum_coo(
                weight, cellxregion_ix.long(), dim_size=len(minibatch.cells_oi) * len(minibatch.regions_oi)
            )
            .reshape(len(minibatch.cells_oi), len(minibatch.regions_oi))
            .numpy()
        )
        return predicted


class V21(Baseline):
    def _predict(self, data, minibatch):
        coord = data.fragments.coordinates.float().mean(1)
        within = (coord > -5000) & (coord < 5000)
        weight = (1 + np.e ** (-1)) * within + (np.exp(-abs(coord / 5000)) + np.e ** (-1)) * ~within

        cellxregion_ix = data.fragments.local_cellxregion_ix
        predicted = (
            torch_scatter.segment_sum_coo(
                weight, cellxregion_ix.long(), dim_size=len(minibatch.cells_oi) * len(minibatch.regions_oi)
            )
            .reshape(len(minibatch.cells_oi), len(minibatch.regions_oi))
            .numpy()
        )
        return predicted


method2cls = {
    "baseline_v21": V21,
    "baseline_v42": V42,
}

from chromatinhd_manuscript.designs_pred import dataset_baseline_combinations as design

design = design.copy()
design["force"] = False

# design = design.query("dataset in ['liver']")
# design = design.query("dataset in ['e18brain', 'lymphoma']")
# design = design.query("regions == '10k10k'")

for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    dataset_folder = chd.get_output() / "datasets" / dataset_name
    folder_data_preproc = chd.get_output() / "data" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

    for (splitter, layer), subdesign in subdesign.groupby(["splitter", "layer"]):
        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / splitter)

        for method_name, subdesign in subdesign.groupby("method"):
            prediction = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / splitter / layer / method_name
            )

            performance = chd.models.pred.interpret.Performance.create(
                path=prediction.path / "scoring" / "performance",
                folds=folds,
                transcriptome=transcriptome,
                fragments=fragments,
                overwrite=False,
            )

            force = subdesign["force"].iloc[0]
            if not performance.scores["scored"].sel_xr().all():
                force = True

            if force:
                print("scoring", prediction.path)

                selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb"))
                print(folder_data_preproc)
                models = method2cls[method_name](fragments, transcriptome, selected_transcripts, layer=layer)

                performance.score(models)

                # performance_simulation = performance.score(fragments, transcriptome, [model] * len(folds), folds)
