import pandas as pd
import numpy as np
import torch

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


print(torch.cuda.memory_allocated(0))


def transform_parameters(mu, dispersion, eps=1e-8):
    # avoids NaNs induced in gradients when mu is very low
    dispersion = np.clip(dispersion, 0, 100.0)

    logits = np.log(mu + eps) - np.log(1 / dispersion + eps)

    total_count = 1 / dispersion

    return total_count, logits


class TranscriptomeSimulated:
    def __init__(self, fragments, transcriptome, noisy_atac=True, noisy_expression=True, atac_modifier=1):
        self.fragments = fragments
        self.transcriptome = transcriptome

        mu = (fragments.counts + 1e-5) / (fragments.counts.mean(0, keepdims=True) + 1e-5)
        mu_expression = mu * transcriptome.var["mean"].values
        mu_atac = mu / mu.mean(0) * fragments.counts.mean(0)
        mu_atac_baseline = mu_atac.mean(0)[None, :].repeat(mu_atac.shape[0], 0)
        dispersion = transcriptome.var["dispersion_mean"].values

        total_count, logits = transform_parameters(mu_expression, dispersion)
        probs = 1 / (1 + np.exp(logits))

        self.total_count = total_count
        self.probs = probs

        if noisy_expression:
            self.expression = np.log1p(np.random.negative_binomial(total_count, probs))
        else:
            self.expression = np.log1p(mu_expression)
        if noisy_atac:
            self.atac = np.random.poisson(mu_atac * atac_modifier) + np.random.poisson(
                mu_atac_baseline * (1 - atac_modifier)
            )
        else:
            self.atac = mu_atac * atac_modifier

        self.layers = {"X": self.expression}


class Simulation:
    layer = None

    def __init__(self, transcriptome):
        self.transcriptome = transcriptome

    def get_prediction(self, fragments, transcriptome, cell_ixs):
        regions = fragments.var.index
        result = xr.Dataset(
            {
                "predicted": xr.DataArray(
                    self.transcriptome.atac[cell_ixs],
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cell_ixs, fragments.var.index.name: regions},
                ),
            }
        )
        return result


from chromatinhd_manuscript.designs_pred import dataset_folds_simulation_combinations as design

design = design.copy()
design["force"] = False

for (dataset_name, regions_name), subdesign in design.groupby(["dataset", "regions"]):
    dataset_folder = chd.get_output() / "datasets" / dataset_name

    fragments = chromatinhd.data.Fragments(dataset_folder / "fragments" / regions_name)
    transcriptome = chromatinhd.data.Transcriptome(dataset_folder / "transcriptome")

    for (folds_name, layer), subdesign in subdesign.groupby(["folds", "layer"]):
        # folds & minibatching
        folds = chd.data.folds.Folds(dataset_folder / "folds" / folds_name)

        for method_name, subdesign in subdesign.groupby("method"):
            method_params = subdesign["params"].iloc[0]
            prediction = chd.flow.Flow(
                chd.get_output() / "pred" / dataset_name / regions_name / folds_name / layer / method_name
            )

            performance = chd.models.pred.interpret.Performance(prediction.path / "scoring" / "performance")

            force = subdesign["force"].iloc[0]
            if not performance.scored:
                force = True

            if force:
                print("scoring", prediction.path)
                transcriptome_simulated = TranscriptomeSimulated(fragments, transcriptome, **method_params)
                simulation = Simulation(transcriptome_simulated)

                performance_simulation = performance.score(
                    fragments, transcriptome_simulated, [simulation] * len(folds), folds
                )
