import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

import chromatinhd as chd

device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# fragments
splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
outcome_source = "counts"
prediction_name = "v20_initdefault"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "100k100k", np.array([-100000, 100000])
# outcome_source = "magic"
# prediction_name = "v20_initdefault"

splitter = "permutations_5fold5repeat"
promoter_name, window = "10k10k", np.array([-10000, 10000])
outcome_source = "magic"
prediction_name = "v20"

promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# create design to run
from design import get_design, get_folds_inference

# folds & minibatching
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step=2000)
folds = folds  # [:1]

# design
from design import get_design, get_folds_training

design = get_design(transcriptome, fragments)

design_row = design[prediction_name]

fragments.window = window
design_row["loader_parameters"]["cellxgene_batch_size"] = cellxgene_batch_size
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

# loaders
if "loaders" in globals():
    loaders.terminate()
    del loaders
    import gc

    gc.collect()

loaders = chd.loaders.LoaderPool(
    design_row["loader_cls"],
    design_row["loader_parameters"],
    n_workers=20,
    shuffle_on_iter=False,
)

# load all models
models = [
    pickle.load(open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb"))
    for fold_ix, fold in enumerate(folds)
]
if outcome_source == "counts":
    outcome = transcriptome.X.dense()
else:
    outcome = torch.from_numpy(transcriptome.adata.layers["magic"])

scores_dir_overall = prediction.path / "scoring" / "overall"
transcriptome_predicted_full = pickle.load(
    (scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb")
)

folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step=2000)
folds = folds  # [:1]

# load nothing scoring
scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)

genes_all = fragments.var.index
genes_all_oi = fragments.var.index
genes_all_oi = transcriptome.var.query("symbol == 'TCF3'").index
genes_all_oi = transcriptome.var.index[
    (nothing_scoring.genescores.sel(phase="test").mean("model").mean("i")["cor"] > 0.1)
]

design = pd.DataFrame({"gene": genes_all_oi})
design["force"] = True

for gene, subdesign in design.groupby("gene", sort=False):
    genes_oi = genes_all == gene
    scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
    scores_folder.mkdir(exist_ok=True, parents=True)

    subdesign_row = subdesign.iloc[0]
    force = subdesign_row["force"] or not (scores_folder / "scores.nc").exists()

    if force:
        window_scoring = chd.scoring.prediction.Scoring.load(
            prediction.path / "scoring" / "window_gene" / gene
        )

        deltacor_cutoff = -0.0001
        windows_oi = window_scoring.design.loc[
            (
                window_scoring.genescores["deltacor"]
                .sel(gene=gene, phase=["test", "validation"])
                .mean("phase")
                .mean("model")
                < deltacor_cutoff
            ).values
        ]

        print(gene)
        folds_filtered, cellxgene_batch_size = get_folds_inference(
            fragments, folds, n_cells_step=2000, genes_oi=genes_oi
        )

        filterer = chd.scoring.prediction.filterers.WindowSizeAll(
            window, window_size=100
        )
        scorer = chd.scoring.prediction.Scorer2(
            models,
            folds_filtered[: len(models)],
            loaders,
            outcome,
            fragments.var.index[genes_oi],
            fragments.obs.index,
            device=device,
        )
        scoring = scorer.score(
            transcriptome_predicted_full=transcriptome_predicted_full,
            filterer=filterer,
            nothing_scoring=nothing_scoring,
        )

        scoring.save(scores_folder)
