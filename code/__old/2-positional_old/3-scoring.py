# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import scanpy as sc
import pathlib

import torch

import tqdm.auto as tqdm

# %%
import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.loaders.minibatching
import peakfreeatac.scorer

import pickle

device = "cuda:0"
# device = "cpu"

folder_root = pfa.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = pfa.data.Transcriptome(
    folder_data_preproc / "transcriptome"
)

# fragments
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoter_name, window = "20kpromoter", np.array([-10000, 0])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = pfa.data.Fragments(
    folder_data_preproc / "fragments" / promoter_name
)

# create design to run
from design import get_design, get_folds_inference

class Prediction(pfa.flow.Flow):
    pass

# folds & minibatching
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
folds = get_folds_inference(fragments, folds)

# design
from design import get_design, get_folds_training
design = get_design(dataset_name, transcriptome, fragments, window = window)
design = {k:design[k] for k in [
    "v14",
    "v14_dummy",
    # "v14_5freq"
    # "v14_3freq",
    # "v14_20freq",
    # "v14_50freq",
    "v14_50freq_sum",
    # "v14_50freq_linear",
    # "v14_50freq_sigmoid",
    "v14_50freq_sum_sigmoid",
    "v14_50freq_sum_sigmoid_initdefault"
]}
fold_slice = slice(0, 10)

# %%
Scorer = pfa.scorer.Scorer

# %%
prediction_name = "v14"
# prediction_name = "v14_50freq_sum"
# prediction_name = "v14_50freq_sum_sigmoid"
prediction_name = "v14_50freq_sum_sigmoid_initdefault"
design_row = design[prediction_name]

# %%
import torch_scatter
class FilteredFragments(pfa.data.Fragments):
    def __init__(self, fragments, fragments_oi):
        assert (torch.is_tensor(fragments_oi))
        self._coordinates = fragments.coordinates[fragments_oi]
        self._genemapping = fragments.genemapping[fragments_oi]

        # filtering has to be done on indices
        self._cellxgene_indptr = torch.ops.torch_sparse.ind2ptr(
            torch.ops.torch_sparse.ptr2ind(fragments.cellxgene_indptr, fragments.cellxgene_indptr[-1]
            )[fragments_oi],
            len(fragments.cellxgene_indptr)
        )
        self.fragments = fragments
        
        self.retained = fragments_oi.float().mean().cpu().numpy()
        self.retained_genes = pd.Series(
            torch_scatter.scatter_mean(fragments_oi.float().to("cpu"), fragments.genemapping, dim_size = fragments.n_genes).cpu().numpy(),
            index = fragments.var.index
        )
        
    def __getattribute__(self, k):
        try:
            return super().__getattribute__(k)
        except:
            return self.fragments.__getattribute__(k)


# %%
import torch_sparse

# %%
fragments_oi = FilteredFragments(fragments, fragments.coordinates[:, 0] > 0)

# %%
print(prediction_name)
prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)

# loaders
if "loaders" in globals():
    loaders.terminate()
    del loaders
    import gc
    gc.collect()

loaders = pfa.loaders.LoaderPool(
    design_row["loader_cls"],
    design_row["loader_parameters"],
    n_workers = 20,
    shuffle_on_iter = False
)

# load all models
models = [pickle.load(open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")) for fold_ix, fold in enumerate(folds[fold_slice])]

# %%
outcome = transcriptome.X.dense()
scorer = Scorer(models, folds, loaders, outcome, fragments.var.index, device = device)


# %% [markdown]
# ## Overall

# %%
# transcriptome_predicted_full, scores_overall, genescores_overall = scorer.score(return_prediction = True)

# %%
# scores_dir = (prediction.path / "scoring" / "overall")
# scores_dir.mkdir(parents = True, exist_ok = True)

# scores_overall.to_pickle(scores_dir / "scores.pkl")
# genescores_overall.to_pickle(scores_dir / "genescores.pkl")
# pickle.dump(transcriptome_predicted_full, (scores_dir / "transcriptome_predicted_full.pkl").open("wb"))

# %%
class PerturbationScorer():
    design:pd.DataFrame
    def __init__(self):
        self.design = pd.DataFrame()
        
    def get_fragments_oi(self, fragments, design_row):
        fragments_oi = FilteredFragments(fragments, torch.ones(fragments.coordinates.shape[0]))
        return fragments_oi
    
    def score(self, scorer, transcriptome_predicted_full):
        scores = {}
        genescores = {}
        for design_name, design_row in tqdm.tqdm(self.design.iterrows(), total = self.design.shape[0]):
            fragments_oi = self.get_fragments_oi(fragments, design_row)
            
            scores_, genescores_ = scorer.score(dict(fragments = fragments_oi), transcriptome_predicted_full = transcriptome_predicted_full)
            scores_["retained"] = fragments_oi.retained
            genescores_ = genescores_.join(fragments_oi.retained_genes.to_frame("retained"))
            scores[design_name] = scores_
            genescores[design_name] = genescores_
            
        design_names = self.design.index.names
        scores = pd.concat(scores, names = [*design_names, "phase"])
        genescores = pd.concat(genescores, names = [*design_names, "phase", "gene"])
        return scores, genescores


# %% [markdown]
# ## Fragment lengths

# %%
class FragmentLengthScorer(PerturbationScorer):
    design:pd.DataFrame
    def __init__(self):
        # cuts = list(np.arange(0, 1000, 50))
        cuts = list(np.arange(0, 1000, 25))
        # cuts = list(np.arange(0, 1000, 500))

        design = []
        for window_start, window_end in zip(cuts, cuts[1:] + [9999999]):  
            design.append({
                "window_start":window_start,
                "window_end":window_end
            })
        design = pd.DataFrame(design).set_index("window_start", drop = False)
        design.index.name = "window"
        self.design = design
        
    def get_fragments_oi(self, fragments, design_row):
        fragment_lengths = (fragments.coordinates[:,1] - fragments.coordinates[:,0])
        fragments_selected = ~((fragment_lengths >= design_row["window_start"]) & (fragment_lengths < design_row["window_end"]))
        
        fragments_oi = FilteredFragments(fragments, fragments_selected)
        return fragments_oi


# %%
scores_dir = (prediction.path / "scoring" / "overall")
transcriptome_predicted_full = pickle.load((scores_dir / "transcriptome_predicted_full.pkl").open("rb"))

# %%
import torch_sparse

# %%
pscorer = FragmentLengthScorer()
scores, genescores = pscorer.score(scorer, transcriptome_predicted_full = transcriptome_predicted_full)

# %%
scores = scores.reorder_levels(["phase", *pscorer.design.index.names])
genescores = genescores.reorder_levels(["phase", "gene", *pscorer.design.index.names])

scores_lengths = scores
genescores_lengths = genescores

# %%
scores_dir = (prediction.path / "scoring" / "lengths")
scores_dir.mkdir(parents = True, exist_ok = True)

pscorer.design.to_pickle(scores_dir / "design.pkl")
scores_lengths.to_pickle(scores_dir / "scores.pkl")
genescores_lengths.to_pickle(scores_dir / "genescores.pkl")


# %% [markdown]
# ## Window

# %% [markdown]
# Hypothesis: **are fragments from certain regions more predictive than others?**

# %%
def select_window(coordinates, window_start, window_end):
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()


# %%
class WindowScorer(PerturbationScorer):
    def __init__(self, window, window_size = 100):
        cuts = np.arange(*window, step = window_size).tolist() + [window[-1]]

        design = []
        for window_start, window_end in zip(cuts[:-1], cuts[1:]):  
            design.append({
                "window_start":window_start,
                "window_end":window_end,
                "window_mid":window_start + (window_end - window_start)/2
            })
        design = pd.DataFrame(design).set_index("window_mid", drop = False)
        design.index.name = "window"
        self.design = design
        
    def get_fragments_oi(self, fragments, design_row):
        fragments_selected = select_window(fragments.coordinates, design_row["window_start"], design_row["window_end"])
        fragments_oi = FilteredFragments(fragments, fragments_selected)
        return fragments_oi


# %%
scores_dir_overall = (prediction.path / "scoring" / "overall")
transcriptome_predicted_full = pickle.load((scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb"))

# %%
# window_size = 5000
# window_size = 500
window_size = 100

# %%
# pscorer = WindowScorer(window)
pscorer = WindowScorer(window, window_size = window_size)
scores, genescores = pscorer.score(scorer, transcriptome_predicted_full = transcriptome_predicted_full)

scores = scores.reorder_levels(["phase", *pscorer.design.index.names])
genescores = genescores.reorder_levels(["phase", "gene", *pscorer.design.index.names])

scores_windows = scores
genescores_windows = genescores

# %%
scores = scores.reorder_levels(["phase", *pscorer.design.index.names])
genescores = genescores.reorder_levels(["phase", "gene", *pscorer.design.index.names])

# %%
if window_size == 100:
    scores_dir = (prediction.path / "scoring" / "windows")
else:
    scores_dir = (prediction.path / "scoring" / ("windows_" + str(window_size)))
scores_dir.mkdir(parents = True, exist_ok = True)

pscorer.design.to_pickle(scores_dir / "design.pkl")
scores.to_pickle(scores_dir / "scores.pkl")
genescores.to_pickle(scores_dir / "genescores.pkl")


# %% [markdown]
# ## Window pairs

# %%
def select_window(coordinates, window_start, window_end):
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

# %%
import itertools


# %%
class WindowPairScorer(PerturbationScorer):
    def __init__(self, window, window_size = 100):
        cuts = np.arange(*window, step = window_size).tolist() + [window[-1]]
        bins = list(itertools.product(zip(cuts[:-1], cuts[1:]), zip(cuts[:-1], cuts[1:])))

        design = []
        for (window_start1, window_end1), (window_start2, window_end2) in bins:  
            design.append({
                "window_start1":window_start1,
                "window_end1":window_end1,
                "window_mid1":window_start1 + (window_end1 - window_start1)/2,
                "window_start2":window_start2,
                "window_end2":window_end2,
                "window_mid2":window_start2 + (window_end2 - window_start2)/2
            })
        design = pd.DataFrame(design).set_index(["window_mid1", "window_mid2"], drop = False)
        design.index.names = ["window1", "window2"]
        self.design = design
        
    def get_fragments_oi(self, fragments, design_row):
        fragments_selected = (
            select_window(fragments.coordinates, design_row["window_start1"], design_row["window_end1"]) |
            select_window(fragments.coordinates, design_row["window_start2"], design_row["window_end2"])
        )
        fragments_oi = FilteredFragments(fragments, fragments_selected)
        return fragments_oi


# %%
scores_dir_overall = (prediction.path / "scoring" / "overall")
transcriptome_predicted_full = pickle.load((scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb"))

# %%
# window_size = 5000
# window_size = 500
window_size = 500

# %%
pscorer = WindowPairScorer(window, window_size = window_size)
scores, genescores = pscorer.score(scorer, transcriptome_predicted_full = transcriptome_predicted_full)

scores = scores.reorder_levels(["phase", *pscorer.design.index.names])
genescores = genescores.reorder_levels(["phase", "gene", *pscorer.design.index.names])

scores_windows = scores
genescores_windows = genescores

# %%
scores = scores.reorder_levels(["phase", *pscorer.design.index.names])
genescores = genescores.reorder_levels(["phase", "gene", *pscorer.design.index.names])

# %%
if window_size == 100:
    scores_dir = (prediction.path / "scoring" / "windowpairs")
else:
    scores_dir = (prediction.path / "scoring" / ("windowpairs_" + str(window_size)))
scores_dir.mkdir(parents = True, exist_ok = True)

pscorer.design.to_pickle(scores_dir / "design.pkl")
scores.to_pickle(scores_dir / "scores.pkl")
genescores.to_pickle(scores_dir / "genescores.pkl")

# %%

# %%
