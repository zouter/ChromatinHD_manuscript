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
import chromatinhd as chd

# %%
device = "cuda:0"
# device = "cpu"

folder_root = chd.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(
    folder_data_preproc / "transcriptome"
)

splitter = "random_5fold"
splitter = "permutations_5fold5repeat"

# fragments
# promoter_name, window = "1k1k", np.array([-1000, 1000])
# promoter_name, window = "10k10k", np.array([-10000, 10000])
promoter_name, window = "100k100k", np.array([-100000, 100000])
# promoter_name, window = "20kpromoter", np.array([-10000, 0])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(
    folder_data_preproc / "fragments" / promoter_name
)

# create design to run
from design import get_design, get_folds_inference

class Prediction(chd.flow.Flow):
    pass

# folds & minibatching
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step = 2000)
folds = folds#[:1]

# design
from design import get_design, get_folds_training
design = get_design(transcriptome, fragments)

# %%
Scorer = chd.scoring.prediction.Scorer

# %%
prediction_name = "v20"
prediction_name = "v20_initdefault"
prediction_name = "counter"
design_row = design[prediction_name]

# %%
fragments.window = window

# %%
design_row["loader_parameters"]["cellxgene_batch_size"] = cellxgene_batch_size

# %%
print(prediction_name)
prediction = Prediction(chd.get_output() / "prediction_positional" / dataset_name / promoter_name / splitter / prediction_name)

# loaders
if "loaders" in globals():
    loaders.terminate()
    del loaders
    import gc
    gc.collect()

loaders = chd.loaders.LoaderPool(
    design_row["loader_cls"],
    design_row["loader_parameters"],
    n_workers = 20,
    shuffle_on_iter = False
)

# load all models
models = [pickle.load(open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")) for fold_ix, fold in enumerate(folds)]

# %%
# outcome = transcriptome.X.dense()
outcome = torch.from_numpy(transcriptome.adata.layers["magic"])
scorer = Scorer(models, folds[:len(models)], loaders, outcome, fragments.var.index, device = device)

# %% [markdown]
# ## Overall

# %%
import xarray as xr

# %%
transcriptome_predicted_full, scores_overall, genescores_overall = scorer.score(return_prediction = True)

# %%
scores_dir = (prediction.path / "scoring" / "overall")
scores_dir.mkdir(parents = True, exist_ok = True)

scores_overall.to_pickle(scores_dir / "scores.pkl")
genescores_overall.to_pickle(scores_dir / "genescores.pkl")

genescores_overall = xr.Dataset({"cor":xr.DataArray(genescores_overall["cor"].unstack())})
scores_overall = xr.Dataset({"cor":xr.DataArray(scores_overall["cor"])})

scores_overall.to_netcdf(scores_dir / "scores.cdf")
genescores_overall.to_netcdf(scores_dir / "genescores.cdf")
pickle.dump(transcriptome_predicted_full, (scores_dir / "transcriptome_predicted_full.pkl").open("wb"))

# %% [markdown]
# ## Filter Fragments def

# %%
import torch_scatter
import xarray as xr


# %%
class PerturbationScorer():
    design:pd.DataFrame
    def __init__(self):
        self.design = pd.DataFrame()
        
    def get_fragments_oi(self, fragments, design_row):
        fragments_oi = FilteredFragments(fragments, torch.ones(fragments.coordinates.shape[0]))
        return fragments_oi
    
    def score(self, scorer, transcriptome_predicted_full):
        scores = []
        retained = []
        genescores = []
        generetained = []
        effects = []
        geneeffects = []
        for design_name, design_row in tqdm.tqdm(self.design.iterrows(), total = self.design.shape[0]):
            fragments_oi = self.get_fragments_oi(fragments, design_row, folds)
            
            scores_, genescores_ = scorer.score(dict(fragments = fragments_oi), transcriptome_predicted_full = transcriptome_predicted_full, folds = folds)
            retained.append(fragments_oi.retained)
            generetained.append(fragments_oi.retained_genes)
            scores.append(scores_["cor"])
            effects.append(scores_["effect"])
            genescores.append(genescores_["cor"])
            geneeffects.append(genescores_["effect"])
            
        scores = xr.concat([xr.DataArray(score) for score in scores], dim = self.design.index)
        genescores = xr.concat([xr.DataArray(score.unstack()) for score in genescores], dim = self.design.index)
        retained = xr.concat([xr.DataArray(score) for score in retained], dim = self.design.index)
        generetained = xr.concat([xr.DataArray(score) for score in generetained], dim = self.design.index)
        effects = xr.concat([xr.DataArray(score) for score in effects], dim = self.design.index)
        geneeffects = xr.concat([xr.DataArray(score.unstack()) for score in geneeffects], dim = self.design.index)
        
        scores = xr.Dataset({"cor":scores, "retained":retained, "effect":effects})
        genescores = xr.Dataset({"cor":genescores, "retained":generetained, "effect":geneeffects})
            
        return scores, genescores


# %%
import torch_scatter
n_cells_step = 5000
n_genes_step = 100
class FilteredFragments(chd.data.Fragments):
    def __init__(self, fragments, fragments_oi, folds):
        assert (torch.is_tensor(fragments_oi))
        self._coordinates = fragments.coordinates[fragments_oi]
        self._genemapping = fragments.genemapping[fragments_oi]

        # filtering has to be done on indices
        cellxgene_indptr = fragments.cellxgene_indptr.clone()
        cellxgene_indptr[1:] = cellxgene_indptr[1:] - torch.cumsum(torch_scatter.segment_sum_csr(1-fragments_oi.to(torch.int), fragments.cellxgene_indptr), 0)
        self._cellxgene_indptr = cellxgene_indptr
        self.fragments = fragments
        
        self.retained = fragments_oi.float().mean().cpu().numpy()
        self.retained_genes = pd.Series(
            torch_scatter.scatter_mean(fragments_oi.float().to("cpu"), fragments.genemapping, dim_size = fragments.n_genes).cpu().numpy(),
            index = fragments.var.index
        )
        
        genes_all = np.arange(fragments.n_genes)
        genes_all = np.arange(fragments.n_genes)[self.retained_genes < 0.99]
        
        self.folds = []
        for fold in folds:
            cells = []
            fold["phases"] = {}
            
            if "cells_validation" in fold:
                cells_validation = list(fold["cells_validation"])
                fold["phases"]["validation"] = [cells_validation, genes_all]
                cells.extend(cells_validation)

            if "cells_test" in fold:
                cells_test = list(fold["cells_test"])
                fold["phases"]["test"] = [cells_test, genes_all]
                cells.extend(cells_test)

            rg = np.random.RandomState(0)

            minibatches = chd.loaders.minibatching.create_bins_ordered(
                cells,
                genes_all,
                n_cells_step=n_cells_step,
                n_genes_step=n_genes_step,
                n_genes_total=fragments.n_genes,
                use_all=True,
                rg=rg,
            )
            fold["minibatches"] = minibatches
            print(len(minibatches))
            self.folds.append(fold)
        
    def __getattribute__(self, k):
        try:
            return super().__getattribute__(k)
        except:
            return self.fragments.__getattribute__(k)
fragments_oi = FilteredFragments(fragments, fragments.coordinates[:, 0] > 0, folds)


# %% [markdown]
# ## Fragment lengths

# %%
class FragmentLengthScorer(PerturbationScorer):
    design:pd.DataFrame
    def __init__(self):
        # cuts = list(np.arange(0, 1000, 50))
        # cuts = list(np.arange(0, 1000, 25))
        # cuts = list(np.arange(0, 1000, 500))
        cuts = [0, 900, 1000]

        design = []
        for window_start, window_end in zip(cuts, cuts[1:] + [9999999]):  
            design.append({
                "window_start":window_start,
                "window_end":window_end
            })
        design = pd.DataFrame(design).set_index("window_start", drop = False)
        design.index.name = "window"
        self.design = design
        
    def get_fragments_oi(self, fragments, design_row, folds):
        fragment_lengths = (fragments.coordinates[:,1] - fragments.coordinates[:,0])
        fragments_selected = ~((fragment_lengths >= design_row["window_start"]) & (fragment_lengths < design_row["window_end"]))
        
        fragments_oi = FilteredFragments(fragments, fragments_selected, folds)
        return fragments_oi


# %%
scores_dir = (prediction.path / "scoring" / "overall")
transcriptome_predicted_full = pickle.load((scores_dir / "transcriptome_predicted_full.pkl").open("rb"))

# %%
pscorer = FragmentLengthScorer()
scores, genescores = pscorer.score(scorer, transcriptome_predicted_full = transcriptome_predicted_full)

# %%
scores.sel(phase = "validation")["cor"].to_pandas().plot()

# %%
scores.sel(phase = "test")["cor"].to_pandas().plot()

# %%
scores_dir = (prediction.path / "scoring" / "lengths")
scores_dir.mkdir(parents = True, exist_ok = True)

pscorer.design.to_pickle(scores_dir / "design.pkl")
scores.to_netcdf(scores_dir / "scores.cdf")
genescores.to_netcdf(scores_dir / "genescores.cdf")


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
        cuts = cuts#[:2]

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
        
    def get_fragments_oi(self, fragments, design_row, folds):
        fragments_selected = select_window(fragments.coordinates, design_row["window_start"], design_row["window_end"])
        fragments_oi = FilteredFragments(fragments, fragments_selected, folds)
        return fragments_oi


# %%
scores_dir_overall = (prediction.path / "scoring" / "overall")
transcriptome_predicted_full = pickle.load((scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb"))

# %%
window_size = 100
# window_size = 1000
# window_size = 100

# %%
pscorer = WindowScorer(window, window_size = window_size)
scores, genescores = pscorer.score(scorer, transcriptome_predicted_full = transcriptome_predicted_full)

# %%
import cProfile

stats = cProfile.run("pscorer.score(scorer, transcriptome_predicted_full = transcriptome_predicted_full)", "restats")
import pstats

p = pstats.Stats("restats")
p.sort_stats("cumulative").print_stats()

# %%
data.filter_fragments(

# %%
if window_size == 100:
    scores_dir = (prediction.path / "scoring" / "windows")
else:
    scores_dir = (prediction.path / "scoring" / ("windows_" + str(window_size)))
scores_dir.mkdir(parents = True, exist_ok = True)

pscorer.design.to_pickle(scores_dir / "design.pkl")
scores.to_netcdf(scores_dir / "scores.cdf")
genescores.to_netcdf(scores_dir / "genescores.cdf")


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
                "window_start1":window1.start,
                "window_end1":window1.end,
                "window_mid1":window1.start + (window1.end - window1.start)/2,
                "window_start2":window2.start,
                "window_end2":window2.end,
                "window_mid2":window2.start + (window2.end - window2.start)/2
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

# %% [markdown]
# ## Real

# %%
scores_dir_overall = (prediction.path / "scoring" / "overall")
transcriptome_predicted_full = pickle.load((scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb"))

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step = 2000)
folds = folds#[:1]

# %% [markdown]
# ## Nothing

# %%
Scorer2 = chd.scoring.prediction.Scorer2

# %%
outcome = torch.from_numpy(transcriptome.adata.layers["magic"])


# %%
class NothingFilterer():
    def __init__(self):
        design = []
        for i in range(1):  
            design.append({"i":0})
        design = pd.DataFrame(design).set_index("i", drop = False)
        design.index.name = "i"
        design["ix"] = np.arange(len(design))
        self.design = design
        
    def setup_next_chunk(self):
        return len(self.design)
        
    def filter(self, data):
        yield data.coordinates[:, 0] > -999*10**10
            
nothing_filterer = NothingFilterer()

# %%
# outcome = transcriptome.X.dense()
outcome = torch.from_numpy(transcriptome.adata.layers["magic"])
nothing_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index, device = device)

# %%
models = [model.to("cpu") for model in models]

# %%
# del nothing_scorer

import gc
gc.collect()
torch.cuda.empty_cache()

# %%
nothing_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = nothing_filterer, extract_total = True)

# %% [markdown]
# ## Subset

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
genes_oi = transcriptome.var["symbol"] == "CTLA4"
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step = 2000, genes_oi = genes_oi)
folds = folds#[:1]

# %% [markdown]
# ## Window

# %%
# outcome = transcriptome.X.dense()
window_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], fragments.obs.index, device = device)


# %%
def select_window(coordinates, window_start, window_end):
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

class WindowFilterer():
    def __init__(self, window, window_size = 100):
        cuts = np.arange(*window, step = window_size).tolist() + [window[-1]]
        cuts = cuts

        design = []
        for window_start, window_end in zip(cuts[:-1], cuts[1:]):  
            design.append({
                "window_start":window_start,
                "window_end":window_end,
                "window_mid":window_start + (window_end - window_start)/2
            })
        design = pd.DataFrame(design).set_index("window_mid", drop = False)
        design.index.name = "window"
        design["ix"] = np.arange(len(design))
        self.design = design
        
    def setup_next_chunk(self):
        return len(self.design)
        
    def filter(self, data):
        for design_ix, window_start, window_end in zip(self.design["ix"], self.design["window_start"], self.design["window_end"]):
            fragments_oi = select_window(data.coordinates, window_start, window_end)
            yield fragments_oi
            
window_filterer = WindowFilterer(window, window_size = 500)

# %%
window_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = window_filterer)

# %%
window_scorer.genescores["retained"] = 1-window_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0)

# %%
gene_ix = transcriptome.gene_ix("CTLA4")
gene_id = transcriptome.var.iloc[gene_ix].name

# %%
window_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "train").plot()
window_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "validation").plot()
window_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "test").plot()

# %%
# genescores["cor"].mean("model").sel(phase = "train").sel(gene = transcriptome.gene_id("IL1B")).plot()
# genescores["cor"].mean("model").sel(phase = "validation").sel(gene = transcriptome.gene_id("CTLA4")).plot()
fig, ax = plt.subplots()
window_scorer.genescores["cor"].sel(phase = "validation").sel(gene = gene_id).plot(ax = ax)
ax2 = ax.twinx()
window_scorer.genescores["effect"].sel(phase = "test").mean("gene").plot(ax = ax2, color = "red")

# %% [markdown]
# ## Pairwindow

# %%
import itertools
class WindowPairFilterer():
    def __init__(self, windows_oi):
        design = []
        for (window1_id, window1), (window2_id, window2) in itertools.combinations_with_replacement(windows_oi.iterrows(), 2):
            design.append({
                "window_start1":window1.window_start,
                "window_end1":window1.window_end,
                "window_mid1":int(window1.window_start + (window1.window_end - window1.window_start)//2),
                "window_start2":window2.window_start,
                "window_end2":window2.window_end,
                "window_mid2":int(window2.window_start + (window2.window_end - window2.window_start)//2),
                "window1":window1_id,
                "window2":window2_id
            })
        design = pd.DataFrame(design)
        design.index = design["window_mid1"].astype(str) + "-" + design["window_mid2"].astype(str)
        design.index.name = "windowpair"
        design["ix"] = np.arange(len(design))
        self.design = design
        
    def setup_next_chunk(self):
        return len(self.design)
        
    def filter(self, data):
        for design_ix, window_start1, window_end1, window_start2, window_end2 in zip(self.design["ix"], self.design["window_start1"], self.design["window_end1"], self.design["window_start2"], self.design["window_end2"]):
            fragments_oi = (
                select_window(data.coordinates, window_start1, window_end1) &
                select_window(data.coordinates, window_start2, window_end2)
            )
            yield fragments_oi
            

class WindowPairBaselineFilterer(WindowPairFilterer):
    def __init__(self, windowpair_filterer):
        self.design = windowpair_filterer.design.copy()
        
    def setup_next_chunk(self):
        return len(self.design)
        
    def filter(self, data):
        for design_ix, window_start1, window_end1, window_start2, window_end2 in zip(self.design["ix"], self.design["window_start1"], self.design["window_end1"], self.design["window_start2"], self.design["window_end2"]):
            fragments_oi = ~(
                ~select_window(data.coordinates, window_start1, window_end1) &
                ~select_window(data.coordinates, window_start2, window_end2)
            )
            yield fragments_oi


# %%
window_scorer.genescores["deltacor"] = window_scorer.genescores["cor"] - nothing_scorer.genescores.sel(i = 0)["cor"]

# %%
windows_oi = window_filterer.design.loc[(window_scorer.genescores["deltacor"].sel(gene = gene_id, phase = "test") < -0.0005).values]
# windows_oi = window_filterer.design
windows_oi.shape

# %%
windowpair_filterer = WindowPairFilterer(windows_oi)
windowpair_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], device = device)
windowpair_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = windowpair_filterer)
windowpair_scorer.genescores["retained"] = 1-windowpair_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0, gene = gene_id)

# %%
windowpair_baseline_filterer = WindowPairBaselineFilterer(windowpair_filterer)
windowpair_baseline_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], device = device)
windowpair_baseline_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = windowpair_baseline_filterer)
windowpair_baseline_scorer.genescores["retained"] = 1-windowpair_baseline_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0, gene = gene_id)

# %%
retained_additive = pd.Series(
    1- ((1-(window_scorer.genescores["retained"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window2"].values, phase = "validation")).values) + (1-(window_scorer.genescores["retained"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window1"].values, phase = "validation")).values)),
    windowpair_filterer.design.index
)

# %%
additive_baseline = windowpair_baseline_scorer.genescores["cor"].sel(gene = gene_id)
additive_base1 = (window_scorer.genescores["cor"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window1"].values)).reindex_like(additive_baseline)
additive_base2 = (window_scorer.genescores["cor"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window2"].values)).reindex_like(additive_baseline)

deltacor1 = additive_base1.values - additive_baseline
deltacor2 = additive_base2.values - additive_baseline
deltacor_additive = (additive_base1.values + additive_base2.values) - 2 * additive_baseline
deltacor_interacting = windowpair_scorer.genescores["cor"].sel(gene = gene_id) - additive_baseline

# %%
phase = "test"
# phase = "validation"

interaction = windowpair_filterer.design.copy()
interaction["deltacor"] = deltacor_interacting.sel(phase = phase).to_pandas()
interaction["deltacor1"] = deltacor1.sel(phase = phase).to_pandas()
interaction["deltacor2"] = deltacor2.sel(phase = phase).to_pandas()

additive = windowpair_filterer.design.copy()
additive["deltacor"] = deltacor_additive.sel(phase = phase).to_pandas()

# %%
sns.heatmap(interaction.set_index(["window_mid1", "window_mid2"])["deltacor"].unstack())

# %%
sns.heatmap(additive.set_index(["window_mid1", "window_mid2"])["deltacor"].unstack())

# %%
interaction["deltacor_interaction"] = (interaction["deltacor"] - additive["deltacor"])

# %%
(window_mid1-window_mid2, window_mid2 - window_mid1)

# %%
norm(deltacor)

# %%
# %config InlineBackend.figure_format='retina'

# %%
radius = 500

# %%
203874196 - promoter["tss"]

# %%
fig, ax = plt.subplots(figsize = (20, 10))

norm = mpl.colors.Normalize(-0.001, 0.001)
norm = mpl.colors.Normalize(-0.0001, 0.0001)

cmap = mpl.cm.RdBu

for (window_mid1, window_mid2), deltacor in interaction.set_index(["window_mid1", "window_mid2"])["deltacor_interaction"].iloc[:10000].items():
    patch = mpl.patches.RegularPolygon((window_mid1+(window_mid2 - window_mid1)/2, (window_mid2 - window_mid1)/2), 4, radius = radius, orientation = np.pi/2, ec = None, lw = 0, fc = cmap(norm(deltacor)))
    ax.add_patch(patch)
ax.set_ylim((window[1] - window[0])/2)
ax.set_xlim(*window)

for x in [0, 6425]:
    ax.plot([window[0]+x, x], [(window[1] - window[0])/2, 0], zorder = -1, color = "#333", lw = 1)
    ax.plot([window[1]+x, x], [(window[1] - window[0])/2, 0], zorder = -1, color = "#333", lw = 1)
    
    # ax.matshow(
#    interaction.set_index(["window_mid1", "window_mid2"])["deltacor_interaction"].unstack(),
#     norm = mpl.colors.Normalize(-0.001, 0.001),
#     cmap = mpl.cm.RdBu,
# )

# %%
interaction["distance"] = np.abs(interaction["window_mid1"] - interaction["window_mid2"])

# %%
interaction["deltacor_min"] = interaction[["deltacor1", "deltacor2"]].min(1)
interaction["deltacor_max"] = interaction[["deltacor1", "deltacor2"]].max(1)

# %%
interaction["deltacor_interaction_ratio"] = interaction["deltacor_interaction"]/interaction["deltacor_max"]

# %%
plt.scatter(
    interaction["deltacor_max"],
    interaction["deltacor_interaction"],
    c = np.log1p(interaction["distance"])
)

# %%
plt.scatter(
    interaction["deltacor_min"],
    interaction["deltacor_interaction"],
    c = interaction["distance"]
)

# %% [markdown]
# ## Multiwindow

# %%
# outcome = transcriptome.X.dense()
multiwindow_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], device = device)

# %%
window_sizes = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)


# %%
def select_window(coordinates, window_start, window_end):
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

class MultiWindowFilterer():
    def __init__(self, window, window_sizes = (100, 200), relative_stride = 0.5):
        design = []
        for window_size in window_sizes:
            cuts = np.arange(*window, step = int(window_size*relative_stride))

            for window_start, window_end in zip(cuts, cuts + window_size):  
                design.append({
                    "window_start":window_start,
                    "window_end":window_end,
                    "window_mid":window_start + (window_end - window_start)/2,
                    "window_size":window_size
                })
        design = pd.DataFrame(design)
        design.index = design["window_start"].astype(str) + "-" + design["window_end"].astype(str)
        design.index.name = "window"
        design["ix"] = np.arange(len(design))
        self.design = design
        
    def setup_next_chunk(self):
        return len(self.design)
        
    def filter(self, data):
        for design_ix, window_start, window_end in zip(self.design["ix"], self.design["window_start"], self.design["window_end"]):
            fragments_oi = select_window(data.coordinates, window_start, window_end)
            yield fragments_oi
            
multiwindow_filterer = MultiWindowFilterer(window, window_sizes = window_sizes)

# %%
multiwindow_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = multiwindow_filterer)

# %%
multiwindow_scorer.genescores["retained"] = 1-multiwindow_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0)

# %%
multiwindow_scorer.genescores["deltacor"] = multiwindow_scorer.genescores["cor"] - nothing_scorer.genescores.sel(i = 0)["cor"]

# %%
gene_ix = transcriptome.gene_ix("CTLA4")
gene_id = transcriptome.var.iloc[gene_ix].name

# %%
# plotdata = multiwindow_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "validation").to_pandas().to_frame("retained")
plotdata = multiwindow_scorer.genescores.sel(gene = gene_id).stack().to_dataframe()
plotdata = multiwindow_filterer.design.join(plotdata)

# %%
fig, ax = plt.subplots()
plt.scatter(plotdata.loc["validation"]["deltacor"],plotdata.loc["test"]["deltacor"])
ax.set_xlim(-0.01)
ax.set_ylim(-0.01)

# %%
window_sizes_info = pd.DataFrame({"window_size":window_sizes}).set_index("window_size")
window_sizes_info["ix"] = np.arange(len(window_sizes_info))

# %%
fig, ax = plt.subplots(figsize = (20, 3))

deltacor_norm = mpl.colors.Normalize(0, 0.001)
deltacor_cmap = mpl.cm.Reds

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size").query("phase == 'validation'").iloc[::2]
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        rect = mpl.patches.Rectangle((plotdata_row["window_start"], y), plotdata_row["window_end"] - plotdata_row["window_start"], 1, lw = 0, fc = deltacor_cmap(deltacor_norm(-plotdata_row["deltacor"])))
        ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)
ax.axvline(6000)

# %%
fig, ax = plt.subplots(figsize = (20, 3))

effect_norm = mpl.colors.CenteredNorm()
effect_cmap = mpl.cm.RdBu_r

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size")
    print(plotdata_oi.shape)
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        if plotdata_row["deltacor"] < -0.0001:
            rect = mpl.patches.Rectangle((plotdata_row["window_start"], y), plotdata_row["window_end"] - plotdata_row["window_start"], 1, lw = 0, fc = effect_cmap(effect_norm(-plotdata_row["effect"])))
            ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)
ax.axvline(6000)

# %% [markdown]
# ### Interpolate per position

# %%
import scipy.interpolate

# %%
positions_oi = np.arange(*window)

# %%
interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size")
    # interpolation = np.clip(scipy.interpolate.interp1d(plotdata_oi["window_mid"], plotdata_oi["deltacor"], fill_value = "extrapolate")(positions_oi) / window_size * 1000, -np.inf, 0)
    interpolation = np.clip(np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["deltacor"]) / window_size * 1000, -np.inf, 0)
    # interpolation = np.clip(np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["deltacor"]), -np.inf, 0)
    interpolated[window_size_info["ix"], :] = interpolation

# %%
fig, ax = plt.subplots(figsize = (20, 3))
plt.plot(positions_oi, interpolated.mean(0))
ax.invert_yaxis()

# %% [markdown]
# ### Motifscan

# %%
motifscan_name = "gwas_immune"

# %%
motifscan_folder = chd.get_output() / "motifscans" / dataset_name / promoter_name / motifscan_name
motifscan = chd.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))
motifscan.n_motifs = len(motifs)
motifs["ix"] = np.arange(motifs.shape[0])

# %%
indptr_start = gene_ix * (window[1] - window[0])
indptr_end = (gene_ix + 1) * (window[1] - window[0])

# %%
motifscan.motifs["ix"] = np.arange(len(motifscan.motifs))
motifs_oi = motifscan.motifs

# %%
position_indices = chd.utils.numpy.indptr_to_indices(motifscan.indptr[indptr_start:indptr_end])
indices = motifscan.indices[motifscan.indptr[indptr_start]:motifscan.indptr[indptr_end]]

# %%
motifdata = []
for pos, motif_ix in zip(position_indices, indices):
    if motif_ix in motifs_oi["ix"].values:
        motifdata.append({"position":pos + window[0], "motif":motifs_oi.query("ix == @motif_ix").index[0]})

motifdata = pd.DataFrame(motifdata, columns = ["position", "motif"])
print(len(motifdata))

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
promoter = promoters.loc[gene_id]

# %%
fig, ax = plt.subplots(figsize = (20, 3))
ax.plot(positions_oi, interpolated.mean(0))

for _, motifdatum in motifdata.iterrows():
    ax.scatter(motifdatum["position"], 0, color = "red", s = 5, marker = "|")

ax.invert_yaxis()

# %%
import gseapy

# %%
rnk = pd.Series(0, index = pd.Series(list("abcdefghijklmnop")))
rnk.values[:] = -np.arange(len(rnk))
genesets = {
    "hi":["a", "b", "c"]
}

# %%
rnk = -pd.Series(interpolated.mean(0), index = positions_oi.astype(str))
genesets = {
    "hi":np.unique(motifdata["position"].astype(str).values)
}

# %%
# ranked = gseapy.prerank(rnk, genesets, min_size = 0)

# %%
rnk_sorted = pd.Series(np.sort(np.log(rnk)), index = rnk.index)
# rnk_sorted = pd.Series(np.sort(rnk), index = rnk.index)
fig, ax = plt.subplots()
sns.ecdfplot(rnk_sorted, ax = ax)
sns.ecdfplot(rnk_sorted[motifdata["position"].astype(str)], ax = ax, color = "orange")
for _, motifdatum in motifdata.iterrows():
    rnk_motif = rnk_sorted[str(motifdatum["position"])]
    q = np.searchsorted(rnk_sorted, rnk_motif) / len(rnk_sorted)
    ax.scatter([rnk_motif], [q], color = "red")
    # ax.scatter(motifdatum["position"], 0, color = "red", s = 5, marker = "|")

# %%
scipy.stats.ks_2samp(rnk_sorted[~rnk_sorted.index.isin(motifdata["position"].astype(str))], rnk_sorted[motifdata["position"].astype(str)])

# %%

# %%
