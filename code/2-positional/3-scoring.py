# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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

import torch

import tqdm.auto as tqdm
import xarray as xr
# -

import chromatinhd as chd

# +
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
fragments.obs.index.name = "cell"

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
# -

Scorer = chd.scoring.prediction.Scorer

prediction_name = "v20"
prediction_name = "v20_initdefault"
# prediction_name = "counter"
design_row = design[prediction_name]

fragments.window = window

design_row["loader_parameters"]["cellxgene_batch_size"] = cellxgene_batch_size

# +
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
# -

# outcome = transcriptome.X.dense()
outcome = torch.from_numpy(transcriptome.adata.layers["magic"])
scorer = Scorer(models, folds[:len(models)], loaders, outcome, fragments.var.index, device = device)

# ## Load folds

scores_dir_overall = (prediction.path / "scoring" / "overall")
transcriptome_predicted_full = pickle.load((scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb"))

folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step = 2000)
folds = folds#[:1]

# ## Nothing

scorer_folder = prediction.path / "scoring" / "nothing"
scorer_folder.mkdir(exist_ok=True, parents=True)

Scorer2 = chd.scoring.prediction.Scorer2

outcome = torch.from_numpy(transcriptome.adata.layers["magic"])


# +
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
# -

# outcome = transcriptome.X.dense()
outcome = torch.from_numpy(transcriptome.adata.layers["magic"])
nothing_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index, fragments.obs.index, device = device)

models = [model.to("cpu") for model in models]

nothing_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = nothing_filterer, extract_total = True)

scorer_folder = prediction.path / "scoring" / "nothing"
scorer_folder.mkdir(exist_ok=True, parents=True)
nothing_scorer.scores.to_netcdf(scorer_folder / "scores.nc")
nothing_scorer.genescores.to_netcdf(scorer_folder / "genescores.nc")
nothing_filterer.design.to_pickle(scorer_folder / "design.pkl")

# ## Subset

# +
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

symbol = "SELL"
symbol = "BACH2"
symbol = "CTLA4"
genes_oi = transcriptome.var["symbol"] == symbol
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step = 2000, genes_oi = genes_oi)
folds = folds#[:1]

gene_ix = transcriptome.gene_ix(symbol)
gene_id = transcriptome.var.iloc[gene_ix].name
# -

# ## Window

# outcome = transcriptome.X.dense()
window_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], fragments.obs.index, device = device)


# +
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
            
window_filterer = WindowFilterer(window, window_size = 1000)
# -

window_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = window_filterer)

window_scorer.genescores["retained"] = 1-window_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0)

gene_ix = transcriptome.gene_ix(symbol)
gene_id = transcriptome.var.iloc[gene_ix].name

window_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "train").plot()
window_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "validation").plot()
window_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "test").plot()

# genescores["cor"].mean("model").sel(phase = "train").sel(gene = transcriptome.gene_id("IL1B")).plot()
# genescores["cor"].mean("model").sel(phase = "validation").sel(gene = transcriptome.gene_id("CTLA4")).plot()
fig, ax = plt.subplots()
window_scorer.genescores["cor"].sel(phase = "validation").sel(gene = gene_id).plot(ax = ax)
ax2 = ax.twinx()
window_scorer.genescores["effect"].sel(phase = "test").mean("gene").plot(ax = ax2, color = "red")

effects = xr.concat(window_scorer.cellgeneeffects, dim = pd.Index(np.arange(len(models)), name = "model"))
effects = effects.mean("model", skipna = True)

#
window_oi = window_scorer.genescores.sel(phase = "test").sel(gene = gene_id).to_pandas().sort_values("effect").index[0]

# +
# delta_zsme = xr.concat(
#     [window_scorer.zsme_diff_cellgene[(model_ix, "validation", window_oi)] for model_ix in range(len(models))]
#     ,dim = pd.Index(np.arange(len(models)), name = "model")
# )
# delta_zsme = delta_zsme.mean("model", skipna = True)
# -

transcriptome.adata.obs["effect"] = effects.sel(gene = gene_id).sel(window = window_oi).to_pandas()
# transcriptome.adata.obs["effect"] = delta_zsme.sel(gene = gene_id).to_pandas()

sc.pl.umap(transcriptome.adata, color = [gene_id, transcriptome.gene_id("CD14")], layer="magic")

# +
# sc.pl.umap(transcriptome.adata, color = ["effect"], vmin = -10, vmax = 0)

max_effect = 0.1
sc.pl.umap(
    transcriptome.adata,
    color=["effect"],
    vmin=-max_effect,
    vmax=max_effect,
    cmap=mpl.cm.RdBu,
)

# +
import faiss

# X = np.array(transcriptome.adata.X.todense())
X = transcriptome.adata.obsm["X_pca"]

index = faiss.index_factory(X.shape[1], "Flat")
index.train(X)
index.add(X)
distances, neighbors = index.search(X, 10)
neighbors = neighbors[:, 1:]
# -

effects_raw = transcriptome.adata.obs["effect"].values
effects_smooth = effects_raw[neighbors].mean(1)
transcriptome.adata.obs["effect_smooth"] = effects_smooth

# sc.pl.umap(transcriptome.adata, color = ["effect"], vmin = -10, vmax = 0)
max_effect = 0.01
sc.pl.umap(
    transcriptome.adata,
    color=["effect", "effect_smooth"],
    vmin=-max_effect,
    vmax=max_effect,
    cmap=mpl.cm.RdBu,
)

scorer_folder = prediction.path / "scoring" / "window_gene" / symbol
scorer_folder.mkdir(exist_ok=True, parents=True)
window_scorer.scores.to_netcdf(scorer_folder / "scores.nc")
window_scorer.genescores.to_netcdf(scorer_folder / "genescores.nc")
window_filterer.design.to_pickle(scorer_folder / "design.pkl")

# ## Pairwindow

# +
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


# -

window_scorer.genescores["deltacor"] = window_scorer.genescores["cor"] - nothing_scorer.genescores.sel(i = 0)["cor"]

windows_oi = window_filterer.design.loc[(window_scorer.genescores["deltacor"].sel(gene = gene_id, phase = "test") < -0.0005).values]
# windows_oi = window_filterer.design
windows_oi.shape

windowpair_filterer = WindowPairFilterer(windows_oi)
windowpair_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], fragments.obs.index, device = device)
windowpair_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = windowpair_filterer)
windowpair_scorer.genescores["retained"] = 1-windowpair_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0, gene = gene_id)

windowpair_baseline_filterer = WindowPairBaselineFilterer(windowpair_filterer)
windowpair_baseline_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], fragments.obs.index, device = device)
windowpair_baseline_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = windowpair_baseline_filterer)
windowpair_baseline_scorer.genescores["retained"] = 1-windowpair_baseline_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0, gene = gene_id)

retained_additive = pd.Series(
    1- ((1-(window_scorer.genescores["retained"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window2"].values, phase = "validation")).values) + (1-(window_scorer.genescores["retained"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window1"].values, phase = "validation")).values)),
    windowpair_filterer.design.index
)

# ### Interpret

# +
additive_baseline = windowpair_baseline_scorer.genescores["cor"].sel(gene = gene_id)
additive_base1 = (window_scorer.genescores["cor"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window1"].values)).reindex_like(additive_baseline)
additive_base2 = (window_scorer.genescores["cor"].sel(gene = gene_id).sel(window = windowpair_filterer.design["window2"].values)).reindex_like(additive_baseline)

deltacor1 = additive_base1.values - additive_baseline
deltacor2 = additive_base2.values - additive_baseline
deltacor_additive = (additive_base1.values + additive_base2.values) - 2 * additive_baseline
deltacor_interacting = windowpair_scorer.genescores["cor"].sel(gene = gene_id) - additive_baseline

# +
phase = "test"
phase = "validation"

interaction = windowpair_filterer.design.copy()
interaction["deltacor"] = deltacor_interacting.sel(phase = phase).to_pandas()
interaction["deltacor1"] = deltacor1.sel(phase = phase).to_pandas()
interaction["deltacor2"] = deltacor2.sel(phase = phase).to_pandas()

additive = windowpair_filterer.design.copy()
additive["deltacor"] = deltacor_additive.sel(phase = phase).to_pandas()
# -

sns.heatmap(interaction.set_index(["window_mid1", "window_mid2"])["deltacor"].unstack())

sns.heatmap(additive.set_index(["window_mid1", "window_mid2"])["deltacor"].unstack())

interaction["deltacor_interaction"] = (interaction["deltacor"] - additive["deltacor"])

# %config InlineBackend.figure_format='retina'

radius = 500

# +
fig, ax = plt.subplots(figsize = (20, 10))

norm = mpl.colors.Normalize(-0.001, 0.001)
norm = mpl.colors.Normalize(-0.0001, 0.0001)

cmap = mpl.cm.RdBu

for (window_mid1, window_mid2), deltacor in interaction.set_index(["window_mid1", "window_mid2"])["deltacor_interaction"].items():
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
# -

interaction["distance"] = np.abs(interaction["window_mid1"] - interaction["window_mid2"])

interaction["deltacor_min"] = interaction[["deltacor1", "deltacor2"]].min(1)
interaction["deltacor_max"] = interaction[["deltacor1", "deltacor2"]].max(1)

interaction["deltacor_interaction_ratio"] = interaction["deltacor_interaction"]/interaction["deltacor_max"]

plt.scatter(
    interaction["deltacor_max"],
    interaction["deltacor_interaction"],
    c = np.log1p(interaction["distance"])
)

plt.scatter(
    interaction["deltacor_min"],
    interaction["deltacor_interaction"],
    c = interaction["distance"]
)

# ## Multiwindow

# outcome = transcriptome.X.dense()
multiwindow_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], fragments.obs.index, device = device)

window_sizes = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
window_sizes = (500, )


# +
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
# -

multiwindow_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = multiwindow_filterer)

multiwindow_scorer.genescores["retained"] = 1-multiwindow_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0)
multiwindow_scorer.genescores["deltacor"] = multiwindow_scorer.genescores["cor"] - nothing_scorer.genescores.sel(i = 0)["cor"]

# plotdata = multiwindow_scorer.genescores["retained"].sel(gene = gene_id).sel(phase = "validation").to_pandas().to_frame("retained")
plotdata = multiwindow_scorer.genescores.sel(gene = gene_id).stack().to_dataframe()
plotdata = multiwindow_filterer.design.join(plotdata)

fig, ax = plt.subplots()
plt.scatter(plotdata.loc["validation"]["deltacor"],plotdata.loc["test"]["deltacor"])
ax.set_xlim(-0.01)
ax.set_ylim(-0.01)

window_sizes_info = pd.DataFrame({"window_size":window_sizes}).set_index("window_size")
window_sizes_info["ix"] = np.arange(len(window_sizes_info))

# +
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

# +
fig, ax = plt.subplots(figsize = (20, 3))

effect_norm = mpl.colors.CenteredNorm()
effect_cmap = mpl.cm.RdBu_r

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size")
    print(plotdata_oi.shape)
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        if plotdata_row["deltacor"] < -0.001:
            rect = mpl.patches.Rectangle((plotdata_row["window_start"], y), plotdata_row["window_end"] - plotdata_row["window_start"], 1, lw = 0, fc = effect_cmap(effect_norm(-plotdata_row["effect"])))
            ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)
ax.axvline(6000)
# -

# ### Interpolate per position

import scipy.interpolate

positions_oi = np.arange(*window)

deltacor_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
retained_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size")
    deltacor_interpolated_ = np.clip(np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["deltacor"]) / window_size * 1000, -np.inf, 0)
    deltacor_interpolated[window_size_info["ix"], :] = deltacor_interpolated_
    retained_interpolated_ = np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["retained"]) / window_size * 1000
    retained_interpolated[window_size_info["ix"], :] = retained_interpolated_

fig, ax = plt.subplots(figsize = (20, 3))
plt.plot(positions_oi, deltacor_interpolated.mean(0))
ax2 = ax.twinx()
ax2.plot(positions_oi, retained_interpolated.mean(0), color = "red", alpha = 0.6)
ax2.set_ylabel("retained")

# ## Variants/haplotypes

promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
promoter = promoters.loc[gene_id]

motifscan_name = "gwas_immune"

folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on = "snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

association_oi = association.loc[
    (association["chr"] == promoter["chr"]) & (association["pos"] >= promoter["start"]) & (association["pos"] <= promoter["end"])
].copy()

association_oi["position"] = (association_oi["pos"] - promoter["tss"]) * promoter["strand"]

# +
variants = pd.DataFrame({
    "disease/trait":association_oi.groupby("snp")["disease/trait"].apply(list),
    "snp_main_first":association_oi.groupby("snp")["snp_main"].first(),
})
variants = variants.join(snp_info)
variants["position"] = (variants["start"] - promoter["tss"]) * promoter["strand"]

haplotypes = association_oi.groupby("snp_main")["snp"].apply(lambda x:sorted(set(x))).to_frame("snps")
haplotypes["color"] = sns.color_palette("hls", n_colors = len(haplotypes))

# +
fig, ax = plt.subplots(figsize = (20, 3))
ax.plot(positions_oi  * promoter["strand"] + promoter["tss"], deltacor_interpolated.mean(0))
ax2 = ax.twinx()
ax2.plot(positions_oi  * promoter["strand"] + promoter["tss"], retained_interpolated.mean(0), color = "red", alpha = 0.6)
ax2.set_ylabel("retained")

for _, variant in variants.iterrows():
    ax.scatter(
        variant["position"] * promoter["strand"] + promoter["tss"],
        0.9,
        color = haplotypes.loc[variant["snp_main_first"], "color"],
        s = 20,
        marker = "|",
        transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
    )

ax.invert_yaxis()
ax2.invert_yaxis()
# -

import gseapy

rnk = pd.Series(0, index = pd.Series(list("abcdefghijklmnop")))
rnk.values[:] = -np.arange(len(rnk))
genesets = {
    "hi":["a", "b", "c"]
}

rnk = -pd.Series(deltacor_interpolated.mean(0), index = positions_oi.astype(str))
genesets = {
    "hi":np.unique(variants["position"].astype(str).values)
}

# +
# ranked = gseapy.prerank(rnk, genesets, min_size = 0)
# -

rnk_sorted = pd.Series(np.sort(np.log(rnk)), index = rnk.index)
# rnk_sorted = pd.Series(np.sort(rnk), index = rnk.index)
fig, ax = plt.subplots()
sns.ecdfplot(rnk_sorted, ax = ax)
sns.ecdfplot(rnk_sorted[variants["position"].astype(int).astype(str)], ax = ax, color = "orange")
for _, motifdatum in variants.iterrows():
    rnk_motif = rnk_sorted[str(int(motifdatum["position"]))]
    q = np.searchsorted(rnk_sorted, rnk_motif) / len(rnk_sorted)
    ax.scatter([rnk_motif], [q], color = "red")
    # ax.scatter(motifdatum["position"], 0, color = "red", s = 5, marker = "|")

scipy.stats.ks_2samp(rnk_sorted[~rnk_sorted.index.isin(motifdata["position"].astype(str))], rnk_sorted[motifdata["position"].astype(str)])


# ## Around variants

# +
def select_window(coordinates, window_start, window_end):
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

class VariantFilterer(WindowFilterer):
    def __init__(self, positions, window_size = 500):
        design = []
        design = pd.DataFrame({"window_start":positions - window_size//2, "window_end":positions + window_size//2, "window_mid":positions})
        design["ix"] = np.arange(len(design))
        self.design = design



# -

variant_filterer = VariantFilterer(variants["position"], window_size = 1000)
variant_scorer = Scorer2(models, folds[:len(models)], loaders, outcome, fragments.var.index[genes_oi], fragments.obs.index, device = device)
variant_scorer.score(transcriptome_predicted_full=transcriptome_predicted_full, filterer = variant_filterer)

variant_scorer.genescores["retained"] = 1-variant_scorer.genescores["lost"]/nothing_scorer.genescores["total"].sel(i = 0)
variant_scorer.genescores["deltacor"] = variant_scorer.genescores["cor"] - nothing_scorer.genescores.sel(i = 0)["cor"]

variant_scores = variant_scorer.genescores.sel(gene = gene_id).sel(phase = "test").to_pandas()

haplotypes["n"] = haplotypes["snps"].apply(len)

for haplotype, snps in haplotypes["snps"].items():
    variant_scores_oi = variant_scores.loc[snps]
    if len(snps) > 5:
        extremity = variant_scores_oi["deltacor"].min() - variant_scores_oi["deltacor"].median()
        extremity_random = []
        for i in range(500):
            variant_scores_oi = variant_scores.sample(len(snps))
            extremity_random.append(variant_scores_oi["deltacor"].min() - variant_scores_oi["deltacor"].mean())
        print(extremity / np.mean(extremity_random), len(snps))

fig, ax = plt.subplots()
sns.histplot(extremity_random, ax = ax)
ax.axvline(extremity)

# +
fig, ax = plt.subplots(figsize = (20, 3))

ax.plot(positions_oi  * promoter["strand"] + promoter["tss"], deltacor_interpolated.mean(0))
ax2 = ax.twinx()
ax2.plot(positions_oi  * promoter["strand"] + promoter["tss"], retained_interpolated.mean(0), color = "red", alpha = 0.6)
ax2.set_ylabel("retained")
print(1)

for variant_id, variant in variants.iterrows():
    if variant_id not in haplotypes.loc["rs3087243", "snps"]:
        continue

    ax.scatter(
        variant["position"] * promoter["strand"] + promoter["tss"],
        variant_scores.loc[variant_id, "deltacor"],
        color = haplotypes.loc[variant["snp_main_first"], "color"],
        s = 20,
        marker = "o",
        # transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
    )

ax.invert_yaxis()
ax2.invert_yaxis()
ax.axvline(snp_info.loc["rs3087243", "start"])
# -

effects = xr.concat(variant_scorer.cellgeneeffects, dim = pd.Index(np.arange(len(models)), name = "model"))
effects = effects.mean("model", skipna = True)

effects_raw = effects.sel(gene = gene_id).to_pandas()
effects_smooth = effects_raw.T.values[neighbors].mean(1)
effects_smooth = pd.DataFrame(effects_smooth.T, index = effects_raw.index, columns = effects_raw.columns)

transcriptome.adata.obs = transcriptome.adata.obs.assign(**{"effect_" + snp:effect for snp, effect in effects_smooth.iterrows()})

snps_oi = pd.Series(haplotypes.loc["rs3087243", "snps"])

variant_scores.loc[snps_oi].style.bar(subset = "deltacor")

sc.pl.umap(
    transcriptome.adata,
    color="effect_" + snps_oi,
    vmin=-max_effect,
    vmax=max_effect,
    cmap=mpl.cm.RdBu,
)
