# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
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

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

# %%
dataset_name = "pbmc10k"
dataset_folder = chd.get_output() / "datasets" / dataset_name

# %%
fragments = chd.data.fragments.Fragments(dataset_folder / "fragments" / "100k100k")
regions = fragments.regions

transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")

# %%
folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x5")

# %%
model = chd.models.pred.model.additive.Model.restore(chd.get_output() / "pred" / "pbmc10k" / "100k100k" / "5x5" / "normalized" / "v20" / "0")
fold = folds[0]

# %%
prediction = model.get_prediction(cell_ixs=np.hstack([fold["cells_test"], fold["cells_validation"], fold["cells_train"]]))

# %%
predicted = prediction["predicted"]
expected = prediction["expected"]

# %%
cors = chd.utils.paircor(predicted.to_pandas(), expected.to_pandas())
cors

# %%
random = []
for i in tqdm.tqdm(range(100)):
    ixs_random = np.random.choice(expected.shape[0], size = expected.shape[0], replace = True)
    random.append(chd.utils.paircor(expected.values[ixs_random], predicted.values[ixs_random]))
random = np.stack(random)

# %%
genescores = pd.DataFrame(
    {
        "n_fragments":prediction["n_fragments"].sum("cell").to_pandas(),
        "cor":cors,
        "n_nonzero":(transcriptome.layers["normalized"][:] > 0).sum(0),
        "normalized_dispersion":transcriptome.var["dispersions_norm"],
        # "n_nonzero":(transcriptome.layers["normalized"][:]).sum(0)
    }
)

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (6, 2), sharey = True)
ax0.scatter(genescores["n_nonzero"], genescores["cor"], s = 1, alpha = 0.3)
ax0.set_xscale("log")
ax0.set_xlabel("Number of cells with\nnon-zero expression")
ax0.set_ylabel("Pearson correlation")

ax1.scatter(genescores["n_fragments"], genescores["cor"], s = 1, alpha = 0.3)
ax1.set_xscale("log")
ax1.set_xlabel("Number of fragments")

ax2.scatter(genescores["normalized_dispersion"], genescores["cor"], s = 1, alpha = 0.3)
ax2.set_xscale("log")
ax2.set_xlabel("Normalized dispersion")

# %%
(cors - 0.1).abs().sort_values()

# %%
pd.Series(np.abs(cors.values - 0.1), index = transcriptome.var.symbol).sort_values().head(20)

# %%
pd.Series(cors.values, index = transcriptome.var.symbol).sort_values()

# %%
gene_oi = transcriptome.gene_id("IL1B")
gene_oi = transcriptome.gene_id("IGSF8")
# gene_oi = "ENSG00000214013"

# %%
import scanpy as sc
sc.pl.umap(transcriptome.adata, color = gene_oi)

# %%
np.quantile(random[:, transcriptome.gene_ix(transcriptome.symbol(gene_oi))], [0.025, 0.975])

# %%
plotdata = pd.DataFrame({
    "expected":expected.sel(gene = gene_oi),
    "predicted":predicted.sel(gene = gene_oi),
    "n_fragments":prediction["n_fragments"].sel(gene = gene_oi)
})
plotdata["predicted"] = plotdata["predicted"] / plotdata["predicted"].std() * plotdata["expected"].std()

norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
cmap = mpl.colormaps["Set1"]

fig, ax = plt.subplots(figsize = (2, 2))
ax.scatter(plotdata["expected"], plotdata["predicted"], s = 0.1, color = cmap(norm(plotdata["n_fragments"])))
ax.set_xlabel("Observed")
ax.set_ylabel("Predicted")
cor = np.corrcoef(plotdata["expected"], plotdata["predicted"])[0, 1]
ci = np.quantile(random[:, transcriptome.gene_ix(transcriptome.symbol(gene_oi))], [0.025, 0.975])
ax.set_title(f"{transcriptome.symbol(gene_oi)}\nCorrelation: {cor:.3f}\n95% CI=[{ci[0]:.3f},{ci[1]:.3f}]")

# %% [markdown]
# ## Similarity in prediction

# %%
gene_oi = transcriptome.gene_id("PPP1R13B")
gene_oi = transcriptome.gene_id("IGSF8")

# %%
fragments = chd.data.fragments.Fragments(dataset_folder / "fragments" / "10k10k")
regions = fragments.regions

transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")

# %%
model_folder = chd.get_output() / "pred" / "pbmc10k" / "10k10k" / "5x5" / "normalized" / "v20"

models = [chd.models.pred.model.additive.Model(model_folder / str(model_ix)) for model_ix in range(25)]
folds = chd.data.folds.Folds(chd.get_output() / "datasets" / "pbmc10k" / "folds" / "5x5")

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)

regionmultiwindow1 = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test", reset = True)
regionmultiwindow1.score(fragments, transcriptome, models[:10], folds[:10], censorer, regions = [gene_oi])
regionmultiwindow1.interpolate()

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)

regionmultiwindow2 = chd.models.pred.interpret.RegionMultiWindow(chd.get_output() / "test2", reset = True)
regionmultiwindow2.score(fragments, transcriptome, models[10:], folds[10:], censorer, regions = [gene_oi])
regionmultiwindow2.interpolate()

# %%
region = fragments.regions.coordinates.loc[gene_oi]
symbol_oi = transcriptome.var.loc[gene_oi, "symbol"]

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height = 0))

binwidth = (regionmultiwindow.design["window_end"] - regionmultiwindow.design["window_start"]).iloc[0]

window = fragments.regions.window
# window = [-10000, 20000]
window = [-10000, 10000] # TSS
# window = [-20000, -10000] # PFKFB3 enhancer

panel, ax = fig.main.add_under(chd.plot.genome.Genes.from_region(region, width = 10, window = window))
ax.set_xlim(*window)

panel, ax = fig.main.add_under(chd.models.pred.plot.Predictivity(regionmultiwindow1.get_plotdata(gene_oi), window = window, width = 10, color_by_effect=False))
ax.set_ylabel("$\Delta$ cor\nDataset 1\n(10k cells)", ha = "right")
ax.set_ylim(0, -0.05)
panel, ax = fig.main.add_under(chd.models.pred.plot.Predictivity(regionmultiwindow2.get_plotdata(gene_oi), window = window, width = 10, color_by_effect=False))
ax.set_ylabel("$\Delta$ cor\nDataset 2\n(3k cells)", ha = "right")
ax.set_ylim(0, -0.05)

fig.plot()

# %%
fig, ax = plt.subplots()
ax.scatter(
    regionmultiwindow1.scores[gene_oi]["deltacor"].mean("model").to_pandas(),
    regionmultiwindow2.scores[gene_oi]["deltacor"].mean("model").to_pandas(),
)

# %%
regionmultiwindow
