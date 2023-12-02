# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
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
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
genome = "GRCh38"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

dataset_folder = chd.get_output() / "datasets" / "pbmc10k"

# %%
fragments = chd.data.fragments.Fragments(dataset_folder / "fragments" / "10k10k")
# fragments = chd.data.fragments.Fragments(dataset_folder / "fragments" / "100k100k")
regions = fragments.regions

transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")

# %%
folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x5")
folds.sample_cells(fragments, 5, 1)
folds = folds

# %%
models_folder = chd.get_output() / "models" / dataset_name

# %%
import chromatinhd.models.pred.interpret

# %%
n_epochs = 60

# %%
deltacors = []
models_dummy = []
for fold in folds:
    model = chd.models.pred.model.additive.Model(fragments = fragments, transcriptome = transcriptome, dummy = True, layer = "normalized", reset = True)
    model.train_model(fold = fold, n_epochs = n_epochs, pbar = True)
    models_dummy.append(model)

performance_dummy = chd.models.pred.interpret.Performance(models_folder / "dummy" / "interpretation" / "performance", reset = True)
performance_dummy.score(fragments, transcriptome, models_dummy, folds)

# %%
deltacors = []
models = []
for fold in folds:
    model = chd.models.pred.model.additive.Model(fragments = fragments, transcriptome = transcriptome, layer = "normalized", reset = True)
    model.fragment_embedder.weight1.weight.data[:] = 0.
    model.train_model(fold = fold, n_epochs = n_epochs, pbar = True)
    models.append(model)

performance = chd.models.pred.interpret.Performance(models_folder / "full" / "interpretation" / "performance", reset = True)
performance.score(fragments, transcriptome, models, folds)

# %%
performance = chd.models.pred.interpret.Performance(models_folder / "full" / "interpretation" / "performance", reset = True)
performance.score(fragments, transcriptome, models, folds)

# %%

performance_dummy = chd.models.pred.interpret.Performance(models_folder / "dummy" / "interpretation" / "performance", reset = True)
performance_dummy.score(fragments, transcriptome, models_dummy, folds)

# %%
fig, ax = plt.subplots()
plt.scatter(performance_dummy.genescores["cmse"].sel(phase = "test").mean("model"), performance.genescores["cmse"].sel(phase = "test").mean("model"))
ax.plot([0, 1], [0, 1])

# %%
fig, ax = plt.subplots()
plt.scatter(performance_dummy.genescores["cmse"].sel(phase = "test").mean("model"), performance_dummy.genescores["cor"].sel(phase = "test").mean("model"))
ax.plot([0, 1], [0, 1])

# %% [markdown]
# ## Simulate

# %%
import scanpy as sc

# %% [markdown]
# ### Add parameters to var

# %%
adata = transcriptome.adata.copy()
adata.X = adata.layers["normalized"]
sc.pp.highly_variable_genes(adata)

# %%
from fit_nbinom import fit_nbinom
params = []
for i in tqdm.tqdm(range(adata.var.shape[0])):
    X = np.array(adata.layers["counts"][:, i].todense())[:, 0]
    params.append(fit_nbinom(X))
params = pd.DataFrame(params, index = adata.var.index)

# %%
transcriptome.var["mean"] = adata.var["means"].values
transcriptome.var["size"] = params["size"].values
transcriptome.var["mean_bin"] = np.digitize(np.log(transcriptome.var["mean"]), bins = np.quantile(np.log(transcriptome.var["mean"]), np.linspace(0, 1, 10))) - 1
transcriptome.var["size_mean"] = transcriptome.var.groupby("mean_bin")["size"].median()[transcriptome.var["mean_bin"]].values
transcriptome.var["dispersion_mean"] = transcriptome.var["size_mean"]
transcriptome.var = transcriptome.var

# %% [markdown]
# ### Simulate

# %%
mu = ((fragments.counts + 1e-5) / (fragments.counts.mean(0, keepdims=True) + 1e-5)) * transcriptome.var["mean"].values
dispersion = transcriptome.var["dispersion_mean"].values

total_count, logits = transform_parameters(mu, dispersion)
probs = 1/(1+np.exp(logits))

# %%
transcriptome_simulated = TranscriptomeSimulated(fragments, transcriptome)
simulation = Simulation(transcriptome_simulated)

performance_simulation = chd.models.pred.interpret.Performance().score(fragments, transcriptome_simulated, [simulation] * len(folds), folds, phases = ["test"])

# %%
noisyexpression_cors = []
noisyexpression_cmse = []
for i in tqdm.trange(10):
    expression = np.log1p(np.random.negative_binomial(total_count, probs))
    noisyexpression_cors.append(chd.utils.paircor(mu, expression))
    noisyexpression_cmse.append(chd.utils.paircmse(mu, expression))
noisyexpression_cors = np.stack(noisyexpression_cors)
noisyexpression_cmse = np.stack(noisyexpression_cmse)

# %%
noisyboth_cors = []
noisyboth_cmse = []
for i in tqdm.trange(10):
    expression = np.log1p(np.random.negative_binomial(total_count, probs))
    atac = np.random.poisson(mu_atac)
    noisyboth_cors.append(chd.utils.paircor(atac, expression))
    noisyboth_cmse.append(chd.utils.paircmse(atac, expression))
noisyboth_cors = np.stack(noisyboth_cors)
noisyboth_cmse = np.stack(noisyboth_cmse)

# %%
noisyatac_cors = []
noisyatac_cmse = []
for i in tqdm.trange(10):
    expression = mu
    atac = np.random.poisson(mu_atac)
    noisyatac_cors.append(chd.utils.paircor(atac, expression))
    noisyatac_cmse.append(chd.utils.paircmse(atac, expression))
noisyatac_cors = np.stack(noisyatac_cors)
noisyatac_cmse = np.stack(noisyatac_cmse)

# %%
noisyboth50_cors = []
noisyboth50_cmse = []
for i in tqdm.trange(10):
    expression = np.log1p(np.random.negative_binomial(total_count, probs))
    atac = np.random.poisson(mu_atac*0.25)
    noisyboth50_cors.append(chd.utils.paircor(atac, expression))
    noisyboth50_cmse.append(chd.utils.paircmse(atac, expression))
noisyboth50_cors = np.stack(noisyboth50_cors)
noisyboth50_cmse = np.stack(noisyboth50_cmse)

# %%
plt.scatter(noisyatac_cors.mean(0), noisyboth_cors.mean(0))

# %% [markdown]
# ## Gene body measures

# %%
fragments.regions.coordinates

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb"))

# %%
import torch_scatter
import torch
import xarray as xr

# %%
model = V21(fragments, transcriptome, selected_transcripts)


# %%
performance_v21 = chd.models.pred.interpret.Performance().score(fragments, transcriptome, [model] * len(folds), folds, phases = ["test"])

# %%
performance_v21.genescores["cor"].sel(phase = "test").mean("model").to_dataframe()["cor"].mean()

# %%
coord = fragments.coordinates[:].mean(1)
within = (coord > -5000) & (coord < 5000)
weight = (1+ np.e**(-1)) * within + (np.exp(-abs(coord/5000)) + np.e**(-1)) * ~within

import torch
weight = torch.tensor(weight)

cellxregion_ix = torch.from_numpy(fragments.mapping[:, 0] * fragments.n_regions + fragments.mapping[:, 1])

predicted = torch_scatter.segment_sum_coo(weight, cellxregion_ix.long(), dim_size = fragments.n_cells * fragments.n_regions).reshape(fragments.n_cells, fragments.n_regions).numpy()
cors_21 = chd.utils.paircor(predicted, transcriptome.layers["normalized"][:])

# %% [markdown]
# ## Compare

# %%
cors = pd.DataFrame({
    "42": cors_42,
    "21": cors_21,
    "noisy_atac": noisyatac_cors.mean(0),
    "noisy_both": noisyboth_cors.mean(0),
    "noisy_both_50": noisyboth50_cors.mean(0),
    "noisy_expression": noisyexpression_cors.mean(0),
    "observed": performance.genescores["cor"].sel(phase = "test").mean("model").to_pandas(),
    "dummy": performance_dummy.genescores["cor"].sel(phase = "test").mean("model").to_pandas(),
})

cmses = pd.DataFrame({
    "noisy_atac": noisyatac_cmse.mean(0),
    "noisy_both": noisyboth_cmse.mean(0),
    "noisy_both_50": noisyboth50_cmse.mean(0),
    "noisy_expression": noisyexpression_cmse.mean(0),
    "observed": performance.genescores["cmse"].sel(phase = "test").mean("model").to_pandas(),
    "dummy": performance_dummy.genescores["cmse"].sel(phase = "test").mean("model").to_pandas(),
})

# %%
methods = pd.DataFrame([
    ["42", "baseline"],
    ["21", "baseline"],
    ["dummy", "baseline"],
    ["noisy_atac", "simulated"],
    ["noisy_both", "simulated"],
    ["noisy_both_50", "simulated"],
    ["noisy_expression", "simulated"],
    ["observed", "ours"],
], columns = ["method", "methodset"]).set_index("method")
methodsets = pd.DataFrame([
    ["ours", "#1f77b4"],
    ["baseline", "grey"],
    ["simulated", "black"],
], columns = ["methodset", "color"]).set_index("methodset")
for methodset_id, methodset in methods.groupby("methodset"):
    if methodset_id == "baseline":
        palette = sns.color_palette("Greens", n_colors = len(methodset))
    elif methodset_id == "ours":
        palette = sns.color_palette("Blues", n_colors = len(methodset))
    elif methodset_id == "simulated":
        palette = sns.color_palette("Reds", n_colors = len(methodset))
    methods.loc[methodset.index, "color"] = methods.loc[methodset.index, "color"] = pd.Series(list(palette), methodset.index)

# %%
fig, ax= plt.subplots(figsize = (4, 2))
plotdata = cors.iloc[:]
for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    sns.ecdfplot(cor, label = key, color = color)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plotdata["noisy_both_50"].mean(), plotdata["observed"].mean()


# %%
fig, ax= plt.subplots(figsize = (4, 2))
plotdata = cors.iloc[-1000:]
for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    sns.ecdfplot(cor, label = key, color = color)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plotdata["noisy_both"].mean(), plotdata["noisy_both_50"].mean(), plotdata["observed"].mean(), plotdata["42"].mean(), plotdata["dummy"].mean()

# %%
gene_ranking = transcriptome.adata.var["dispersions_norm"].sort_values(ascending = False).index
# gene_ranking = transcriptome.adata.var["means"].sort_values(ascending = False).index
# gene_ranking = pd.Series(fragments.counts.mean(0), index = transcriptome.var.index).sort_values(ascending = False).index
plotdata = cors.loc[gene_ranking]
plotdata = (np.cumsum(plotdata) / np.arange(1, plotdata.shape[0] + 1)[:, None])

fig, ax = plt.subplots()

for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    plt.plot(np.arange(cor.shape[0]), cor, label = key, color = color)
ax.legend()


# %%
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plotdata = cors.loc[gene_ranking]
plotdata = (plotdata["observed"].values[:, None] - plotdata[["dummy", "42", "21"]])

fig, ax = plt.subplots()

for key, cor in plotdata.items():
    color = methods.loc[key, "color"]
    ax.scatter(np.arange(cor.shape[0]), cor, label = key, color = color, s= 0.3)
    plotdata_mean = plotdata[key].rolling(1000).mean().fillna(np.cumsum(plotdata[key])/np.arange(1, plotdata.shape[0] + 1))
    ax.plot(np.arange(plotdata.shape[0])[100:], plotdata_mean[100:], color = color)
ax.axhline(0.)
ax.legend()

# %%
sns.ecdfplot(transcriptome.var["dispersions_norm"])

# %%
fig, ax = plt.subplots()

reference_id = "noisy_both_50"
baseline_id = "42"
observed_id = "observed"
plotdata = cors.iloc[-1000:]

segs = np.stack([plotdata[[reference_id, reference_id]], plotdata[[baseline_id, observed_id]]], -1)
line_segments = mpl.collections.LineCollection(segs, linestyle='solid', alpha = 0.1, color = "black")
ax.add_collection(line_segments)
ax.scatter(plotdata[reference_id], plotdata[observed_id], c = "k", s = 3)
ax.scatter(plotdata[reference_id], plotdata[baseline_id], c = "k", s = 10, marker = "_", alpha = 0.1)

xs_perfect = np.linspace(0, 1, 100)
ys_perfect = xs_perfect
ax.plot(xs_perfect, ys_perfect, color = "#333", linestyle = "dashed")
