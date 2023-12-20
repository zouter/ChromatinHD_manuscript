# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import torch

import tqdm.auto as tqdm

# %%
import chromatinhd as chd
import tempfile

# %% [markdown]
# ## Get the dataset

# %%
# dataset_name = "pbmc10k/subsets/top250"
dataset_name = "pbmc10kx"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "lymphoma"
# regions_name = "100k100k"
regions_name = "10k10k"
latent = "leiden_0.1"
transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / dataset_name / "transcriptome")
if dataset_name == "pbmc10k/subsets/top250":
    transcriptome = chd.data.Transcriptome(chd.get_output() / "datasets" / "pbmc10k" / "transcriptome")
fragments = chd.flow.Flow.from_path(chd.get_output() / "datasets" / dataset_name / "fragments" / regions_name)
clustering = chd.data.clustering.Clustering(chd.get_output() / "datasets" / dataset_name / "latent" / latent)

folds = chd.data.folds.Folds(chd.get_output() / "datasets" / dataset_name / "folds" / "5x1")
fold = folds[0]

# %%
cluster_counts = pd.DataFrame(transcriptome.X[:], index = transcriptome.obs.index).groupby(clustering.labels).mean()
# pca
import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=cluster_counts.shape[0], whiten = True)
# pca.fit(cluster_counts)
# components = pca.components_
components = pca.fit_transform(cluster_counts)
a, b = 0, 1
plt.scatter(components[:, a], components[:, b])
# plt.scatter(components[a, :], components[b, :])
plt.xlabel(f"PC{a}")
plt.ylabel(f"PC{b}")
plt.title("PCA of cluster counts")
for i, txt in enumerate(cluster_counts.index):
    # plt.annotate(txt, (components[a, i], components[b, i]))
    plt.annotate(txt, (components[i, a], components[i, b]))

# %%
# fragments2 = fragments.filter_regions(
#     fragments.regions.filter(transcriptome.gene_id(["IRF1", "CCL4"]))
# )
# fragments2.create_regionxcell_indptr()
fragments2 = fragments

# %%
import chromatinhd.models.diff.model.binary
model = chd.models.diff.model.binary.Model.create(
    fragments2,
    clustering,
    fold = fold,
    # encoder = "shared_lowrank",
    encoder = "shared",
    # encoder = "split",
    encoder_params=dict(
        delta_regularization=True,
        delta_p_scale = 1.5,
        bias_regularization=True,
        binwidths = (5000, 1000, 500, 100, 50, 25),
        # transcriptome = transcriptome,
    )
)
self = model

# %%
# model.encoder.w_0.weight.data[0, 0] = 1
# model.encoder.w_1.weight.data[0, 2] = 1
# model.encoder.w_1.weight.data[0, 10] = 1
# model.encoder.w_1.weight.data[0, 100] = 1
# w, kl = model.encoder._calculate_w(torch.tensor([0]))

# final_w = w.detach()#.repeat_interleave(model.encoder.total_width // model.encoder.n_final_bins, -1).detach()
# (torch.exp(final_w)[0, 0].repeat_interleave(model.encoder.total_width // model.encoder.n_final_bins, -1)).sum()

# fig, ax = plt.subplots(1, 1, figsize=(20, 2))
# plt.plot((final_w)[0, 0])

# assert set(model.parameters_sparse()).__contains__(model.encoder.w_delta_0.weight)
# assert set(model.parameters_sparse()).__contains__(model.overall_delta.weight)

# %% [markdown]
# ## Test

# %%
loader = chd.loaders.clustering_fragments.ClusteringCuts(fragments, clustering, 50000)

symbol_oi = "IRF1"
minibatch = chd.loaders.minibatches.Minibatch(np.arange(fragments.n_cells), fragments.var.index.get_indexer(transcriptome.gene_id(["IL1B", "CCL4"])))
data = loader.load(minibatch)

coords = torch.clamp(data.cuts.coordinates, self.window[0], self.window[1] - 1) - self.window[0]
bin_ix = coords // self.encoder.binwidths[-1]

# %%
model.forward(data)

# %%
len(list(model.encoder.parameters_sparse()))

# %% [markdown]
# ## Train

# %%
model.train_model(lr = 1e-3)
model.trace.plot()
""

# %%
loader = chd.loaders.clustering_fragments.ClusteringCuts(fragments2, clustering, 500000)

genes_oi = fragments2.var.index[:100]
gene_ixs = fragments2.var.index.get_indexer(genes_oi)
minibatch = chd.loaders.minibatches.Minibatch(fold["cells_validation"], gene_ixs)
data = loader.load(minibatch)

# %%
# multiplier = torch.tensor(1.0, requires_grad=True)
# elbo = model.forward(data, w_delta_multiplier = multiplier)
# elbo.backward()
# multiplier.grad

# %%
# scores = []
# multipliers = np.linspace(0.8, 1.2, 10)
# for multiplier in multipliers:
#     elbo = model.forward(data, w_delta_multiplier = multiplier)
#     scores.append(elbo.item())

# plt.plot(multipliers, scores, marker = ".")

# %% [markdown]
# ## Evaluate

# %%
# gene_ix = fragments2.var.index.get_loc(transcriptome.gene_id("CD74"))

# w_delta = []
# for i in range(len(model.encoder.binwidths)):
#     w_delta.append(getattr(model.encoder, f"w_delta_{i}").get_full_weight()[gene_ix].reshape(clustering.n_clusters, -1).detach().numpy())
# for i, w_delta_level in enumerate(w_delta):
#     print(w_delta_level.std())

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(chd.get_output() / "test_regionpositional2", reset = True)
regionpositional.score(fragments2, clustering, [model], device = "cpu")

# %%
# test whether we got a probability
symbol = "CCL4"
ct = "CD14+ Monocytes"
ct = "NK"
ct_ix = clustering.cluster_info.index.get_loc(ct)

plotdata, plotdata_mean = regionpositional.get_plotdata(transcriptome.gene_id(symbol))
probs = np.exp(plotdata.loc[ct]["prob"])

gene_ix = fragments.var.index.get_loc(transcriptome.gene_id(symbol))
cells_oi = clustering.labels == ct
np.trapz(probs, plotdata.loc[ct].index)* 2/100/100, fragments.counts[cells_oi, gene_ix].mean()

# %% [markdown]
# ### Viz

# %%
symbol = "CCL4"
# symbol = "C5AR1"
# symbol = "NOL3"
# symbol = "IRF1"
# symbol = "IL1B"
# symbol = "CD74"
# symbol = "TCF4"
# symbol = "CD79A"
# symbol = "QKI"
# symbol = "GZMH"

gene_id = transcriptome.gene_id(symbol)

# %%
plotdata, plotdata_mean = regionpositional.get_plotdata(transcriptome.gene_id(symbol))

# %%
plotdata["prob_diff"] = (plotdata.unstack() - plotdata.unstack().mean(0).values[None, :]).stack()["prob"].values
fig, ax = plt.subplots()
ax.plot(plotdata.loc["Plasma"].index, plotdata.loc["Plasma"]["prob_diff"])
# ax.set_xlim(-10000, 10000)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))
width = 10

# window = fragments.regions.window
# window = [-50000, 50000]
window = [-10000, 10000]
# window = [-25000, -15000]
# window = [-25000-10000, -15000+10000]
# window = [-100000, -90000]
# window = [-60000, -50000]
# window = [-50000, -40000]
# window = [-20000, 0]
# window = [0, 100000]

region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, window = window)
fig.main.add_under(panel_genes)

plotdata, plotdata_mean = regionpositional.get_plotdata(transcriptome.gene_id(symbol))

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome=transcriptome, clustering=clustering, gene=transcriptome.gene_id(symbol), panel_height=0.4, order = True
)

panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=clustering.cluster_info, panel_height=0.4, width=width, window = window, order = panel_expression.order, ymax = 10
)

fig.main.add_under(panel_differential)
fig.main.add_right(panel_expression, row=panel_differential)

# motifs_oi = motifscan.motifs.loc[motifscan.motifs.index.str.contains("SPI1") | motifscan.motifs.index.str.contains("PO2") | motifscan.motifs.index.str.contains("TCF")]
# panel_motifs = chd.data.motifscan.plot.Motifs(motifscan, gene_id, motifs_oi, width = width, window = window)

# fig.main.add_under(panel_motifs)

import chromatinhd_manuscript as chdm
panel_peaks = chdm.plotting.Peaks(region, chd.get_output() / "peaks" / dataset_name, window = window, width = width)
fig.main.add_under(panel_peaks)

fig.plot()


# %% [markdown]
# ### Differential slices

# %%
def extract_slices(x, cutoff = 0.):
    selected = (x > cutoff).any(0).astype(int)
    selected_padded = np.pad(selected,((1, 1)))
    start_position_indices,  = np.where(np.diff(selected_padded, axis=-1) == 1)
    end_position_indices,  = np.where(np.diff(selected_padded, axis=-1) == -1)
    start_position_indices = start_position_indices+1
    end_position_indices = end_position_indices+1-1

    data = []
    for start_ix, end_ix in zip(start_position_indices, end_position_indices):
        data.append(x[:, start_ix:end_ix].transpose(1, 0))
    if len(data) == 0:
        data = np.zeros((0, x.shape[0]))
    else:
        data = np.concatenate(data, axis=0)

    return start_position_indices, end_position_indices, data


# %%
class Slices():
    """
    Stores data of slices within regions

    Parameters
    ----------
    region_ixs : np.ndarray
        Region indices
    start_position_ixs : np.ndarray
        Start position indices
    end_position_ixs : np.ndarray
        End position indices
    data : np.ndarray
        Data of slices
    n_regions : int
        Number of regions
    """
    def __init__(self, region_ixs, start_position_ixs, end_position_ixs, data, n_regions):
        self.region_ixs = region_ixs
        self.start_position_ixs = start_position_ixs
        self.end_position_ixs = end_position_ixs
        self.n_regions = n_regions
        self.data = data
        indptr = np.concatenate([[0], np.cumsum(end_position_ixs - start_position_ixs), ], axis=0)
        self.indptr = indptr


# %%
import xarray as xr

prob_cutoff = 0.
# prob_cutoff = -1.
# prob_cutoff = -4.

start_position_ixs = []
end_position_ixs = []
data = []
region_ixs = []
for region, probs in tqdm.tqdm(regionpositional.probs.items()):
    region_ix = fragments.var.index.get_loc(region)
    desired_x = np.arange(*fragments.regions.window) - fragments.regions.window[0]
    x = probs.coords["coord"].values - fragments.regions.window[0]
    y = probs.values

    y_interpolated = chd.utils.interpolate_1d(
        torch.from_numpy(desired_x), torch.from_numpy(x), torch.from_numpy(y)
    ).numpy()

    # from y_interpolated, determine start and end positions of the relevant slices
    start_position_ixs_region, end_position_ixs_region, data_region = extract_slices(y_interpolated, prob_cutoff)
    start_position_ixs.append(start_position_ixs_region)
    end_position_ixs.append(end_position_ixs_region)
    data.append(data_region)
    region_ixs.append(np.ones(len(start_position_ixs_region), dtype=int) * region_ix)
data = np.concatenate(data, axis=0)
start_position_ixs = np.concatenate(start_position_ixs, axis=0)
end_position_ixs = np.concatenate(end_position_ixs, axis=0)
region_ixs = np.concatenate(region_ixs, axis=0)

slices = Slices(region_ixs, start_position_ixs, end_position_ixs, data, fragments.n_regions)

# %%
fc_cutoff = 2.0

# %%
data_diff = slices.data - slices.data.mean(1, keepdims=True)

region_indices = np.repeat(slices.region_ixs, slices.end_position_ixs - slices.start_position_ixs)
position_indices = np.concatenate([np.arange(start, end) for start, end in zip(slices.start_position_ixs, slices.end_position_ixs)])

positions = []
region_ixs = []
cluster_ixs = []
for ct_ix in range(clustering.n_clusters):
    # select which data is relevant
    oi = (data_diff[:, ct_ix] > np.log(fc_cutoff))
    if oi.sum() == 0:
        continue
    positions_oi = position_indices[oi]
    regions_oi = region_indices[oi]

    start = (
        np.where(
            np.pad(np.diff(positions_oi) != 1, (1, 0), constant_values = True) |
            np.pad(np.diff(regions_oi) != 0, (1, 0), constant_values = True)
    )[0])
    end = np.pad(start[1:], (0, 1), constant_values = len(positions_oi)) - 1

    positions.append(np.stack([positions_oi[start], positions_oi[end]], axis=1))
    region_ixs.append(regions_oi[start])
    cluster_ixs.append(np.ones(len(start), dtype=int) * ct_ix)
start_position_ixs, end_position_ixs = np.concatenate(positions, axis=0).T
region_ixs = np.concatenate(region_ixs, axis=0)
cluster_ixs = np.concatenate(cluster_ixs, axis=0)


# %%
class ClusterSlices():
    """
    Stores data of slices within regions linked to a specific cluster

    Parameters
    ----------
    region_ixs : np.ndarray
        Region indices
    cluster_ixs : np.ndarray
        Cluster indices
    start_position_ixs : np.ndarray
        Start position indices
    end_position_ixs : np.ndarray
        End position indices
    data : np.ndarray
        Data of slices
    n_regions : int
        Number of regions
    """
    def __init__(self, region_ixs, cluster_ixs, start_position_ixs, end_position_ixs, data, n_regions):
        self.region_ixs = region_ixs
        self.start_position_ixs = start_position_ixs
        self.end_position_ixs = end_position_ixs
        self.n_regions = n_regions
        self.cluster_ixs = cluster_ixs
        self.data = data
        indptr = np.concatenate([[0], np.cumsum(end_position_ixs - start_position_ixs), ], axis=0)
        self.indptr = indptr

    def get_slice_scores(self, regions = None, clustering = None):
        slicescores = pd.DataFrame(
            {
                "start":self.start_position_ixs,
                "end":self.end_position_ixs,
                "region_ix":self.region_ixs,
                "cluster_ix":self.cluster_ixs,
            }
        )
        if clustering is not None:
            slicescores["cluster"] = pd.Categorical(clustering.cluster_info.index[self.cluster_ixs], clustering.cluster_info.index)
        if regions is not None:
            slicescores["region"] = pd.Categorical(fragments.regions.coordinates.index[self.region_ixs], fragments.regions.coordinates.index)
        slicescores["length"] = slicescores["end"] - slicescores["start"]
        return slicescores


# %%
cluster_slices = ClusterSlices(region_ixs, cluster_ixs, start_position_ixs, end_position_ixs, data_diff, fragments.n_regions)

# %%
# fig, ax = plt.subplots()
# plotdata, plotdata_mean = regionpositional.get_plotdata(transcriptome.gene_id(symbol))
# ax.plot(plotdata.loc[ct].index, plotdata.loc[ct]["prob"])

# for start_ix, end_ix in zip(start_position_ixs[oi], end_position_ixs[oi]):
#     ax.axvspan(start_ix+fragments.regions.window[0], end_ix+fragments.regions.window[0], color="red", alpha=0.2)
# ax.set_xlim(-10000, 10000)

# %%
import chromatinhd.models.diff.interpret.differential
# differential = chromatinhd.models.diff.interpret.differential.DifferentialSlices(positions, regions, clusters, fragments.regions.window, fragments.var.shape[0], clustering.var.shape[0])

# %%
slicescores = cluster_slices.get_slice_scores(regions = fragments.regions, clustering = clustering)
slicescores.groupby("cluster")["length"].sum().sort_values(ascending = False)

# %%
# slicescores = pd.DataFrame(
#     {
#         "start":positions[:, 0],
#         "end":positions[:, 1],
#         "region":pd.Categorical(fragments.regions.coordinates.index[regions], fragments.regions.coordinates.index),
#         "region_ix":regions,
#         "cluster":pd.Categorical(clustering.cluster_info.index[clusters], clustering.cluster_info.index),
#     }
# )
# # slicescores = slicescores.loc[(slicescores["start"] > -10000 - fragments.regions.window[0]) & (slicescores["end"] < 10000 - fragments.regions.window[0])]
# slicescores["length"] = slicescores["end"] - slicescores["start"]


# %%
motifscan_name = "hocomocov12_1e-3"
# motifscan_name = "hocomocov12_5e-4"
# motifscan_name = "hocomocov12_1e-4"
motifscan = chd.data.motifscan.Motifscan(chd.get_output() / "datasets" / dataset_name / "motifscans" / regions_name / motifscan_name)

# %%
# slicescores.query("region_ix == 4").query("cluster == 'NK'")["start"].plot()

# %%
# count motifs in each slice
window_width = fragments.regions.width
motif_counts = np.zeros((len(slicescores), motifscan.n_motifs), dtype=int)
for i, (relative_start, relative_end, region_ix) in enumerate(zip(slicescores["start"], slicescores["end"], slicescores["region_ix"])):
    start_ix = region_ix * window_width + relative_start
    end_ix = region_ix * window_width + relative_end
    motif_indices = motifscan.indices[motifscan.indptr[start_ix] : motifscan.indptr[end_ix]]
    motif_counts[i] = (np.bincount(motif_indices, minlength=motifscan.n_motifs))
motif_counts.sum()

# %%
import scipy.stats

# %%
scores = []
for cluster in tqdm.tqdm(clustering.cluster_info.index):
    x_foreground = motif_counts[slicescores["cluster"] == cluster].sum(0)
    x_background = motif_counts[slicescores["cluster"] != cluster].sum(0)
    n_foreground = slicescores["length"][slicescores["cluster"] == cluster].sum()
    n_background = slicescores["length"][slicescores["cluster"] != cluster].sum()

    contingencies = np.stack([
        n_background - x_background,
        x_background,
        n_foreground - x_foreground,
        x_foreground,
    ], axis=1).reshape(-1, 2, 2).astype(np.int64)

    odds = (contingencies[:, 1, 1] * contingencies[:, 0, 0]) / (contingencies[:, 1, 0] * contingencies[:, 0, 1])

    p_values = np.array([scipy.stats.chi2_contingency(c).pvalue if (c > 0).all() else 1. for c in contingencies])
    q_values = chd.utils.fdr(p_values)

    scores.append(pd.DataFrame(
        {
            "cluster":cluster,
            "odds":odds,
            "p_value":p_values,
            "q_value":q_values,
            "motif":motifscan.motifs.index,
            "contingency":[c for c in contingencies],
            "n_foreground":n_foreground,
        }
    ))
scores = pd.concat(scores, axis=0).set_index(["cluster", "motif"])

# %%
motifs_oi = motifscan.motifs.loc[(motifscan.motifs["datatype"] != "M")].sort_values("quality").copy().reset_index().groupby("symbol").first().reset_index().set_index("motif")
motifs_oi["gene"] = [transcriptome.gene_id(symbol) if symbol in transcriptome.var["symbol"].tolist() else None for symbol in motifs_oi["symbol"]]
motifs_oi = motifs_oi.dropna(subset=["gene"])
cluster_transcriptome = pd.DataFrame(transcriptome.layers["magic"][:], index = transcriptome.obs.index, columns = transcriptome.var.index).groupby(clustering.labels).mean()
diffexp = cluster_transcriptome - cluster_transcriptome.mean(0)

# %%
cors = []
for cluster in clustering.cluster_info.index:
    subscores = pd.DataFrame({
        "lfc":diffexp.loc[cluster, motifs_oi["gene"]],
        "odds":scores.loc[cluster].loc[motifs_oi.index]["odds"].values,
    })
    subscores["logodds"] = np.log(subscores["odds"])
    subscores = subscores.dropna()
    subscores = subscores.query("abs(lfc) > log(1.2)")
    if len(subscores) > 5:
        contingency = np.array([
            [subscores.query("lfc > 0").query("logodds > 0").shape[0], subscores.query("lfc > 0").query("logodds < 0").shape[0]],
            [subscores.query("lfc < 0").query("logodds > 0").shape[0], subscores.query("lfc < 0").query("logodds < 0").shape[0]],
        ])
        odds = (contingency[1, 1] * contingency[0, 0] + 1) / (contingency[1, 0] * contingency[0, 1] + 1)
        agreement = ((subscores["logodds"] * subscores["lfc"]) > 0).mean()
        expected_agreement = ((subscores["logodds"] * subscores.iloc[np.random.choice(len(subscores), len(subscores))]["lfc"]) > 0).mean()
        cors.append({
            "cluster":cluster,
            "cor":(np.corrcoef(subscores["lfc"], subscores["logodds"])[0, 1]),
            "spearman":(scipy.stats.spearmanr(subscores["lfc"], subscores["logodds"])[0]),
            "expected_agreement":expected_agreement,
            "agreement":agreement,
            "agreement_ratio":agreement / expected_agreement,
            "odds":odds,
            "log_odds":np.log(odds),
        })
cors = pd.DataFrame(cors).set_index("cluster")
cors["log_agreement_ratio"] = np.log(cors["agreement_ratio"])

# %%
clustering.cluster_info["n_cells"] = clustering.labels.value_counts()
cors["n_cells"] = clustering.cluster_info["n_cells"]

# %%
cors.sort_values("n_cells").style.bar()

# %%
cors.mean()

# %%
fasta_file = "/data/genome/GRCh38/GRCh38.fa"
onehots = chd.data.motifscan.motifscan.create_region_onehots(fragments.regions, fasta_file)

# %%

# %%
cors.mean()

# %%
scores.sort_values("q_value", ascending = True).query("motif in @motifs_oi.index").head(30)

# %%
import scipy.stats

# %%
scores = pd.DataFrame({
    "odds":odds,
    # "p":scipy.stats.fisher_exact(contingencies)[1],
    "motif":motifscan.motifs.index,
})

# %%
scores.sort_values("odds")

# %%
# enrichment = chd.models.diff.enrichment.enrich_cluster_vs_clusters(motifscan, fragments.regions.window, slicescores, n_regions = fragments.n_regions)

# %%
cluster_id = ["memory B", "naive B"]
cluster_ixs = clustering.cluster_info.index.get_indexer(cluster_id)

# %%
level = 0
location = -10000
w_delta = getattr(model.encoder, f"w_delta_{level}").weight.data.reshape(fragments.n_regions, clustering.n_clusters, -1)

bin_ix = (location - self.window[0]) // self.encoder.binwidths[level]

for w in [-0.5, -0.1, 0., 0.1, 0.5, 1., -200.0]:
    w_delta[gene_ix, cluster_ixs, bin_ix] = w

    getattr(model.encoder, f"w_delta_{level}").weight.data[:] = w_delta.reshape(getattr(model.encoder, f"w_delta_{level}").weight.data.shape)

    for phase in ["validation"]:
        cell_ixs = fold["cells_" + phase]
        prediction = model.get_prediction(cell_ixs = cell_ixs, regions = [gene_id])
        print(len(cell_ixs), prediction["likelihood"].sum().item() / len(cell_ixs))

# %%
minibatch = chd.loaders.minibatches.Minibatch(fold["cells_test"], np.array([gene_ix]))
data = loader.load(minibatch)
model.forward(data)
