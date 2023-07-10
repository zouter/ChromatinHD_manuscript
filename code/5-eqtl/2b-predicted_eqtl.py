# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import tempfile
import requests
import xarray as xr

# %%
fragment_dataset_name = "pbmc10k_leiden_0.1"
dataset_name = fragment_dataset_name + "_gwas"

# %%
genotype_data_folder = chd.get_output() / "data" / "eqtl" / "onek1k"

# %%
dataset_folder = chd.get_output() / "datasets" / "genotyped" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Predictive model

# %% [markdown]
# ### Load direct model

# %% [markdown]
# ### Data

# %%
variantxgene_effect = chd.load(pathlib.Path("variantxgene_effect.pkl").open("rb"))
model_direct = chd.load(pathlib.Path("model_direct.pkl").open("rb"))

# %%
fragments = chd.data.fragments.ChunkedFragments(dataset_folder / "fragments")
genes = pd.read_csv(dataset_folder / "genes.csv", index_col=0)
transcriptome = chd.data.transcriptome.transcriptome.ClusteredTranscriptome(
    dataset_folder / "transcriptome"
)
genotype = chd.data.genotype.genotype.Genotype(dataset_folder / "genotype")
# gene_variants_mapping = pickle.load((dataset_folder / "gene_variants_mapping.pkl").open("rb"))
gene_variants_mapping = pickle.load(
    pathlib.Path("gene_variants_mapping.pkl").open("rb")
)

# %%
variantxgene_index = []
for gene, gene_ix in zip(transcriptome.var.index, transcriptome.var["ix"]):
    variantxgene_index.extend(
        [
            [gene, genotype.variants_info.index[variant_ix]]
            for variant_ix in gene_variants_mapping[gene_ix]
        ]
    )
variantxgene_index = pd.MultiIndex.from_frame(
    pd.DataFrame(variantxgene_index, columns=["gene", "variant"])
)
variantxgenes_info = pd.DataFrame(index=variantxgene_index)

# %%
ground_truth_variantxgene_effect = chd.load(
    pathlib.Path("ground_truth_variantxgene_effect.pkl").open("rb")
)
ground_truth_bf = chd.load(pathlib.Path("ground_truth_bf.pkl").open("rb"))
scores_eqtl = pd.read_pickle("scores.pkl")
scores_eqtl["significant"] = scores_eqtl["bf"] > np.log(10)
ground_truth_significant = (
    scores_eqtl["significant"].unstack().loc[variantxgene_index].values.T
)

# %%
import torch

# %%
ground_truth_prioritization = np.clip(
    ground_truth_variantxgene_effect / variantxgene_effect.values[None, :], 0, 1
).to(torch.float32)

# %%
variantxgenes_info["tss_distance"] = (
    genes["tss_position"][variantxgene_index.get_level_values("gene")].values
    - genotype.variants_info["position"]
    .loc[variantxgene_index.get_level_values("variant")]
    .values
)

# %%
import chromatinhd.models.eqtl.prediction.v2 as prediction_model

# %%
loader = prediction_model.Loader(
    transcriptome, genotype, fragments, gene_variants_mapping, variantxgenes_info
)
loaders = chd.loaders.LoaderPool(
    prediction_model.Loader,
    loader=loader,
    n_workers=10,
    shuffle_on_iter=True,
)
loaders_validation = chd.loaders.LoaderPool(
    prediction_model.Loader,
    loader=loader,
    n_workers=5,
    shuffle_on_iter=False,
)

# %%
all_genes = transcriptome.var.query("chr != 'chr6'")

validation_chromosomes = ["chr1"]
train_genes = all_genes.query("chr not in @validation_chromosomes")
validation_genes = all_genes.query("chr in @validation_chromosomes")
# train_genes = all_genes.query("symbol == 'CTLA4'")

# %%
minibatches_train = prediction_model.loader.create_bins_ordered(
    train_genes["ix"].values, n_genes_step=300
)
len(minibatches_train)

minibatches_validation = prediction_model.loader.create_bins_ordered(
    validation_genes["ix"].values, n_genes_step=300
)

minibatches_full = prediction_model.loader.create_bins_ordered(
    transcriptome.var["ix"].values, n_genes_step=300
)

len(minibatches_train), len(minibatches_validation)


# %% [markdown]
# #### Test loader

# %%
import torch

# %%
# fragments.chunkcoords = fragments.chunkcoords.to(torch.int64)

# %%
loader = prediction_model.Loader(
    transcriptome,
    genotype,
    fragments,
    gene_variants_mapping,
    variantxgenes_info,
    window_size=20000,
)

# %%
import copy

# %%
# genes_oi = np.array(np.arange(100).tolist() + transcriptome.var.loc[transcriptome.gene_id(["CTLA4", "BACH2", "BLK"]), "ix"].tolist())
genes_oi = np.array(
    transcriptome.var.loc[transcriptome.gene_id(["CTLA4", "CDK1"]), "ix"]
)

minibatch = prediction_model.Minibatch(genes_oi)

data = loader.load(minibatch)

# %%
import torch_scatter

# %%
# data.variants_oi.tolist().index(genotype.variants_info.loc[genotype.variants_info["rsid"] == 'rs4987360']["ix"][0])

# %%
# data = data.to("cpu")
# model_pred = model_pred.to("cpu")
# normalized = torch.diff(data.local_clusterxvariant_indptr).reshape(data.n_clusters, data.n_variants) / model_pred.variant_embedder.cluster_cut_lib.unsqueeze(-1)
# sns.heatmap(normalized.cpu().detach())

# %% [markdown]
# ### Model

# %%
model_pred = prediction_model.Model.create(
    transcriptome,
    genotype,
    fragments,
    gene_variants_mapping,
    variantxgene_effect=variantxgene_effect,
    reference_expression_predictor=model_direct.expression_predictor,
    ground_truth_variantxgene_effect=ground_truth_variantxgene_effect,
    ground_truth_significant=torch.from_numpy(
        ground_truth_significant.astype(np.float32)
    ),
    ground_truth_prioritization=ground_truth_prioritization,
    ground_truth_bf=ground_truth_bf,
    window_size=loader.window_size,
    dummy=True,
    # dumb = True,
)

# %%
data = data.to("cpu")

# %%
model_pred.forward(data)

# %%
# variant_ix = genotype.variants_info.query("rsid == 'rs3087243'")["ix"].tolist()[0]

genes_oi = transcriptome.var.loc[
    transcriptome.gene_id(["CTLA4", "BACH2", "CTLA4"]), "ix"
].values
# genes_oi = transcriptome.var["ix"][[500]].values

minibatch = prediction_model.Minibatch(genes_oi)

data = loader.load(minibatch)
# data.variants_oi.tolist().index(variant_ix)

# %%
model_pred = model_pred.to("cpu")

# %%
x = model_pred.variant_embedder.forward(
    data.relative_coordinates,
    data.local_clusterxvariant_indptr,
    data.n_variants,
    data.n_clusters,
)

# %%
assert x.shape[-1] == model_pred.variant_embedder.n_embedding_dimensions


# %%
def paircor(x, y, dim=0, eps=1e-10):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(
        dim
    ) / divisor
    return cor


model_pred = model_pred.to("cpu")

cors = []
minibatch = prediction_model.Minibatch(transcriptome.var["ix"].values)
data = loader.load(minibatch)
x = model_pred.variant_embedder.forward(
    data.relative_coordinates,
    data.local_clusterxvariant_indptr,
    data.n_variants,
    data.n_clusters,
)

y = x[..., -1].detach()[:, data.local_variant_to_local_variantxgene_selector]
z = np.abs(
    model_pred.ground_truth_variantxgene_effect[:, data.variantxgene_ixs].detach()
)
# z = model_pred.ground_truth_significant[
#     :, data.variantxgene_ixs
# ].detach()
# z = torch.from_numpy(ground_truth_lr[
#     :, data.variantxgene_ixs
# ])
cors.extend(paircor(y, z).numpy())

variantxgenes_info["cor"] = cors
# variantxgenes_info["correlated"] = variantxgenes_info["cor"] > 0.

np.mean(cors)

# %%
# x = torch.linspace(-loader.window_size-5, +loader.window_size+5, 1000)
# embedding = model_pred.cut_embedder(x)
# sns.heatmap(embedding)

# %% [markdown]
# ### Train

# %%
loaders.initialize(minibatches_train)
loaders_validation.initialize(minibatches_validation)


# %%
class GeneLikelihoodHook:
    def __init__(self, n_variantxgene_ixs):
        self.n_variantxgene_ixs = n_variantxgene_ixs
        self.likelihood = []

    def start(self):
        self.likelihood_checkpoint = np.zeros(self.n_variantxgene_ixs)
        return {}

    def run_individual(self, model, data):
        self.likelihood_checkpoint[data.variantxgene_ixs.cpu().numpy()] += (
            model.get_full_elbo().cpu().detach().numpy().sum(0).sum(0)
        )

    def finish(self):
        self.likelihood.append(self.likelihood_checkpoint)


# %%
# hooks_checkpoint = [GeneLikelihoodHook(loader.n_variantxgenes)]
hooks_checkpoint = []

# %%
optim = chd.optim.SparseDenseAdam(
    model_pred.parameters_sparse(), model_pred.parameters_dense(), lr=5e-3
)
trainer = prediction_model.Trainer(
    model_pred,
    loaders,
    loaders_validation,
    optim,
    checkpoint_every_epoch=5,
    n_epochs=30,
    hooks_checkpoint=hooks_checkpoint,
    device="cuda",
)
trainer.train()

# %%
output_bias = model_pred.fc_log_predictor.nn[0].bias[0].item()

fig, ax = plt.subplots()
x = torch.linspace(-5.0, 5.0, 100)
for weight, input_bias in zip(
    model_pred.fc_log_predictor.nn[0].weight[0],
    model_pred.fc_log_predictor.embedding_bias,
):
    weight = weight.item()
    input_bias = input_bias.item()
    y = torch.sigmoid((x + input_bias) * weight + output_bias)
    ax.plot(x, y)

# %% [markdown]
# ### Inference & interpretation

# %%
minibatches_inference = minibatches_train + minibatches_validation
minibatches_inference = minibatches_full

# %%
loaders_inference = chd.loaders.LoaderPool(
    prediction_model.Loader,
    loader=loader,
    n_workers=5,
    shuffle_on_iter=False,
)

# %%
device = "cuda"
# device = "cpu"

# %%
loaders_inference.initialize(minibatches_inference)
prioritization = np.zeros(
    (len(transcriptome.clusters_info), len(variantxgenes_info.index))
)
effect = np.zeros((len(transcriptome.clusters_info), len(variantxgenes_info.index)))

model_pred = model_pred.to(device)
for data in tqdm.tqdm(loaders_inference):
    data = data.to(device)
    model_pred.forward(data)
    prioritization_mb = (
        model_pred.fc_log_predictor.prioritization.squeeze(-1).detach().cpu().numpy()
    )
    prioritization[:, data.variantxgene_ixs.cpu().numpy()] += prioritization_mb
    effect_mb = model_pred.fc_log_predictor.effect.squeeze(-1).detach().cpu().numpy()
    effect[:, data.variantxgene_ixs.cpu().numpy()] += effect_mb

    loaders_inference.submit_next()

prioritization = xr.DataArray(
    prioritization, coords=[transcriptome.clusters_info.index, variantxgenes_info.index]
)
effect = xr.DataArray(
    effect, coords=[transcriptome.clusters_info.index, variantxgenes_info.index]
)

# %%
np.mean((ground_truth_prioritization.cpu().numpy() - prioritization.values) ** 2)

# %%
np.mean((ground_truth_prioritization.cpu().numpy() - prioritization.values) ** 2)

# %%
variantxgenes_info["cor"] = paircor(
    ground_truth_prioritization.cpu().numpy(), prioritization.values
)
variantxgenes_info["cor"].mean()

# %%
variantxgenes_info["ix"] = np.arange(len(variantxgenes_info))
variantxgenes_info["phase"] = "train"
variantxgenes_info.loc[
    variantxgenes_info.index.get_level_values("gene").isin(validation_genes.index),
    "phase",
] = "validation"
variantxgenes_info.groupby("phase")["cor"].mean()

# %%
scores = prioritization.to_pandas().T.stack().to_frame("prioritization")
scores["effect"] = effect.to_pandas().T.stack()

# %%
# variant_id = genotype.variants_info.query("rsid == 'rs3087243'").index[0]
# gene_id = transcriptome.gene_id("CTLA4")

# %%
scores_eqtl["significant"] = scores_eqtl["bf"] > np.log(10)
scores_joined = scores.join(scores_eqtl)
scores_joined = scores_joined.query("effect != 0.")

# %%
# scores_joined.join(genotype.variants_info[["rsid"]]).xs(variant_id, level = "variant").xs(gene_id, level = "gene").style.bar()

# %%
# scores.join(genotype.variants_info[["rsid"]]).xs("CD4 T", level = "cluster").xs(gene_id, level = "gene").style.bar()
# scores.join(genotype.variants_info[["rsid"]]).xs(variant_id, level = "variant").xs(gene_id, level = "gene").style.bar()

# %%
gene_id = transcriptome.gene_id("CTLA4")
variant_id = (
    scores_joined.loc[gene_id]
    .sort_values("bf", ascending=False)
    .index.get_level_values("variant")[8]
)

# %%
fig, ax = plt.subplots()
ax.scatter(
    scores_joined.loc[gene_id].loc[variant_id]["fc_log"],
    scores_joined.loc[gene_id].loc[variant_id]["effect"],
)

# %%
# cors = []
# for gene_id in tqdm.tqdm(train_genes.index):
#     cors.append(np.corrcoef(
#         np.abs(scores_eqtl.xs("CD4 T", level = "cluster").loc[gene_id]["bf"].values),
#         scores.xs("CD4 T", level = "cluster").loc[gene_id]["prioritization"].values
#     )[0, 1])
# np.mean(np.array(cors)[~np.isnan(cors)])

# %%
gene_id = transcriptome.gene_id("THEMIS2")

# %%
scores_eqtl

# %%
np.corrcoef(scores_joined["effect"].values, np.abs(scores_joined["fc_log"].values))

# %%
plt.scatter(
    scores_joined.loc[gene_id]["prioritization"].values,
    scores_joined.loc[gene_id]["fc_log"].values,
)


# %%
def paircor(x, y, dim=0, eps=1e-10):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(
        dim
    ) / divisor
    return cor


# %%
import scipy.stats

# %%
# variants_oi = genotype.variants_info.query("main").index
variants_oi = genotype.variants_info.query("~imputed").index

# %%
indices_oi = (
    scores_joined.reset_index("gene")
    .droplevel("cluster")
    .query("gene in @validation_genes.index")
    # .query("significant")
    # .query("variant in @variants_oi")
    .groupby(["variant", "gene"])
    .first()
    .index
)

# indices_oi = (
#     scores_joined.reset_index("gene").droplevel("cluster")
#     .query("significant")
#     # .query("variant in @variants_oi")
#     # .query("gene in @validation_genes.index")
#     ["gene"]
#     # .groupby("gene")["bf"].idxmax()
# )
# indices_oi = pd.MultiIndex.from_frame(pd.DataFrame({"gene":indices_oi.index, "variant":indices_oi.values}))
# indices_oi = pd.MultiIndex.from_frame(pd.DataFrame({"gene":indices_oi.values, "variant":indices_oi.index}))
print(len(indices_oi))

# %%
scores_oi = indices_oi.to_frame()[[]].join(scores_joined)

# %%
import sklearn.metrics

# %%
sklearn.metrics.average_precision_score(
    scores_oi["significant"], scores_oi["prioritization"]
) / sklearn.metrics.average_precision_score(
    scores_oi["significant"],
    np.random.choice(scores_oi["prioritization"], len(scores_oi)),
)

# %%
# scores_oi = scores_oi.query("cluster == 'B'")
sklearn.metrics.roc_auc_score(scores_oi["significant"], scores_oi["prioritization"])

# %%
curve = sklearn.metrics.roc_curve(scores_oi["significant"], scores_oi["prioritization"])
fig, ax = plt.subplots(figsize=(3, 3))
ax.set_aspect(1)
ax.plot(curve[0], curve[1])
ax.axline((0, 0), slope=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# %%
sklearn.metrics.log_loss(
    scores_oi["significant"], scores_oi["prioritization"]
) / sklearn.metrics.log_loss(
    scores_oi["significant"],
    np.random.choice(scores_oi["prioritization"], len(scores_oi)),
)

# %%
x = scipy.stats.rankdata(scores_oi["fc_log"].unstack().values.T, axis=0)
y = scipy.stats.rankdata(scores_oi["effect"].unstack().values.T, axis=0)

# %%
(np.argmax(x, 0) == np.argmax(y, 0)).mean() / (1 / x.shape[0])

# %%
cors = paircor(x, y)
np.mean(cors)

# %%
x = scipy.stats.rankdata(scores_oi["fc_log"].unstack().values.T, axis=0)
y = scipy.stats.rankdata(scores_oi["effect"].unstack().values.T, axis=0)

# %%
np.corrcoef(
    scores_joined.xs("NK", level="cluster")["effect"],
    scores_joined.xs("NK", level="cluster")["fc_log"],
)

# %%
import sklearn.metrics

# %%
sklearn.metrics.average_precision_score(
    scores_joined.query("cluster == 'CD4 T'")["significant"],
    scores_joined.query("cluster == 'CD4 T'")["prioritization"],
)

# %%
ground_truth_significant

# %%
prioritization_oi = prioritization.to_pandas().T.loc[variantxgenes_info.index]

# %%
paircor(prioritization_oi.values.T, ground_truth_significant).mean()

# %%
variantxgenes_info["cor"].mean()

# %%
fig, ax = plt.subplots()
ax.scatter(
    variantxgenes_info["cor"],
    paircor(prioritization_oi.values.T, ground_truth_significant),
)

# %%
