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

# %%
import latenta as la
import crispyKC as ckc
import eyck
import tempfile

# %% [markdown]
# ## Get the dataset

# %%
# dataset_name = "pbmc10k/subsets/top250"
# dataset_name = "pbmc10kx"
# dataset_name = "pbmc10k"
dataset_name = "liverkia_lsecs"
# dataset_name = "e18brain"
# dataset_name = "lymphoma"
# dataset_name = "hspc"
# dataset_name = "hspc_cycling"
# dataset_name = "hspc_meg_cycling"
# dataset_name = "hspc_gmp_cycling"
# dataset_name = "liver"
regions_name = "2-8"
# regions_name = "10k10k"
latent = "leiden_0.1"
# latent = "phase"
transcriptome = eyck.modalities.Transcriptome(
    ckc.get_output() / "datasets" / dataset_name / "transcriptome"
)
fragments = eyck.flow.Flow.from_path(
    ckc.get_output() / "datasets" / dataset_name / "fragments" / regions_name
)
clustering = eyck.modalities.clustering.Clustering(
    ckc.get_output() / "datasets" / dataset_name / "latent" / latent
)
clustering.labels = transcriptome.obs["celltype"].astype("category")

# %%
fragments

# %%
import eyck.modalities.fragments.loaders

# %% [markdown]
# ## Single loader

# %%
fragments_loader = eyck.modalities.fragments.loaders.Fragments(
    fragments,
    cell_batch_size=500,
    requests=tuple(
        [
            "coordinates",
            "local_cellxregion_ix",
            "mask",
            "cut_coordinates",
            "cut_mask",
            "cut_region_cluster",
        ]
    ),
)
# %%
fragments_loader.load(
    {"cell": np.arange(500), "region": np.arange(min(500, len(fragments.var)))}
)["cut_region_cluster"].shape

# %% [markdown]
# ## Pool
# %%
pool = la.loaders.pool.Pool(loader=fragments_loader, n_workers=10)

# %%
import jax

minibatcher = la.train.minibatching.Minibatcher(
    la.Dim(transcriptome.obs.index), key=jax.random.PRNGKey(0), size=500
)

# %%
pool.initialize(minibatcher)
pool.start()

# %%
for i in range(100):
    d = pool.pull()
    pool.submit_next()


# %% [markdown]
# ## In a useless latenta model

# %%
cells = la.Dim(transcriptome.obs.index)
gene_ids_loader = la.loaders.Take(
    fragments, key="regionmapping", definition=la.Definition([cells])
)
gene_ids = la.Fixed(gene_ids_loader)

# %%
program = la.Program()
gene_ids.loader.parent.load(program, minibatcher=minibatcher, n_workers=30)
gene_ids.loader.load(program, minibatcher=minibatcher)
gene_ids.value(program)
# %%
program.run()

# %%
runner = program.create_runner()
state = program.create_state()
cpu_reloader = program.cpu_reloader()

# %%
import time

start = time.time()
for i, minibatch in zip(range(500), minibatcher):
    state = cpu_reloader(state)
    state = runner(state)
end = time.time()
# %%
pool = program.aux[fragments.uuid][1]
print(pool.n_workers, np.sum(program.aux[fragments.uuid][1].wait) / (end - start))
pool.join()
# %%
plt.plot(program.aux[fragments.uuid][1].wait)

# %% [markdown]
# ## In a real latenta model

# %%
assert (fragments.obs.index == transcriptome.obs.index).all()

# %%
cuts = la.Dim(10, name="cut")
cells = la.Dim(transcriptome.obs.index)
regions = la.Dim(fragments.var.index)

# %%
cells = la.Dim(transcriptome.obs.index)
gene_ids_loader = la.loaders.Take(
    fragments, key="regionmapping", definition=la.Definition([cuts])
)
gene_ids = la.Fixed(gene_ids_loader)

# %%
lsec = la.Fixed((transcriptome.obs["celltype"] != "LSEC").astype(int), label="lsec")

# %%
local_indices = la.Fixed(
    la.loaders.Take(
        fragments,
        key="local_indices",
        definition=la.Definition([cuts, la.Dim(["cell", "gene"], "dimension")]),
    )
)
mask = la.Fixed(
    la.loaders.Take(
        fragments,
        key="mask",
        definition=la.Definition([cuts]),
    )
)

# %%
import latenta.operations.bincount

counts = la.operations.bincount.Bincount(
    x=local_indices,
    mask=mask,
    definition=la.Definition([cells, regions]),
)

# %%
prior_counts = fragments.counts
lib_value = prior_counts.sum(1)
prior_normalized = prior_counts / lib_value[:, None]
lib = la.Fixed(np.log(lib_value), definition=la.Definition([cells]))
initial_b_value = np.log(prior_normalized.mean(0))
b = la.Parameter(initial_b_value, definition=la.Definition([regions]))

# %%
a_scale = la.Latent(la.distributions.LogNormal(), label="scale")
a = la.Latent(
    la.distributions.Normal(0.0, a_scale),
    # la.distributions.Normal(0.0, 2.0),
    label="a",
    definition=la.Definition([regions]),
)

# %%
y = la.links.scalar.Linear(
    la.links.scalar.Linear(lsec, a=a, b=b, definition=la.Definition([cells, regions])),
    b=lib,
    transforms=[la.transforms.Exp()],
)

# %%
p = la.distributions.Poisson(
    y,
    definition=la.Definition([cells, regions]),
)
p = la.distributions.NegativeBinomial2(
    y,
    la.Parameter(
        1.0,
        transforms=[la.transforms.Softplus()],
        label="dispersion",
        definition=la.Definition([regions]),
    ),
    definition=la.Definition([cells, regions]),
)

# %%
from latenta.variables.observation import ComputedObservation

observation = ComputedObservation(
    counts,
    p,
    definition=la.Definition([cells, regions]),
)

# %%
minibatcher = la.train.minibatching.Minibatcher(
    la.Dim(transcriptome.obs.index),
    key=jax.random.PRNGKey(0),
    # size = 100,
    size=len(transcriptome.obs.index),
    permute=False,
)

# %%
observation.rootify()
program = la.Program(minibatcher=minibatcher)
observation.elbo(program)
mask.value(program)
lsec.value(program)
lib.value(program)
counts.value(program)

# %%
out = program.run()

# %%
sns.boxplot(
    x=transcriptome.obs["celltype"],
    y=out[(counts.uuid, "value")][:, transcriptome.gene_ix("Vwf")]
    / np.exp(out[(lib.uuid, "value")]),
    # y = fragments.counts[:, transcriptome.gene_ix("Vwf")],
    # y = prior_normalized[:, transcriptome.gene_ix("Vwf")],
)

# %%
plt.scatter(y.prior().mean(0), prior_counts.mean(0))

# %%
import optax

svi = la.train.svi.SVI(observation, optax.adam(1e-2), minibatcher=minibatcher)

# %%
losses = svi.train(500)

# %%
plt.plot(losses)

# %%
scores = pd.DataFrame(
    {
        "a": a.q.loc.loader.value,
        "b": b.prior(),
        "b_initial": initial_b_value,
        "symbol": transcriptome.var.symbol,
        "dispersion": p.dispersion.prior(),
    },
    index=fragments.var.index,
).sort_values("a")
scores

# %%
transcriptome_cluster_conts = (
    pd.DataFrame(
        transcriptome.layers["normalized"][:],
        index=clustering.labels,
        columns=transcriptome.var.index,
    )
    .groupby(level=0)
    .mean()
)
# %%
fragment_cluster_counts = (
    pd.DataFrame(
        fragments.counts / fragments.counts.sum(0, keepdims=True) * 1000,
        index=clustering.labels,
        columns=fragments.var.index,
    )
    .groupby(level=0)
    .mean()
)
scores["a_empirical"] = np.log(
    fragment_cluster_counts.loc["EC"] / fragment_cluster_counts.loc["LSEC"]
)
plt.scatter(scores["a"], scores["a_empirical"], c=scores["b"])
plt.colorbar()
scores


# %%
gene_id = transcriptome.gene_id("Stab2")

(
    transcriptome_cluster_conts[gene_id],
    fragment_cluster_counts[gene_id],
    scores.loc[gene_id],
)

# %%
scores.sort_values("a")
# %%
sns.ecdfplot(fragments.counts.sum(0))

# %%
# plt.plot(prior_counts[:, transcriptome.gene_ix("Rspo3")])
plt.plot(prior_normalized[:, transcriptome.gene_ix("Stab2")])

# %%
eyck.modalities.transcriptome.plot_umap(
    transcriptome, ["Stab2", "Dll4", "Wnt2"]
).display()
# %%

# %% [markdown]
# ## Differential

# %%
transcriptome.obs["celltype"] = transcriptome.obs["celltype"].astype("category")

# %%
assert (fragments.obs.index == transcriptome.obs.index).all()

# %%
cuts = la.Dim(10, name="cut")
cells = la.Dim(transcriptome.obs.index)
regions = la.Dim(fragments.var.index)
celltypes = la.Dim(transcriptome.obs["celltype"].cat.categories, name="celltype")

# %%
celltype = la.variables.discrete.DiscreteFixed(
    transcriptome.obs["celltype"], definition=la.Definition([cells, celltypes])
)

# %% [markdown]
# ### Counts

# %%
gene_ids_loader = la.loaders.Take(
    fragments, key="regionmapping", definition=la.Definition([cuts])
)
gene_ids = la.Fixed(gene_ids_loader)

# %%
local_indices = la.Fixed(
    la.loaders.Take(
        fragments,
        key="local_indices",
        definition=la.Definition([cuts, la.Dim(["cell", "gene"], "dimension")]),
    )
)
mask = la.Fixed(
    la.loaders.Take(
        fragments,
        key="mask",
        definition=la.Definition([cuts]),
    )
)

# %%
import latenta.operations.bincount

counts = la.operations.bincount.Bincount(
    x=local_indices,
    mask=mask,
    definition=la.Definition([cells, regions]),
)

# %%
prior_counts = fragments.counts
lib_value = prior_counts.sum(1)
prior_normalized = prior_counts / lib_value[:, None]
lib = la.Fixed(np.log(lib_value), definition=la.Definition([cells]))
initial_b_value = np.log(prior_normalized.mean(0))
b = la.Parameter(initial_b_value, definition=la.Definition([regions]))

# %%
a_scale = la.Latent(la.distributions.LogNormal(), label="scale")
a = la.Latent(
    la.distributions.Normal(0.0, a_scale),
    label="a",
    definition=la.Definition([regions, celltypes]),
)

# %%
y = la.links.scalar.Linear(
    la.links.vector.Matmul(
        celltype, a=a, b=b, definition=la.Definition([cells, regions])
    ),
    b=lib,
    transforms=[la.transforms.Exp()],
)

# %%
a.prior().shape

# %%
p = la.distributions.Poisson(
    y,
    definition=la.Definition([cells, regions]),
)
p = la.distributions.NegativeBinomial2(
    y,
    la.Parameter(
        1.0,
        transforms=[la.transforms.Softplus()],
        label="dispersion",
        definition=la.Definition([regions]),
    ),
    definition=la.Definition([cells, regions]),
)

# %%
from latenta.variables.observation import ComputedObservation

observation = ComputedObservation(
    counts,
    p,
    definition=la.Definition([cells, regions]),
)

# %% [markdown]
# ### Landscape

# %%
import latenta.distributions.binary_uniform as bu
import latenta.links.vector.binary_spline as bs
import math

# %%
tot = fragments.regions.window[1] - fragments.regions.window[0]
width = tot
expdim_min = int(math.log2(tot) - math.log2(256))
expdim = int(math.log2(tot) - math.log2(8))

# %%
a_zooms = bs.create_zooms(expdim, expdim_min=expdim_min)
a_split = bs.create_a_values(a_zooms)
a_concatenated, splits, shapes = bs.concatenate_a(a_split)
a_original = a_concatenated
a_original.shape

# %%
knot = la.Dim(len(a_original), name="knot")

# %%
a_baseline_value = np.repeat(a_original[..., None], len(regions), axis=-1)
a_baseline = la.Latent(
    la.distributions.Normal(),
    label="baseline",
    definition=la.Definition([knot, regions]),
)

# %%
a_diff_value = np.repeat(
    np.repeat(a_original[..., None], len(regions), axis=-1)[..., None],
    len(celltypes),
    axis=-1,
)
# a_scale = la.Latent(
#     la.distributions.LogNormal(),
#     label = "scale"
# )
a_scale = la.Fixed(1.0)
a_diff = la.Latent(
    la.distributions.Normal(0.0, a_scale),
    label="a_diff",
    definition=la.Definition([knot, regions, celltypes]),
)

# %%
a = la.links.scalar.Linear(
    a_baseline,
    b=a_diff,
)

# %%
x = la.Fixed(
    la.loaders.Take(
        fragments,
        key="cut_coordinates",
        definition=la.Definition([cuts]),
    )
)
fragments.obs["celltype"] = transcriptome.obs["celltype"]
f = la.Fixed(
    la.loaders.Take(
        fragments,
        key="cut_region_cluster",
        definition=la.Definition([cuts, la.Dim(["region", "cluster"], "feature")]),
    ),
    label="f",
)
mask = la.Fixed(
    la.loaders.Take(
        fragments,
        key="cut_mask",
        definition=la.Definition([cuts]),
    )
)

# %%
import jax

minibatcher = la.train.minibatching.Minibatcher(
    la.Dim(transcriptome.obs.index), key=jax.random.PRNGKey(0), size=100
)
program = la.Program(minibatcher=minibatcher)
x.value(program)
f.value(program)
mask.value(program)
program.run()[(x.uuid, "value")].max()

# %%
dist = bu.BinaryUniform(a=a, f=f, mask=mask, zooms=a_zooms, width=width)

# %%
from latenta.variables.observation import ComputedObservation

observation_cut_sites = ComputedObservation(
    x,
    dist,
    definition=la.Definition([cuts]),
)

# %%
root = la.variables.Root(
    observation_cut_sites=observation_cut_sites,
    observation=observation,
)

# %% [markdown]
# ### Train

# %%
minibatcher = la.train.minibatching.Minibatcher(
    la.Dim(transcriptome.obs.index),
    key=jax.random.PRNGKey(0),
    # size = 100,
    size=len(transcriptome.obs.index),
    permute=False,
)

# %%
root.rootify()
program = la.Program(minibatcher=minibatcher)

# %%
observation_cut_sites.elbo(program)

# %%
out = program.run()
# %%
plt.scatter(y.prior().mean(0), prior_counts.mean(0))

# %%
import optax

svi = la.train.svi.SVI(root, optax.adam(5e-3), minibatcher=minibatcher)

# %%
svi.run()

# %%
losses = svi.train(500)

# %%
plt.plot(losses)

# %% [markdown]
# ### Inference

# %%
a_overall = root.observation.find("a")

# %%
scores = pd.DataFrame(
    {
        "a": a_overall.q.loc.loader.value[:, 0] - a_overall.q.loc.loader.value[:, 1],
        "b": b.prior(),
        "b_initial": initial_b_value,
        "symbol": transcriptome.var.symbol,
        "dispersion": p.dispersion.prior(),
    },
    index=fragments.var.index,
).sort_values("a")
scores

# %%
transcriptome_cluster_conts = (
    pd.DataFrame(
        transcriptome.layers["normalized"][:],
        index=clustering.labels,
        columns=transcriptome.var.index,
    )
    .groupby(level=0)
    .mean()
)
# %%
fragment_cluster_counts = (
    pd.DataFrame(
        fragments.counts / fragments.counts.sum(0, keepdims=True) * 1000,
        index=clustering.labels,
        columns=fragments.var.index,
    )
    .groupby(level=0)
    .mean()
)
scores["a_empirical"] = np.log(
    fragment_cluster_counts.loc["EC"] / fragment_cluster_counts.loc["LSEC"]
)
plt.scatter(scores["a"], scores["a_empirical"], c=scores["b"])
plt.colorbar()
scores


# %%
gene_id = transcriptome.gene_id("Stab2")

(
    transcriptome_cluster_conts[gene_id],
    fragment_cluster_counts[gene_id],
    scores.loc[gene_id],
)

# %%
scores.sort_values("a")
# %%
sns.ecdfplot(fragments.counts.sum(0))

# %%
# plt.plot(prior_counts[:, transcriptome.gene_ix("Rspo3")])
plt.plot(prior_normalized[:, transcriptome.gene_ix("Stab2")])

# %%
eyck.modalities.transcriptome.plot_umap(
    transcriptome, ["Stab2", "Dll4", "Wnt2"]
).display()

# %% [markdown]
# ### Landscape inference

# %%
design = eyck.utils.crossing(
    pd.Series(np.arange(0, width), name="coordinate"),
    pd.Series(range(len(celltypes)), name="celltype"),
    pd.Series([transcriptome.gene_ix("Stab2")], name="region"),
)
design["mask"] = 0
design.index.name = "cell"
design.columns.name = "feature"
# x = la.Fixed(design["coordinate"].astype(int))
# f = la.Fixed(design[["cluster", "region"]])

# %%
program = la.Program()
root.observation_cut_sites.find("x").condition(
    program, value=design["coordinate"].values
)
f.condition(program, value=design[["celltype", "region"]].values)
mask.condition(program, value=design["mask"].values)
root.observation_cut_sites.likelihood(program)
a = root.observation_cut_sites.p.a
a.value(program)

# %%
out = program.run()
# %%
design["likelihood"] = np.exp(out[("root.observation_cut_sites", "likelihood")])

# %%
design.set_index(["celltype", "coordinate"]).unstack("celltype")["likelihood"].plot()
# %%
design
# %%
out[(a.uuid, "value")][:, design["region"].iloc[0]]

# %%[markdown]
# ## Test

# %%
sc.tl.rank_genes_groups(
    transcriptome.adata,
    groupby="celltype2",
    method="wilcoxon",
    use_raw=False,
    key_added="rank_genes_groups",
)

# %%
sc.get.rank_genes_groups_df(transcriptome.adata, group="LSEC_central").assign(symbol = lambda x: transcriptome.var.loc[x.names]["symbol"].values).query("logfoldchanges > 0.5").head(20)

# %%
lib_value = fragments.counts.sum(1)

# %%
symbol = "Dll4"
gene_ix = transcriptome.gene_ix(symbol)
print(sc.get.rank_genes_groups_df(transcriptome.adata, group=["LSEC_portal", "LSEC_central"]).assign(symbol = lambda x: transcriptome.var.loc[x.names]["symbol"].values).set_index("symbol").loc[symbol])
fragments_oi = fragments.mapping[:, 1] == gene_ix
cell_ixs = np.repeat(fragments.mapping[fragments_oi, 0], 2, axis=0).flatten()
coordinates = fragments.coordinates[fragments_oi].flatten()

# %%
data = pd.DataFrame(
    {
        "cell": cell_ixs,
        "coordinate": coordinates,
        "lib": lib_value[cell_ixs] * 1000,
        "cluster": pd.Categorical(transcriptome.adata.obs["celltype2"].iloc[cell_ixs], categories=["EC_portal", "LSEC_portal", "LSEC_central", "EC_central"]),
    }
)
# data["weight"] = 1 / data["lib"]
data["weight"] = 1

# %%
bins = np.arange(
    fragments.regions.window[0], fragments.regions.window[1] + 501, step=500
)
data["bin_ix"] = np.digitize(data["coordinate"], bins)
data["bin_ix"] = pd.Categorical(bins[data["bin_ix"]], categories=bins)

# %%
fig, ax = plt.subplots(figsize=(50, 2))
plotdata = (
    data.groupby(["cluster", "bin_ix"])["weight"].sum().unstack("cluster").fillna(0.0)
)

for cluster in plotdata.columns:
    ax.plot(plotdata.index, plotdata[cluster], label=cluster)
plt.legend()
fig.savefig("/home/wouters/test.png")

# %%
data

# %%
eyck.modalities.transcriptome.plot_umap(
    transcriptome, ["celltype2", "Dll4", "Wnt2", "Lyve1", "Vwf", "Rspo3", "Kit", "Stab2", "Cdh13"]
).display()

# %%
clustering.labels.iloc[cell_ixs]

# %%
fig, ax = plt.subplots()
ax.hist(coordinates, bins = 300, lw=0)


# %%
