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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

#export LD_LIBRARY_PATH=/data/peak_free_atac/software/peak_free_atac/lib
import torch
import torch_sparse

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa

# %% [markdown]
# ### Simulation

# %%
from peakfreeatac.simulation import Simulation
import pathlib
import tempfile

# %%
simulation = Simulation(n_genes = 100, n_cells = 500)

# %% [markdown]
# Create fragments

# %%
fragments = pfa.data.Fragments(path = pathlib.Path(tempfile.TemporaryDirectory().name))

# %%
# need to make sure the order of fragments is cellxgene
order = np.argsort((simulation.mapping[:, 0] * simulation.n_genes) + simulation.mapping[:, 1])

fragments.coordinates = torch.from_numpy(simulation.coordinates)[order]
fragments.mapping = torch.from_numpy(simulation.mapping)[order]
fragments.var = pd.DataFrame({"gene":np.arange(simulation.n_genes)}).set_index("gene")
fragments.obs = pd.DataFrame({"cell":np.arange(simulation.n_cells)}).set_index("cell")
fragments.create_cellxgene_indptr()

# %%
fragments.window = simulation.window

# %%
fragments.create_cut_data()

# %% [markdown]
# ## Real

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
fragments.window = window
fragments.create_cut_data()

# %%
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
fold = folds[0]

# %%
from design import get_folds_inference
folds = get_folds_inference(fragments, folds)

# %% [markdown]
# ## Create GS

# %%
adata_transcriptome = transcriptome.adata

# %%
gslatents = {}

sc.pp.neighbors(adata_transcriptome)
sc.tl.leiden(adata_transcriptome, resolution = 0.1)
gslatents["leiden_0.01"] = adata_transcriptome.obs["leiden"].astype(int)

# sc.pp.neighbors(adata_transcriptome)
# sc.tl.leiden(adata_transcriptome, resolution = 0.5)
# gslatents["leiden_0.5"] = adata_transcriptome.obs["leiden"].astype(int)

sc.pp.neighbors(adata_transcriptome)
sc.tl.leiden(adata_transcriptome, resolution = 1.0)
gslatents["leiden_1"] = adata_transcriptome.obs["leiden"].astype(int)


# %%
def score_latent(X, latents, cells_train, cells_validation):
    import xgboost
    import sklearn

    scores = []
    for gslatent_name, gslatent in gslatents.items():
        classifier = xgboost.XGBClassifier()
        # classifier = sklearn.tree.DecisionTreeClassifier()
        # from sklearn.ensemble import RandomForestClassifier; classifier = RandomForestClassifier()
        # from sklearn.neighbors import KNeighborsClassifier; classifier = KNeighborsClassifier()
        # from sklearn.naive_bayes import GaussianNB; classifier = GaussianNB()
        from sklearn.svm import SVC; classifier = SVC()
        classifier.fit(X[cells_train], gslatent[cells_train])

        # validation
        prediction = classifier.predict(X[cells_validation])
        accuracy = sklearn.metrics.accuracy_score(gslatent[cells_validation], prediction)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(gslatent[cells_validation], prediction)
        scores.append({"gslatent":gslatent_name, "phase":"validation", "accuracy":accuracy, "balanced_accuracy":balanced_accuracy})

        # train
        prediction = classifier.predict(X[cells_train])
        accuracy = sklearn.metrics.accuracy_score(gslatent[cells_train], prediction)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(gslatent[cells_train], prediction)
        scores.append({"gslatent":gslatent_name, "phase":"train", "accuracy":accuracy, "balanced_accuracy":balanced_accuracy})
    scores = pd.DataFrame(scores)
    return scores


# %% [markdown]
# ## Create loaders

# %%
cells_train = np.arange(0, int(fragments.n_cells * 4 / 5))
cells_validation = np.arange(int(fragments.n_cells * 4 / 5), fragments.n_cells)


# %%
class Minibatcher():
    def __init__(self, cells, genes, n_cells_step, n_genes_step):
        self.cells = cells
        self.genes = genes
        self.n_cells_step = n_cells_step
        self.n_genes_step = n_genes_step
        
    def create_minibatches(self, *args, **kwargs):
        minibatches = create_bins_random(self.cells, self.genes, fragments.n_genes, n_genes_step = n_genes_step, n_cells_step=n_cells_step, *args, **kwargs)
        return minibatches


# %%
import peakfreeatac.loaders.fragments
n_cells_step = 200
n_genes_step = 5000

loaders_train = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step},
    n_workers = 10
)
# minibatches_train = pfa.loaders.minibatching.create_bins_random(cells_train, np.arange(fragments.n_genes), fragments.n_genes, n_genes_step = n_genes_step, n_cells_step=n_cells_step, use_all = True)
# assert len(minibatches_train) > 0
# print(len(minibatches_train))
minibatcher = pfa.loaders.minibatching.Minibatcher(cells_train, np.arange(fragments.n_genes), fragments.n_genes , n_genes_step = n_genes_step, n_cells_step=n_cells_step)
minibatches_train_sets = [{"tasks":minibatcher.create_minibatches(use_all = True, rg = np.random.RandomState(i))} for i in range(10)]
loaders_train.initialize(next_task_sets=minibatches_train_sets)

loaders_validation = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step},
    n_workers = 2
)
minibatches_validation = pfa.loaders.minibatching.create_bins_random(cells_validation, np.arange(fragments.n_genes), fragments.n_genes, n_genes_step = n_genes_step, n_cells_step=n_cells_step, use_all = True, permute_genes=False)
assert len(minibatches_validation) > 0
loaders_validation.initialize(minibatches_validation)

# %%
n_cells_step = 200
n_genes_step = 5000
loaders_all = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells_step * n_genes_step},
    n_workers = 2
)
minibatches_all = pfa.loaders.minibatching.create_bins_ordered(
    np.arange(fragments.n_cells),
    np.arange(fragments.n_genes),
    fragments.n_genes,
    n_genes_step = n_genes_step,
    n_cells_step=n_cells_step,
    use_all = True,
    permute_genes=False,
    permute_cells=False
)
loaders_all.initialize(minibatches_all)

# %% [markdown]
# ## Infer

# %%
device = "cuda"

# %% [markdown]
# Train

# %%
# import peakfreeatac.models.vae.v3 as vae_model
# model = vae_model.VAE(fragments, n_bins = 20)

# import peakfreeatac.models.vae.v2 as vae_model
# model = vae_model.VAE(fragments, n_components=128)

# # import peakfreeatac.models.vae.v4 as vae_model
# import peakfreeatac.models.vae.v5 as vae_model
# model = vae_model.VAE(
#     fragments,
#     n_encoder_bins=z,
#     nbins=(z, ),
#     # baseline = True,
#     # n_frequencies = 1,
#     # baseline = True
# )

model = pickle.load((pfa.get_output() / "prediction_vae" / dataset_name / promoter_name / "v5_s0.8" / "model_0.pkl").open("rb"))
# model = pickle.load((pfa.get_output() / "prediction_vae" / dataset_name / promoter_name / "v5_baseline" / "model_0.pkl").open("rb"))

# %%
model

# %%
optimizer = pfa.optim.SparseDenseAdam(model.parameters_sparse(), model.parameters_dense(), lr = 1e-3, weight_decay = 1e-5)

# %%
import torch_scatter
class GeneLikelihoodHook():
    def __init__(self, n_genes):
        self.n_genes = n_genes
        self.likelihood_mixture = []
        self.likelihood_counts = []
        
    def start(self):
        self.likelihood_mixture_checkpoint = np.zeros(self.n_genes)
        self.likelihood_counts_checkpoint = np.zeros(self.n_genes)
        return {}
        
    def run_individual(self, model, data):
        self.likelihood_mixture_checkpoint[data.genes_oi] += torch_scatter.scatter_sum(model.track["likelihood_mixture"], data.cut_local_gene_ix, dim_size = data.n_genes).detach().cpu().numpy()
        self.likelihood_counts_checkpoint[data.genes_oi] += model.track["likelihood_fragmentcounts"].sum(0).detach().cpu().numpy()
        
    def finish(self):
        self.likelihood_mixture.append(self.likelihood_mixture_checkpoint)
        self.likelihood_counts.append(self.likelihood_counts_checkpoint)
        
class EmbeddingHook():
    def __init__(self, n_cells, n_latent_dimensions, loaders_all):
        self.n_cells = n_cells
        self.n_latent_dimensions = n_latent_dimensions
        self.embeddings_checkpoint = []
        
    def run(self, model):
        embedding = np.zeros((self.n_cells, self.n_latent_dimensions))

        loaders_all.restart()
        for data in loaders_all:
            data = data.to(device)
            with torch.no_grad():
                embedding[data.cells_oi] = model.evaluate_latent(data).detach().cpu().numpy()

            loaders_all.submit_next()
        self.embeddings_checkpoint.append(embedding)


# %%
hook_genelikelihood = GeneLikelihoodHook(fragments.n_genes)
hook_embedding = EmbeddingHook(fragments.n_cells, model.n_latent_dimensions, loaders_all)
hooks_checkpoint = [
    hook_genelikelihood
]
hooks_checkpoint2 = [
    hook_embedding
]

# %%
model = model.to(device)

# %%
loaders_train.restart()
loaders_validation.restart()

trainer = pfa.train.Trainer(model, loaders_train, loaders_validation, optimizer, n_epochs = 200, checkpoint_every_epoch=3, optimize_every_step = 1, hooks_checkpoint = hooks_checkpoint, hooks_checkpoint2 = hooks_checkpoint2)
trainer.train()

# %%
likelihood_mixture = pd.DataFrame(np.vstack(hook_genelikelihood.likelihood_mixture), columns = fragments.var.index).T
likelihood_counts = pd.DataFrame(np.vstack(hook_genelikelihood.likelihood_counts), columns = fragments.var.index).T

# %%
# scores = (likelihood_mixture.iloc[:, -1] - likelihood_mixture[2]).sort_values().to_frame("lr")
# scores["label"] = transcriptome.symbol(scores.index)

# %%
fig, axes = plt.subplots(1, 3, figsize = (12, 4), gridspec_kw={"wspace":0.5})
for ax, plotdata_mixture, plotdata_counts in [[axes[0], likelihood_mixture.mean(), likelihood_counts.mean()], [axes[1], likelihood_mixture.mean()[4:], likelihood_counts.mean()[4:]], [axes[2], likelihood_mixture.mean()[-50:], likelihood_counts.mean()[-50:]]]:
    plotdata_mixture.plot(color = "green", label = "mixture")
    plt.legend()
    ax.twinx()
    plotdata_counts.plot(color = "red", label = "counts")

# %% [markdown]
# ## Extract embedding

# %%
import itertools

# %%
model = model.to(device).eval()

# %%
embedding = np.zeros((fragments.n_cells, model.n_latent_dimensions))

loaders_all.restart()
for data in loaders_all:
    data = data.to(device)
    with torch.no_grad():
        embedding[data.cells_oi] = model.evaluate_latent(data).detach().cpu().numpy()
    
    loaders_all.submit_next()

# %%
cell_order = np.arange(fragments.obs.shape[0])
sns.heatmap(embedding[cell_order], vmin = -4, vmax = 4)

# %%
adata = sc.AnnData(obs = fragments.obs)

# %%
adata.obsm["latent"] = embedding

# %%
sc.pp.neighbors(adata, use_rep = "latent")
sc.tl.umap(adata)

# %%
adata.obs["n_fragments"] = np.log1p(torch.bincount(fragments.cellmapping, minlength = fragments.n_cells).cpu().numpy())

# %%
obs_latent = pd.DataFrame(adata.obsm["latent"], columns = "latent_" + pd.Series(np.arange(adata.obsm["latent"].shape[1]).astype(str)))
obs_latent.index = adata.obs.index
adata.obs[obs_latent.columns] = obs_latent

# %%
# gslatent = simulation.cell_latent_space
# obs_gslatent = pd.DataFrame(gslatent, columns = "gslatent_" + pd.Series(np.arange(gslatent.shape[1]).astype(str)))
# obs_gslatent.index = adata.obs.index
# adata.obs[obs_gslatent.columns] = obs_gslatent

# %%
cut_coordinates = fragments.cut_coordinates[fragments.cut_local_gene_ix == 0]

# %%
n = 32

# %%
cuts = cut_coordinates.quantile(torch.linspace(0, 1, n-1))

# %%
# torch.tensor_split(cut_coordinates, fragments.cut_local_gene_ix)

# %%
sc.pl.umap(adata, color = ["n_fragments"])
sc.pl.umap(adata, color = [*obs_latent.columns[:4]], vmin = -4, vmax = 4)
# sc.pl.umap(adata, color = [*obs_gslatent.columns[:10]])

# %%
sc.pl.umap(adata, color = ["n_fragments"])
sc.pl.umap(adata, color = [*obs_latent.columns[:4]], vmin = -4, vmax = 4)
# sc.pl.umap(adata, color = [*obs_gslatent.columns[:10]])

# %%
sc.pl.umap(adata_transcriptome, color = ["leiden"], legend_loc = "on data")
adata.obs["leiden_transcriptome"] = adata_transcriptome.obs["leiden"]

# %%
sc.pl.umap(adata, color = ["leiden_transcriptome"], legend_loc = "on data")

# %%
sc.pl.umap(adata, color = ["leiden_transcriptome"], legend_loc = "on data")

# %%
sc.pl.umap(adata, color = ["leiden_transcriptome"], legend_loc = "on data")

# %%
model

# %%
adata_atac = adata

# %%
# sc.pp.neighbors(adata_transcriptome, n_neighbors=50, key_added = "100")
# sc.pp.neighbors(adata_atac, n_neighbors=50, key_added = "100")

sc.pp.neighbors(adata_transcriptome, n_neighbors=100, key_added = "100")
sc.pp.neighbors(adata_atac, n_neighbors=100, key_added = "100", use_rep = "latent")

# %%
assert (adata_transcriptome.obs.index == adata_atac.obs.index).all()

# %%
A = np.array(adata_transcriptome.obsp["100_connectivities"].todense() != 0)
B = np.array(adata_atac.obsp["100_connectivities"].todense() != 0)

# %%
intersect = A * B
union = (A+B) != 0

# %%
ab = intersect.sum() / union.sum()
ab

# %%
C = A[np.random.choice(A.shape[0], A.shape[0], replace = False)]
# C = B[np.random.choice(B.shape[0], B.shape[0], replace = False)]

# %%
intersect = C * B
union = (C+B) != 0

# %%
ac = intersect.sum() / union.sum()
ac

# %%
ab/ac

# %%
adata_atac.obs["ix"] = np.arange(adata_atac.obs.shape[0])

# %%
cells_train = np.arange(0, int(fragments.n_cells * 5 / 10))
cells_validation = np.arange(int(fragments.n_cells * 5 / 10), fragments.n_cells)
# cells_validation = adata_atac.obs.iloc[cells_validation].loc[adata_atac.obs["n_fragments"] > 8]["ix"].values

# %%
import xgboost
import sklearn
classifier = xgboost.XGBClassifier()
classifier.fit(adata_atac.obsm["X_pca"][cells_train], adata_atac.obs["leiden_transcriptome"].astype(int)[cells_train])
prediction = classifier.predict(adata_atac.obsm["X_pca"][cells_validation])
sklearn.metrics.balanced_accuracy_score(adata_atac.obs["leiden_transcriptome"].astype(int)[cells_validation], prediction)
# sklearn.metrics.accuracy_score(adata_atac.obs["leiden_transcriptome"].astype(int)[cells_validation], prediction)

# %%
adata_atac.obsm["X_pca"].shape

# %% [markdown]
# ## Jaccard

# %%
adata_atac = adata

# %%
from sklearn.neighbors import NearestNeighbors


# %%
def sk_indices_to_sparse(indices):
    k = indices.shape[1]-1
    A = scipy.sparse.coo_matrix((np.repeat(1, k * indices.shape[0]), (indices[:, 1:].flatten(), np.repeat(np.arange(indices.shape[0]), k))), shape = (indices.shape[0], indices.shape[0]))
    return A.tocsr()


# %%
ks = [50]
k = max(ks)

# %%
X = adata_atac.obsm["latent"].copy()
# chosen = np.random.choice(X.shape[0], 5000, replace = False)
# permuted = np.random.choice(chosen, len(chosen), replace = False)
# X[chosen] = X[permuted]

nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs = 10).fit(X)
distances, indices_A = nbrs.kneighbors(X)

# %%
X = adata_transcriptome.obsm["X_pca"]

nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs = 10).fit(X)
distances, indices_B = nbrs.kneighbors(X)


# %%
# import scipy
# def sparse_jaccard(m1,m2):
#     intersection = m1.multiply(m2).sum(axis=1)
#     jaccard = intersection 
#     # a = m1.sum(axis=1)
#     # b = m2.sum(axis=1)
#     # jaccard = intersection/(a+b-intersection)

#     # force jaccard to be 0 even when a+b-intersection is 0
#     jaccard = np.nan_to_num(jaccard)
#     return np.array(jaccard)#.flatten() 

# %%
def score_knn(indices_A, indices_B, ks):
    scores = []
    for k in ks:
        A = sk_indices_to_sparse(indices_A[:, :(k+1)])
        B = sk_indices_to_sparse(indices_B[:, :(k+1)])
        jac = (np.array(A.multiply(B).sum(1)) / k)[:, 0].mean()
        
        normalizer = k / A.shape[1]
        norm_jaccard = jac / normalizer
        scores.append({
            "k":k,
            "jaccard":jac,
            "norm_jaccard":norm_jaccard
        })
    scores = pd.DataFrame(scores)
    return scores


# %%
import scipy.sparse
A = sk_indices_to_sparse(indices_A[:, :(k+1)])
B = sk_indices_to_sparse(indices_B[:, :(k+1)])

# %%
scores = score_knn(indices_A, indices_B, ks) # external baseline
scores

# %% [markdown]
# ## Jaccard over time

# %%
import faiss
import sklearn
def search(X, k):
    import faiss
    X = np.ascontiguousarray(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    D, I = index.search(X, k)
    return I

def convert(I):
    return I + (np.arange(I.shape[0])[:, None]) * I.shape[0]

def calculate_overlap(K, indices_A, indices_B):   
    overlap = np.in1d(convert(indices_A), convert(indices_B)).reshape(indices_A.shape[0], K)[:, 1:]
    return overlap


# %%
def score_overlap(overlap, ks = (50, )):
    scores = []
    for k in ks:
        jac = overlap[:, :k].mean()
        scores.append({
            "k":k,
            "jaccard":jac,
            "norm_jaccard":jac
        })
    return pd.DataFrame(scores)
def score_overlap_phased(overlap, phases,  ks = (50, )):
    scores = []
    for k in ks:
        for phase, cells in phases.items():
            jac = overlap[cells, :k].mean()
            scores.append({
                "k":k,
                "jaccard":jac,
                "norm_jaccard":jac,
                "phase":phase
            })
    return pd.DataFrame(scores)
def score_overlap_phased_weighted(overlap, phases, weights, ks = (50, )):
    scores = []
    for k in ks:
        for phase, cells in phases.items():
            jac = (overlap[cells, :k].mean(1) * weights[cells]).sum() / weights[cells].sum()
            scores.append({
                "k":k,
                "jaccard":jac,
                "norm_jaccard":jac,
                "phase":phase
            })
    return pd.DataFrame(scores)


# %%
import igraph as ig
import leidenalg
def partition(indices):
    edges = np.vstack([
        np.repeat(np.arange(indices.shape[0]), indices.shape[1]),
        indices.flatten()
    ]).T

    graph = ig.Graph(edges = edges)
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter = 1.)
    return np.array(partition.membership)

def score_ari_phased(clusters_B, indices_A, phases, ks = (50, )):
    scores = []
    for k in ks:
        cluster_A = partition(indices_A[:, 1:(k+1)])
        cluster_B = clusters_B[k]
        for phase, cells in phases.items():            
            ari = sklearn.metrics.adjusted_mutual_info_score(cluster_A[cells], cluster_B[cells])
            
            scores.append({
                "k":k,
                "ari":ari,
                "phase":phase
            })
    return pd.DataFrame(scores)


# %%
phases = {"train":cells_train, "validation":cells_validation}

# %%
groups = adata_transcriptome.obs["leiden"]
group_weights = 1/groups.value_counts()
cell_weights = group_weights[adata_transcriptome.obs["leiden"]].values
cell_weights = cell_weights / cell_weights.sum()

# %%
B = adata_transcriptome.obsm["X_pca"]
ks = (1, 2, 5, 10, 20, 50)
K = max(ks)+1
indices_B = search(B, K)

# %%
clusters_B = {}
for k in ks:
    cluster_B = partition(indices_B[:, 1:(k+1)])
    clusters_B[k] = cluster_B

# %%
# embeddings_checkpoint = hook_embedding.embeddings_checkpoint
embeddings_checkpoint = pickle.load((pfa.get_output() / "prediction_vae" / dataset_name / promoter_name / "v6" / "embeddings_checkpoint_0.pkl").open("rb"))

# %%
scores_overlap = []
scores_woverlap = []
scores_ari = []
for checkpoint_ix, embedding in enumerate(tqdm.tqdm(embeddings_checkpoint)):
    A = embedding.astype(np.float32)
    
    indices_A = search(A, K)
    
    overlap = calculate_overlap(K, indices_A, indices_B)
    scores_checkpoint_overlap = score_overlap_phased(overlap, phases, ks = ks)
    scores_checkpoint_overlap["checkpoint"] = checkpoint_ix
    scores_overlap.append(scores_checkpoint_overlap)

    scores_checkpoint_ari = score_ari_phased(clusters_B, indices_A, phases, ks = (50, ))
    scores_checkpoint_ari["checkpoint"] = checkpoint_ix
    scores_ari.append(scores_checkpoint_ari)
    
    scores_checkpoint_woverlap = score_overlap_phased_weighted(overlap, phases, cell_weights, ks = ks)
    scores_checkpoint_woverlap["checkpoint"] = checkpoint_ix
    scores_woverlap.append(scores_checkpoint_woverlap)
scores_overlap = pd.concat(scores_overlap)
scores_woverlap = pd.concat(scores_woverlap)
scores_ari = pd.concat(scores_ari)

# %%
scores_overlap.reset_index().set_index(["phase", "k", "checkpoint"]).unstack().T.loc["norm_jaccard"].plot()

# %%
scores_overlap.reset_index().set_index(["phase", "k", "checkpoint"]).unstack().T.loc["norm_jaccard"].plot()

# %%
scores_woverlap.reset_index().set_index(["phase", "k", "checkpoint"]).unstack().T.loc["norm_jaccard"].plot()

# %%
scores_woverlap.reset_index().set_index(["phase", "k", "checkpoint"]).unstack().T.loc["norm_jaccard"].plot()

# %%
scores_ari.reset_index().set_index(["phase", "k", "checkpoint"]).unstack().T.loc["ari"].plot()

# %%
scores_ari.reset_index().set_index(["phase", "k", "checkpoint"]).unstack().T.loc["ari"].plot()

# %% [markdown]
# ## Interpret output

# %%
import itertools

# %%
model = model.to(device).eval()

# %%
# gene_ix = 0
gene_ix = int(transcriptome.gene_ix("FOSB"));gene_id = transcriptome.var.index[gene_ix]
# gene_id = "ENSG00000005379";gene_ix = int(transcriptome.gene_ix(transcriptome.symbol(gene_id)))

# %%
n = 50
probs = np.zeros((fragments.n_cells, n))
pseudocoordinates = torch.linspace(0, 1, n)

loaders_all.restart()
for data in loaders_all:
    data = data.to(device)
    with torch.no_grad():
        latent = model.evaluate_latent(data).detach().cpu().numpy()
        
        pseudocoordinates_pseudocells = pseudocoordinates.repeat_interleave(latent.shape[0])
        pseudolatent = torch.from_numpy(np.tile(latent.T, latent.shape[0]).T)
        
        prob = model.evaluate_pseudo(pseudocoordinates_pseudocells.to(device), latent = pseudolatent.to(device), gene_oi = gene_ix, cells_oi = data.cells_oi_torch, n = n).detach().cpu()
        prob = prob.reshape(n, latent.shape[0]).T
        probs[data.cells_oi, :] = prob
    
    loaders_all.submit_next()

# %%
# gslatents = {"simul":gslatent}

# %%
gslatent = pd.get_dummies(gslatents[list(gslatents.keys())[0]]).values

# %%
for dim in range(gslatent.shape[1]):
    pd.Series(np.exp(probs[gslatent[:, dim] != 0].mean(0))).plot(label = dim)
plt.legend()

# %%
sns.heatmap(model.decoder.logit_weight(torch.tensor([gene_ix], device = device))[0].cpu().detach().numpy())

# %%
sc.pl.umap(transcriptome.adata, color = "leiden")
sc.pl.umap(transcriptome.adata, color = gene_id)

# %%
gslatent = torch.from_numpy(pd.get_dummies(gslatents[list(gslatents.keys())[0]]).values)

# %%
## Plot prior distribution
fig, axes = plt.subplots(gslatent.shape[1], 1, figsize=(20, 1*gslatent.shape[1]), sharex = True, sharey = True)

fragments_oi_all = (fragments.cut_local_gene_ix == gene_ix)
for i, ax, color in zip(range(latent.shape[1]), axes, sns.color_palette("husl", gslatent.shape[1])):
    fragments_oi = (gslatent[fragments.cut_local_cell_ix, i] != 0) & (fragments.cut_local_gene_ix == gene_ix)
    ax.hist(fragments.cut_coordinates[fragments_oi_all].cpu().numpy(), bins=200, range = (0, 1), lw = 1, density = True, histtype = "step", ec = "#333333FF", zorder = 10)
    ax.hist(fragments.cut_coordinates[fragments_oi].cpu().numpy(), bins=200, range = (0, 1), lw = 0, density = True, zorder = 0, histtype = "bar", color = "#333333AA")

    # Plot initial posterior distribution
    
    cells_oi = (gslatent[:, i] != 0).numpy()
    prob = np.exp(probs[cells_oi].mean(0))
    
    # prob = torch.exp(model.evaluate_pseudo(pseudocoordinates.to(device), latent = pseudolatent.to(device), gene_oi = gene_ix, cells_oi = data.cells_oi_torch, n = n).detach().cpu())
    ax.plot(pseudocoordinates.cpu().numpy(), prob, label = i, color = color, lw = 2, zorder = 20)
    ax.plot(pseudocoordinates.cpu().numpy(), prob, label = i, color = "#FFFFFFFF", lw = 3, zorder = 10)
    ax.set_ylabel(f"{i} n={fragments_oi.sum()}")

# %%
cell_order = np.arange(fragments.obs.shape[0])
sns.heatmap(embedding[cell_order], vmin = -4, vmax = 4)

# %%
k = 50

# %%
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)
index.add(xb)
D, I = index.search(xb, k)


# %%
def search(X, k):
    import faiss                   # make faiss available
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    D, I = index.search(X, k)
    return I
def convert(I):
    return I + (np.arange(I.shape[0])[:, None]) * I.shape[0]


# %%
def scoreit(A, B = None, indices_B = None, ks = (50, )):
    K = max(ks) + 1
    assert indices_B.shape[1] == K
    if indices_B is None:
        indices_B = convert(search(B, K))
        
    indices_A = convert(search(A, K))
    
    overlap = np.in1d(indices_A, indices_B).reshape(indices_A.shape[0], K)[:, 1:]
    
    scores = []
    for k in ks:
        jac = overlap[:, :k].mean()
        scores.append({
            "k":k,
            "jaccard":jac,
            "norm_jaccard":jac
        })
    return pd.DataFrame(scores)


# %%
X = adata_transcriptome.obsm["X_pca"]
indices_B = convert(search(X, k))

# %%
np.in1d(indices_A, indices_B).reshape(

# %%
X = adata_atac.obsm["latent"]
indices_A = convert(search(X, k))
np.in1d(indices_A, indices_B).mean()

# %%
K = 50

# %%
scoreit(A, indices_B = indices_B, ks = ks)

# %%
B = adata_transcriptome.obsm["X_pca"]
ks = (1, 2, 5, 10, 20, 50, )
K = max(ks)+1
indices_B = convert(search(B, K))

# %%
scores = []
for checkpoint_ix, embedding in enumerate(tqdm.tqdm(hook_embedding.embeddings_checkpoint)):
    A = embedding
    
    scores_checkpoint = scoreit(A, indices_B = indices_B, ks = ks)
    scores_checkpoint["checkpoint"] = checkpoint_ix
    scores.append(scores_checkpoint)
scores = pd.concat(scores)

# %%
model

# %%
scores.set_index(["k", "checkpoint"]).unstack().T.loc["norm_jaccard"].plot()

# %%
scores.set_index(["k", "checkpoint"]).unstack().T.loc["norm_jaccard"].plot()

# %%
scores.set_index(["k", "checkpoint"]).unstack().T.loc["norm_jaccard"].plot()

# %%
