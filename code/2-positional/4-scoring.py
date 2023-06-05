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
#     display_name: Python 3
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

import pickle

import scanpy as sc

import torch

import tqdm.auto as tqdm
import xarray as xr

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
# dataset_name = "pbmc10k_gran"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

splitter = "random_5fold"
promoter_name, window = "10k10k", np.array([-10000, 10000])
prediction_name = "v20_initdefault"
outcome_source = "counts"

# splitter = "permutations_5fold5repeat"
# promoter_name, window = "10k10k", np.array([-10000, 10000])
# outcome_source = "magic"
# prediction_name = "v20"
# prediction_name = "v21"

splitter = "permutations_5fold5repeat"
promoter_name, window = "100k100k", np.array([-100000, 100000])
prediction_name = "v20_initdefault"
outcome_source = "magic"

# fragments
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = chd.data.Fragments(folder_data_preproc / "fragments" / promoter_name)
fragments.obs.index.name = "cell"

# create design to run
from design import get_design, get_folds_inference


class Prediction(chd.flow.Flow):
    pass


# folds & minibatching
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step=2000)
folds = folds  # [:1]

# design
from design import get_design, get_folds_training

design = get_design(transcriptome, fragments)

# %%
Scorer = chd.scoring.prediction.Scorer

# %%
design_row = design[prediction_name]

# %%
fragments.window = window

# %%
design_row["loader_parameters"]["cellxgene_batch_size"] = cellxgene_batch_size

# %%
print(prediction_name)
prediction = chd.flow.Flow(
    chd.get_output()
    / "prediction_positional"
    / dataset_name
    / promoter_name
    / splitter
    / prediction_name
)

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

# %%
if outcome_source == "counts":
    outcome = transcriptome.X.dense()
else:
    outcome = torch.from_numpy(transcriptome.adata.layers["magic"])

# %% [markdown]
# ## Neighbors

# %%
import faiss

# X = np.array(transcriptome.adata.X.todense())
X = transcriptome.adata.obsm["X_pca"]

index = faiss.index_factory(X.shape[1], "Flat")
index.train(X)
index.add(X)
distances, neighbors = index.search(X, 50)
neighbors = neighbors[:, 1:]

# %% [markdown]
# ## Load folds

# %%
# if not (scores_dir_overall / "transcriptome_predicted_full.zarr").exists():
#     scores_dir_overall = prediction.path / "scoring" / "overall"
#     transcriptome_predicted_full = pickle.load(
#         (scores_dir_overall / "transcriptome_predicted_full.pkl").open("rb")
#     )
#     transcriptome_predicted_full = np.stack(transcriptome_predicted_full.values)
#     import zarr

#     zarr.save(
#         scores_dir_overall / "transcriptome_predicted_full.zarr",
#         transcriptome_predicted_full,
#     )

# %%
# from numcodecs import Blosc

# compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
# zarr.save(
#     scores_dir_overall / "transcriptome_predicted_full.zarr",
#     transcriptome_predicted_full,
#     compressor=compressor,
# )

# %%
# import h5py

# with h5py.File(scores_dir_overall / "transcriptome_predicted_full.h5", "w") as hf:
#     for k, val in transcriptome_predicted_full.items():
#         hf.create_dataset(str(k), data=val)

# %%
# transcriptome_predicted_full = h5py.File(
#     scores_dir_overall / "transcriptome_predicted_full.h5", "r"
# )

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))
folds, cellxgene_batch_size = get_folds_inference(fragments, folds, n_cells_step=2000)
folds = folds  # [:1]

# %% [markdown]
# ## Nothing

# %%
scorer_folder = prediction.path / "scoring" / "nothing"
scorer_folder.mkdir(exist_ok=True, parents=True)


# %%
nothing_filterer = chd.scoring.prediction.filterers.NothingFilterer()
Scorer2 = chd.scoring.prediction.Scorer2
nothing_scorer = Scorer2(
    models,
    folds[: len(models)],
    loaders,
    outcome,
    fragments.var.index,
    fragments.obs.index,
    device=device,
)

# %%
models = [model.to("cpu") for model in models]

# %%
nothing_scoring = nothing_scorer.score(
    filterer=nothing_filterer,
    extract_total=True,
)


# %%
scorer_folder = prediction.path / "scoring" / "nothing"
scorer_folder.mkdir(exist_ok=True, parents=True)
nothing_scoring.save(scorer_folder)

# %%
nothing_scoring.genescores.sel(phase="validation")["cor"].mean("model").sel(
    i=0
).to_pandas().plot(kind="hist")

# %% [markdown]
# ## Size

# %%
# load nothing scoring
scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)

# %%
size_filterer = chd.scoring.prediction.filterers.SizeFilterer(20)
Scorer2 = chd.scoring.prediction.Scorer2
size_scorer = Scorer2(
    models,
    folds[: len(models)],
    loaders,
    outcome,
    fragments.var.index,
    fragments.obs.index,
    device=device,
)

# %%
models = [model.to("cpu") for model in models]
size_scoring = size_scorer.score(
    transcriptome_predicted_full=transcriptome_predicted_full,
    filterer=size_filterer,
    extract_per_cellxgene=False,
    nothing_scoring=nothing_scoring,
)

# %%
scorer_folder = prediction.path / "scoring" / "size"
scorer_folder.mkdir(exist_ok=True, parents=True)
size_scoring.save(scorer_folder)

# %% [markdown]
# ## Subset

# %%
folds = pickle.load((fragments.path / "folds" / (splitter + ".pkl")).open("rb"))

# symbol = "SELL"
# symbol = "BACH2"
# symbol = "CTLA4"
# symbol = "IL1B"
# symbol = "SPI1"
# symbol = "IL1B"
# symbol = "TCF3"
# symbol = "CCL4"
# symbol = "CD74"
symbol = "BCL2"
symbol = "ZNF652"
symbol = "IPP"
# symbol = "LYN"
# symbol = "CD74"
symbol = "TNFAIP2"
symbol = "CD74"
symbol = "BCL2"
# symbol = "BCL11B"
genes_oi = transcriptome.var["symbol"] == symbol
gene = transcriptome.var.index[genes_oi][0]
folds, cellxgene_batch_size = get_folds_inference(
    fragments, folds, n_cells_step=2000, genes_oi=genes_oi
)
folds = folds  # [:1]

gene_ix = transcriptome.gene_ix(symbol)
gene = transcriptome.var.iloc[gene_ix].name

# %%
sc.pl.umap(transcriptome.adata, color=gene, use_raw=False, show=False)

# %% [markdown]
# ## Window

# %%
# load nothing scoring
scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)

# %% [markdown]
# ### Score

# %%
Scorer2 = chd.scoring.prediction.Scorer2
window_scorer = Scorer2(
    models,
    folds[: len(models)],
    loaders,
    outcome,
    fragments.var.index[genes_oi],
    fragments.obs.index,
    device=device,
)

# %%
window_filterer = chd.scoring.prediction.filterers.WindowFilterer(
    window, window_size=100
)

# %%
window_scoring = window_scorer.score(
    filterer=window_filterer,
    nothing_scoring=nothing_scoring,
)

# %%
scores_folder = prediction.path / "scoring" / "window_gene_full" / gene
scores_folder.mkdir(exist_ok=True, parents=True)
window_scoring.save(scores_folder)

# %% [markdown]
# ### Load

# %%
# scores_folder = prediction.path / "scoring" / "window_gene_full" / gene
# window_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
sns.scatterplot(
    x=window_scoring.genescores["deltacor"]
    .sel(phase="test")
    .mean("model")
    .values.flatten(),
    y=window_scoring.genescores["deltacor"]
    .sel(phase="validation")
    .mean("model")
    .values.flatten(),
    hue=np.log1p(
        window_scoring.genescores["lost"]
        .sel(phase="validation")
        .mean("model")
        .values.flatten()
    ),
)

# %%
window_scoring.genescores["retained"].sel(gene=gene).sel(phase="train").mean(
    "model"
).plot()
window_scoring.genescores["retained"].sel(gene=gene).sel(phase="validation").mean(
    "model"
).plot()
window_scoring.genescores["retained"].sel(gene=gene).sel(phase="test").mean(
    "model"
).plot()

# %%
# genescores["cor"].mean("model").sel(phase = "train").sel(gene = transcriptome.gene("IL1B")).plot()
# genescores["cor"].mean("model").sel(phase = "validation").sel(gene = transcriptome.gene("CTLA4")).plot()
fig, ax = plt.subplots()
window_scoring.genescores["deltacor"].sel(phase="validation").sel(gene=gene).mean(
    "model"
).plot(ax=ax)
window_scoring.genescores["deltacor"].sel(phase="test").sel(gene=gene).mean(
    "model"
).plot(ax=ax, color="blue")
# ax2 = ax.twinx()
# window_scoring.genescores["effect"].sel(phase="test").mean("gene").mean("model").plot(
#     ax=ax2, color="red"
# )
# window_scoring.genescores["effect"].sel(phase="validation").mean("gene").mean(
#     "model"
# ).plot(ax=ax2, color="orange")

# %%
nothing_scoring.genescores["cor"].sel(
    gene=gene, phase="test"
).mean().item(), window_scoring.genescores["deltacor"].sel(
    gene=gene, phase="test"
).mean(
    "model"
).sum().item()

# %% [markdown]
# ### Explore celltype

# %%
effects = window_scoring.effects
losts = window_scoring.losts
deltacors = window_scoring.deltacors

cellgenedeltacors = window_scoring.cellgenedeltacors


# %%
cors = []
for x in tqdm.tqdm(cellgenedeltacors):
    cors.append(np.corrcoef(x.sel(gene=gene).values))
cors = np.stack(cors)
cors[np.isnan(cors)] = 0

# %%
random_cors = []
for i in tqdm.tqdm(range(100)):
    random_cors_ = []
    for x in cellgenedeltacors:
        random_cors_.append(
            np.corrcoef(
                x.sel(gene=gene).values,
                x.sel(gene=gene).values[:, np.random.permutation(x.shape[1])],
            )[x.shape[0] :, : x.shape[0]]
        )
    random_cors.append(random_cors_)
random_cors = np.stack(random_cors)
random_cors[np.isnan(random_cors)] = 0

# %%
random_cors = np.concatenate(
    [
        random_cors,
        np.swapaxes(random_cors, -1, -2),
    ],
    0,
)

# %%
import scipy.stats

# %%
pvals = 1 - (cors >= random_cors).mean(0)
# chipvals = [
#     scipy.stats.combine_pvalues(pval).pvalue
#     for pval in pvals.reshape((pvals.shape[0], -1)).T
# ]
# chis = -2 * np.log(pvals + 1e-3).sum(0)
# chipvals = 1 - scipy.stats.chi2.cdf(chis, df=2 * pvals.shape[0])

# %%
cors_flattened = cors.reshape((cors.shape[0], -1))
interaction = pd.DataFrame(
    {
        "cor": cors_flattened.mean(0),
        "window_ix1": np.tile(
            np.arange(cors.shape[1])[:, None], cors.shape[1]
        ).flatten(),
        "window_ix2": np.tile(
            np.arange(cors.shape[1])[:, None], cors.shape[1]
        ).T.flatten(),
        "window1": np.tile(
            window_scoring.genescores.coords["window"].values[:, None],
            len(window_scoring.genescores.coords["window"].values),
        ).flatten(),
        "window2": np.tile(
            window_scoring.genescores.coords["window"].values[:, None],
            len(window_scoring.genescores.coords["window"].values),
        ).T.flatten(),
    }
)
interaction["distance"] = np.abs(interaction["window1"] - interaction["window2"])
interaction["pval"] = (1 - (cors >= random_cors).mean(0).mean(0)).flatten()
# interaction["pval"] = 1 - (cors.mean(0) > random_cors.mean(1)).mean(0).flatten()
# interaction = interaction.query("distance > 500")
interaction["deltacor1"] = (
    window_scoring.genescores.sel(gene=gene)
    .mean("model")
    .sel(phase="test")["deltacor"]
    .to_pandas()[interaction["window1"]]
    .values
)
interaction["deltacor2"] = (
    window_scoring.genescores.sel(gene=gene)
    .mean("model")
    .sel(phase="test")["deltacor"]
    .to_pandas()[interaction["window2"]]
    .values
)
# interaction = interaction.query("deltacor1 < -0.01")

# %%

# import scipy.stats

# interaction["pval"] = scipy.stats.ttest_1samp(
#     cors_flattened, 0, alternative="greater"
# ).pvalue
# interaction = interaction.query("distance > 500")
# import statsmodels.stats.multitest

import statsmodels.stats.multitest

interaction["qval"] = statsmodels.stats.multitest.fdrcorrection(interaction["pval"])[1]


# %%
sns.heatmap(
    interaction.set_index(["window1", "window2"])["qval"].unstack(), vmin=0, vmax=0.05
)

# %%
interaction["significant"] = interaction["qval"] < 0.05
interaction["cor_masked"] = interaction["cor"]
interaction.loc[~interaction["significant"], "cor_masked"] = 0.0

# %%

interaction["deltacor_prod"] = np.abs(interaction["deltacor1"]) * np.abs(
    interaction["deltacor2"]
)

# %%
interaction_oi = interaction.query("(distance > 1000)")
plt.scatter(
    interaction_oi["deltacor_prod"],
    interaction_oi["cor"],
    c=interaction_oi["distance"],
)

# %%
sns.heatmap(
    interaction_oi.set_index(["window1", "window2"])["cor_masked"].unstack(),
    vmin=0,
    vmax=0.2,
)

# %% [markdown]
# ### Per celltype

# %%
deltacors_celltype = {
    celltype: pd.Series(0, index=window_scoring.genescores.coords["window"].values)
    for celltype in transcriptome.adata.obs["celltype"].unique()
}
for celltype, obs_celltype in transcriptome.adata.obs.groupby("celltype"):
    # deltacors_celltype[celltype] = (
    #     deltacors.sel(gene
    # =gene, cell=obs_celltype.index).mean("cell").to_pandas()
    # )
    for cellgenedeltacors_fold in cellgenedeltacors:
        deltacors_celltype[celltype] += (
            cellgenedeltacors_fold.sel(
                gene=gene,
                cell=obs_celltype.index.intersection(
                    cellgenedeltacors_fold.coords["cell"]
                ),
            )
            .mean("cell")
            .to_pandas()
        )

deltacors_celltype = pd.concat(deltacors_celltype)

# %%
x = deltacors_celltype.unstack()
# x = x.loc[:, x.columns > 50000]
# x = x.loc[:, (x.columns < 10000) & (x.columns > -10000)]
sns.heatmap(x)

# %%
window_scoring.genescores["cor"].sel(phase="test").mean("model").mean(
    "gene"
).to_pandas().plot()


# %%
fig, ax = plt.subplots()
# deltacors.mean("gene").mean("cell").to_pandas().plot()
# deltacors.mean("gene").sel(
#     cell=transcriptome.obs.query("celltype == 'CD8 naive T'").index
# ).sum("cell").to_pandas().plot()
# deltacors.mean("gene").sel(
#     cell=transcriptome.obs.query("celltype == 'CD4 naive T'").index
# ).sum("cell").to_pandas().plot()
# deltacors.mean("gene").sel(
#     cell=transcriptome.obs.query("celltype == 'CD14+ Monocytes'").index
# ).sum("cell").to_pandas().plot()
# deltacors.mean("gene").sel(
#     cell=transcriptome.obs.query("celltype == 'memory B'").index
# ).mean("cell").to_pandas().plot(label="memory B")
# deltacors.mean("gene").sel(
#     cell=transcriptome.obs.query("celltype == 'CD14+ Monocytes'").index
# ).mean("cell").to_pandas().plot(label="CD14+ Monocytes")
deltacors.mean("gene").sel(
    cell=transcriptome.obs.query("celltype == 'memory B'").index
).mean("cell").to_pandas().plot(label="memory B")
deltacors.mean("gene").sel(
    cell=transcriptome.obs.query("celltype == 'CD14+ Monocytes'").index
).mean("cell").to_pandas().plot(label="CD14+ Monocytes")
deltacors.mean("gene").sel(
    cell=transcriptome.obs.query("celltype == 'FCGR3A+ Monocytes'").index
).mean("cell").to_pandas().plot(label="FCGR3A+ Monocytes")

# deltacors.mean("gene").sel(
#     cell=transcriptome.obs.query("celltype == 'pDCs'").index
# ).mean("cell").to_pandas().plot(label="pDCs")
plt.legend()
# ax.set_xlim(-500, 500)

# %%
cors = np.corrcoef(deltacors.sel(gene=gene))
cors = pd.DataFrame(
    cors,
    index=deltacors.sel(gene=gene).coords["window"].values,
    columns=deltacors.sel(gene=gene).coords["window"].values,
)
sns.heatmap(cors, vmax=0.1, vmin=-0.02)

# %%
interaction = cors.unstack().to_frame()
interaction.columns = ["deltacor_interaction"]
interaction.index.names = ("window1", "window2")
interaction["deltacor1"] = deltacors.sel(gene=gene).sum("cell").values.flatten()

# %%
fig, ax = plt.subplots()
# for window in [-50, 9450]:
for window in [-50, 5950, 4350]:
    cors.loc[window].plot(ax=ax, label=window)
plt.legend()

ax.set_ylim(0, 0.5)

# %%
# deltacors_celltype["naive B"].plot()
deltacors_celltype["CD14+ Monocytes"].plot()
deltacors_celltype["cDCs"].plot()
deltacors_celltype["memory B"].plot()

# %%
sns.heatmap(pd.concat(deltacors_celltype, axis=1))

# %%
transcriptome.adata.obs.query("celltype == 'CD14+ Monocytes'")

# %% [markdown]
# ### Smooth effects

# %%
effects = window_scoring.effects
losts = window_scoring.losts
deltacors = window_scoring.deltacors

# %%
# effects_raw = losts.sel(gene = gene).to_pandas()
# effects_raw = effects.sel(gene = gene).to_pandas()
effects_raw = window_scoring.deltacors.sel(gene=gene).to_pandas()
effects_smooth = effects_raw.T.values[neighbors].mean(1)
effects_smooth = pd.DataFrame(
    effects_smooth.T, index=effects_raw.index, columns=effects_raw.columns
)

# %%
sc.pl.umap(transcriptome.adata, color=["celltype"])

# %%
transcriptome.adata.obs = transcriptome.adata.obs.copy()
transcriptome.adata.obs = transcriptome.adata.obs.assign(
    **{"effect_" + str(window): effect for window, effect in effects_smooth.iterrows()}
)

# %%
transcriptome.adata.obs["log_n_counts"] = np.log10(transcriptome.adata.obs["n_counts"])

# %%
sc.pl.umap(
    transcriptome.adata,
    color=[
        gene,
        # transcriptome.gene_id("CD74"),
        "celltype"
        # "log_n_counts",
    ],
    layer="magic",
)

# %%
# windows_oi2 = interaction.sort_values("deltacor_interaction").iloc[-1][["window1", "window2"]].values
windows_oi2 = (
    window_scoring.genescores["deltacor"]
    .sel(gene=gene, phase="validation")
    .mean("model")
    .to_pandas()
    .sort_values()
    .index[:10]
)
windows_oi2 = [-6850.0, 150.0, -150.0]
windows_oi2 = [-50, 9550]


# %%
def plot_windows_oi2(windows_oi2):
    main = chd.grid.Wrap()
    fig = chd.grid.Figure(main)
    for window_oi2 in windows_oi2:
        panel = fig.main.add(chd.grid.Panel((5, 5)))
        ax = panel.ax
        # plotdata = effects_smooth.loc[window_oi2]
        plotdata = effects_raw.loc[window_oi2]
        cellorder = np.argsort(np.abs(plotdata))
        norm = mpl.colors.CenteredNorm(halfrange=0.001)
        cmap = mpl.colors.ListedColormap(["#FF4136", "#DDD", "#0074D9"])
        # norm = mpl.colors.CenteredNorm(halfrange = np.quantile(np.abs(plotdata), 0.95))
        ax.scatter(
            *[*transcriptome.adata.obsm["X_umap"][cellorder].T],
            c=plotdata[cellorder],
            s=1,
            norm=norm,
            cmap=cmap,
        )
        ax.set_title(window_oi2)
        ax.axis("off")
    fig.plot()
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    return fig


# %%
plot_windows_oi2(windows_oi2)

# %% [markdown]
# ### Explore ambiguity

# %%
window_scoring.genescores["deltacor_down_ratio"] = (deltacors < -0.01).sum("cell") / (
    deltacors > 0.01
).sum("cell")

# %%
window_scores = (
    window_scoring.genescores.sel(gene=gene, phase="validation")
    .mean("model")
    .to_pandas()
    .join(window_filterer.design)
)
window_scores["deltacor_down_ratio"] = (deltacors.sel(gene=gene) < -0.01).sum(
    "cell"
) / (deltacors.sel(gene=gene) > 0.01).sum("cell")

# %%
sns.heatmap(
    deltacors.sel(cell=transcriptome.obs.query("celltype == 'memory B'").index)
    .sel(gene=gene)
    .to_pandas()
)
deltacors.sel(cell=transcriptome.obs.query("celltype == 'memory B'").index).median(
    "window"
).to_pandas().sort_values("ENSG00000019582").head(20)

# %%
fig, ax = plt.subplots()
# (deltacors > 0.01).sel(gene = gene).to_pandas().sum(1).plot(ax = ax)
# (deltacors < -0.01).sel(gene = gene).to_pandas().sum(1).plot(ax = ax)
plotdata = window_scores.query("deltacor < -0.001")
cmap = mpl.cm.get_cmap("RdBu_r")
ax.scatter(
    plotdata["window_mid"],
    -plotdata["deltacor"],
    c=np.log(plotdata["deltacor_down_ratio"]),
    s=10,
    cmap=cmap,
    norm=mpl.colors.CenteredNorm(),
)
ax.axhline(0)
# ax2 = ax.twinx()
ax.plot(window_scores["window_mid"], -window_scores["deltacor"], color="red", zorder=0)

# %%
window_scores["log_deltacor_down_ratio"] = np.log2(window_scores["deltacor_down_ratio"])
window_scores.query("deltacor < -0.001").style.bar(
    subset=["log_deltacor_down_ratio", "deltacor", "effect"]
)

# %%
windows_oi2 = [
    *window_scores.query("deltacor < -0.001")
    .sort_values("log_deltacor_down_ratio")
    .index[:3],
    *window_scores.query("deltacor < -0.001")
    .sort_values("log_deltacor_down_ratio")
    .index[-3:],
]
# windows_oi2 = [-37500.0	]

# %%
plot_windows_oi2(windows_oi2)

# %% [markdown]
# ## Pairwindow

# %%
windows_oi = window_scoring.design.loc[
    (
        window_scoring.genescores["deltacor"].sel(gene=gene, phase="test") < -0.0001
    ).values
]
# windows_oi = window_filterer.design
windows_oi.shape

# %%
windowpair_filterer = chd.scoring.prediction.filterers.WindowPairFilterer(windows_oi)
windowpair_scorer = chd.scoring.prediction.Scorer2(
    models,
    folds[: len(models)],
    loaders,
    outcome,
    fragments.var.index[genes_oi],
    fragments.obs.index,
    device=device,
)
windowpair_scoring = windowpair_scorer.score(
    transcriptome_predicted_full=transcriptome_predicted_full,
    filterer=windowpair_filterer,
    nothing_scoring=nothing_scoring,
)

# %%
windowpair_baseline_filterer = (
    chd.scoring.prediction.filterers.WindowPairBaselineFilterer(windowpair_filterer)
)
windowpair_baseline_scorer = chd.scoring.prediction.Scorer2(
    models,
    folds[: len(models)],
    loaders,
    outcome,
    fragments.var.index[genes_oi],
    fragments.obs.index,
    device=device,
)
windowpair_baseline_scoring = windowpair_baseline_scorer.score(
    transcriptome_predicted_full=transcriptome_predicted_full,
    filterer=windowpair_baseline_filterer,
    nothing_scoring=nothing_scoring,
)

# %% [markdown]
# ### Interpret

# %%
retained_additive = pd.Series(
    1
    - (
        (
            1
            - (
                window_scoring.genescores["retained"]
                .sel(gene=gene)
                .sel(
                    window=windowpair_filterer.design["window2"].values,
                    phase="validation",
                )
            ).values
        )
        + (
            1
            - (
                window_scoring.genescores["retained"]
                .sel(gene=gene)
                .sel(
                    window=windowpair_filterer.design["window1"].values,
                    phase="validation",
                )
            ).values
        )
    ),
    windowpair_filterer.design.index,
)

# %%
# because some fragments may be in two windows, we need to use a baseline to correct for this
additive_baseline = windowpair_baseline_scoring.genescores["cor"].sel(gene=gene)
additive_base1 = (
    window_scoring.genescores["cor"]
    .sel(gene=gene)
    .sel(window=windowpair_filterer.design["window1"].values)
).reindex_like(additive_baseline)
additive_base2 = (
    window_scoring.genescores["cor"]
    .sel(gene=gene)
    .sel(window=windowpair_filterer.design["window2"].values)
).reindex_like(additive_baseline)

deltacor1 = additive_base1.values - additive_baseline
deltacor2 = additive_base2.values - additive_baseline
deltacor_additive = (
    additive_base1.values + additive_base2.values
) - 2 * additive_baseline
deltacor_interacting = (
    windowpair_scoring.genescores["cor"].sel(gene=gene) - additive_baseline
)
reldeltacor_interacting = deltacor_interacting - np.minimum(deltacor1, deltacor2)

# %%
phase = "test"
phase = "validation"

interaction = windowpair_scoring.design.copy()
interaction["deltacor"] = deltacor_interacting.sel(phase=phase).to_pandas()
interaction["reldeltacor"] = reldeltacor_interacting.sel(phase=phase).to_pandas()
interaction["deltacor1"] = deltacor1.sel(phase=phase).to_pandas()
interaction["deltacor2"] = deltacor2.sel(phase=phase).to_pandas()

additive = windowpair_scoring.design.copy()
additive["deltacor"] = deltacor_additive.sel(phase=phase).to_pandas()

# %%
sns.heatmap(
    interaction.set_index(["window_mid1", "window_mid2"])["reldeltacor"].unstack()
)

# %%
sns.heatmap(interaction.set_index(["window_mid1", "window_mid2"])["deltacor"].unstack())

# %%
interaction["deltacor_interaction"] = interaction["deltacor"] - additive["deltacor"]

# %%
# %config InlineBackend.figure_format='retina'

# %%
radius = 500

# %%
fig, ax = plt.subplots(figsize=(20, 10))

norm = mpl.colors.Normalize(-0.001, 0.001)
norm = mpl.colors.Normalize(-0.0001, 0.0001)

cmap = mpl.cm.RdBu

for (window_mid1, window_mid2), deltacor in interaction.set_index(
    ["window_mid1", "window_mid2"]
)["deltacor_interaction"].items():
    patch = mpl.patches.RegularPolygon(
        (
            window_mid1 + (window_mid2 - window_mid1) / 2,
            (window_mid2 - window_mid1) / 2,
        ),
        4,
        radius=radius,
        orientation=np.pi / 2,
        ec=None,
        lw=0,
        fc=cmap(norm(deltacor)),
    )
    ax.add_patch(patch)
ax.set_ylim((window[1] - window[0]) / 2)
ax.set_xlim(*window)

for x in [0, 22500, -16500]:
    ax.plot(
        [window[0] + x, x],
        [(window[1] - window[0]) / 2, 0],
        zorder=-1,
        color="#333",
        lw=1,
    )
    ax.plot(
        [window[1] + x, x],
        [(window[1] - window[0]) / 2, 0],
        zorder=-1,
        color="#333",
        lw=1,
    )

ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")


# %%
window_scoring.genescores.sel(gene=gene, phase="test")["effect"].plot()

# %%
fig, ax = plt.subplots()
x = interaction.set_index(["window_mid1", "window_mid2"])[
    "deltacor_interaction"
].unstack()
x[np.isnan(x)] = 0

# make x symmetric
x = x + x.T

# cluster x and sort by leaves
import scipy.cluster.hierarchy

dist = scipy.spatial.distance.squareform((1 - np.corrcoef(x)), checks=False)
linkage = scipy.cluster.hierarchy.linkage(dist, method="complete")
order = scipy.cluster.hierarchy.leaves_list(linkage)
x = x.iloc[order, order]
# x = x.iloc[leaves_list[linkage_list[x.index].max().astype(int)], leaves_list[linkage_list[x.columns].max().astype(int)]]

ax.matshow(
    x,
    cmap=cmap,
    norm=norm,
)
ax.set_xticks(np.arange(len(x.columns)))
ax.set_yticks(np.arange(len(x.index)))
ax.set_xticklabels(x.columns, rotation=90, va="bottom", ha="center", fontsize=5)
ax.set_yticklabels(x.index, rotation=0, fontsize=5)

# %% [markdown]
# ### Interaction versus distance

# %%
interaction["distance"] = np.abs(
    interaction["window_mid1"] - interaction["window_mid2"]
)

# %%
interaction["deltacor_min"] = interaction[["deltacor1", "deltacor2"]].min(1)
interaction["deltacor_max"] = interaction[["deltacor1", "deltacor2"]].max(1)

# %%
interaction["deltacor_interaction_ratio"] = (
    interaction["deltacor_interaction"] / interaction["deltacor_max"]
)

# %%
plt.scatter(
    interaction["deltacor_max"],
    interaction["deltacor_interaction"],
    c=np.log1p(interaction["distance"]),
)

# %%
plt.scatter(
    interaction["deltacor_min"],
    interaction["deltacor_interaction"],
    c=interaction["distance"],
)

# %% [markdown]
# ### Do ambiguous predictive regions have to interact to regulate gene expression?

# %%
# make symmetric
window_scores["interacting"] = (
    interaction.set_index(["window1", "window2"])["deltacor_interaction"]
    .unstack()
    .fillna(0)
    + interaction.set_index(["window2", "window1"])["deltacor_interaction"]
    .unstack()
    .fillna(0)
).mean(0)

# %%
window_scores["interacting_ratio"] = (
    window_scores["interacting"] / window_scores["deltacor"]
)
window_scores["log_interacting_ratio"] = np.log2(window_scores["interacting_ratio"])
window_scores_oi = window_scores.loc[~pd.isnull(window_scores["interacting"])]

# %%
sns.regplot(
    x=window_scores_oi["log_deltacor_down_ratio"],
    y=window_scores_oi["log_interacting_ratio"],
)

# %% [markdown]
# ## Multiwindow

# ### Score

# %%
multiwindow_scorer = Scorer2(
    models,
    folds[: len(models)],
    loaders,
    outcome,
    fragments.var.index[genes_oi],
    fragments.obs.index,
    device=device,
)

# %%
window_sizes = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
window_sizes = (500,)

# %%
multiwindow_filterer = chd.scoring.prediction.filterers.MultiWindowFilterer(
    window, window_sizes=window_sizes
)

# %%
multiwindow_scoring = multiwindow_scorer.score(
    transcriptome_predicted_full=transcriptome_predicted_full,
    filterer=multiwindow_filterer,
    nothing_scoring=nothing_scoring,
)

# %%
scorer_folder = prediction.path / "scoring" / "nothing"
scorer_folder.mkdir(exist_ok=True, parents=True)
multiwindow_scoring.save(scorer_folder)


# %%
plotdata = multiwindow_scoring.genescores.sel(gene=gene).stack().to_dataframe()
plotdata = multiwindow_scoring.design.join(plotdata)


# %%
fig, ax = plt.subplots()
plt.scatter(plotdata.loc["validation"]["deltacor"], plotdata.loc["test"]["deltacor"])
ax.set_xlim(-0.01)
ax.set_ylim(-0.01)

# %%
window_sizes_info = pd.DataFrame({"window_size": window_sizes}).set_index("window_size")
window_sizes_info["ix"] = np.arange(len(window_sizes_info))

# %%
fig, ax = plt.subplots(figsize=(20, 3))

deltacor_norm = mpl.colors.Normalize(0, 0.001)
deltacor_cmap = mpl.cm.Reds

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = (
        plotdata.query("window_size == @window_size")
        .query("phase == 'validation'")
        .iloc[::2]
    )
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        rect = mpl.patches.Rectangle(
            (plotdata_row["window_start"], y),
            plotdata_row["window_end"] - plotdata_row["window_start"],
            1,
            lw=0,
            fc=deltacor_cmap(deltacor_norm(-plotdata_row["deltacor"])),
        )
        ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)
ax.axvline(6000)

# %%
fig, ax = plt.subplots(figsize=(20, 3))

effect_norm = mpl.colors.CenteredNorm()
effect_cmap = mpl.cm.RdBu_r

for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size")
    print(plotdata_oi.shape)
    y = window_size_info["ix"]
    for _, plotdata_row in plotdata_oi.iterrows():
        if plotdata_row["deltacor"] < -0.001:
            rect = mpl.patches.Rectangle(
                (plotdata_row["window_start"], y),
                plotdata_row["window_end"] - plotdata_row["window_start"],
                1,
                lw=0,
                fc=effect_cmap(effect_norm(-plotdata_row["effect"])),
            )
            ax.add_patch(rect)
ax.set_xlim(*window)
ax.set_ylim(0, window_sizes_info["ix"].max() + 1)
ax.axvline(6000)

# %% [markdown]
# ### Interpolate per position

# %%
positions_oi = np.arange(*window)

# %%
deltacor_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
retained_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
for window_size, window_size_info in window_sizes_info.iterrows():
    plotdata_oi = plotdata.query("window_size == @window_size")
    deltacor_interpolated_ = np.clip(
        np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["deltacor"])
        / window_size
        * 1000,
        -np.inf,
        0,
    )
    deltacor_interpolated[window_size_info["ix"], :] = deltacor_interpolated_
    retained_interpolated_ = (
        np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["retained"])
        / window_size
        * 1000
    )
    retained_interpolated[window_size_info["ix"], :] = retained_interpolated_

# %%
fig, ax = plt.subplots(figsize=(20, 3))
plt.plot(positions_oi, deltacor_interpolated.mean(0))
ax2 = ax.twinx()
ax2.plot(positions_oi, retained_interpolated.mean(0), color="red", alpha=0.6)
ax2.set_ylabel("retained")

# %% [markdown]
# ## Variants/haplotypes

# %%
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
promoter = promoters.loc[gene]

# %%
motifscan_name = "gwas_immune"

# %%
folder_qtl = chd.get_output() / "data" / "qtl" / "hs" / "gwas"
snp_info = pickle.load((chd.get_output() / "snp_info.pkl").open("rb"))
qtl_mapped = pd.read_pickle(folder_qtl / ("qtl_mapped_" + motifscan_name + ".pkl"))
qtl_mapped.index = np.arange(len(qtl_mapped))
association = qtl_mapped.join(snp_info, on="snp")
association = association.loc[~pd.isnull(association["start"])]
association["pos"] = association["start"].astype(int)

# %%
association_oi = association.loc[
    (association["chr"] == promoter["chr"])
    & (association["pos"] >= promoter["start"])
    & (association["pos"] <= promoter["end"])
].copy()

# %%
association_oi["position"] = (association_oi["pos"] - promoter["tss"]) * promoter[
    "strand"
]

# %%
variants = pd.DataFrame(
    {
        "disease/trait": association_oi.groupby("snp")["disease/trait"].apply(list),
        "snp_main_first": association_oi.groupby("snp")["snp_main"].first(),
    }
)
variants = variants.join(snp_info)
variants["position"] = (variants["start"] - promoter["tss"]) * promoter["strand"]

haplotypes = (
    association_oi.groupby("snp_main")["snp"]
    .apply(lambda x: sorted(set(x)))
    .to_frame("snps")
)
haplotypes["color"] = sns.color_palette("hls", n_colors=len(haplotypes))

# %% [markdown]
# ### Compare to individual position ranking

# %%
fig, ax = plt.subplots(figsize=(20, 3))
ax.plot(
    positions_oi * promoter["strand"] + promoter["tss"], deltacor_interpolated.mean(0)
)
ax2 = ax.twinx()
ax2.plot(
    positions_oi * promoter["strand"] + promoter["tss"],
    retained_interpolated.mean(0),
    color="red",
    alpha=0.6,
)
ax2.set_ylabel("retained")

for _, variant in variants.iterrows():
    ax.scatter(
        variant["position"] * promoter["strand"] + promoter["tss"],
        0.9,
        color=haplotypes.loc[variant["snp_main_first"], "color"],
        s=20,
        marker="|",
        transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
    )

ax.invert_yaxis()
ax2.invert_yaxis()

# %%
import gseapy

# %%
rnk = pd.Series(0, index=pd.Series(list("abcdefghijklmnop")))
rnk.values[:] = -np.arange(len(rnk))
genesets = {"hi": ["a", "b", "c"]}

# %%
rnk = -pd.Series(deltacor_interpolated.mean(0), index=positions_oi.astype(str))
genesets = {"hi": np.unique(variants["position"].astype(str).values)}

# %%
# ranked = gseapy.prerank(rnk, genesets, min_size = 0)

# %%
rnk_sorted = pd.Series(np.sort(np.log(rnk)), index=rnk.index)
# rnk_sorted = pd.Series(np.sort(rnk), index = rnk.index)
fig, ax = plt.subplots()
sns.ecdfplot(rnk_sorted, ax=ax)
sns.ecdfplot(
    rnk_sorted[variants["position"].astype(int).astype(str)], ax=ax, color="orange"
)
for _, motifdatum in variants.iterrows():
    rnk_motif = rnk_sorted[str(int(motifdatum["position"]))]
    q = np.searchsorted(rnk_sorted, rnk_motif) / len(rnk_sorted)
    ax.scatter([rnk_motif], [q], color="red")
    # ax.scatter(motifdatum["position"], 0, color = "red", s = 5, marker = "|")

# %% [markdown]
# ## Around variants

# %%
# load nothing scoring
scorer_folder = prediction.path / "scoring" / "nothing"
nothing_scoring = chd.scoring.prediction.Scoring.load(scorer_folder)

# %%
variant_filterer = chd.scoring.prediction.filterers.VariantFilterer(
    variants["position"], window_sizes=(500,)
)
variant_scorer = chd.scoring.prediction.Scorer2(
    models,
    folds[: len(models)],
    loaders,
    outcome,
    fragments.var.index[genes_oi],
    fragments.obs.index,
    device=device,
)
variant_scoring = variant_scorer.score(
    transcriptome_predicted_full=transcriptome_predicted_full,
    filterer=variant_filterer,
    nothing_scoring=nothing_scoring,
)

# %% [markdown]
# ### Prioritize

# %%
variant_size_scores = (
    variant_scoring.genescores.sel(gene=gene)
    .sel(phase="test")
    .mean("model")
    .to_pandas()
    .join(variant_filterer.design)
)
variant_size_scores["deltacor_norm"] = (
    variant_size_scores["deltacor"] / variant_size_scores["window_size"] * 1000
)
variant_scores = variant_size_scores.groupby("snp").mean(numeric_only=True)
variant_scores["deltacor"] = variant_scores["deltacor_norm"]

# %%
gene

# %%
variant_prioritized_scores = (
    variant_scores.join(variants)
    .reset_index()
    .sort_values("deltacor", ascending=True)
    .groupby("snp_main_first")
    .first()
)
variant_prioritized_scores

# %%
haplotypes["n"] = haplotypes["snps"].apply(len)


# %%
def calculate_sample_kurtosis(x):
    n = len(x)
    return (1 / n * (x - x.mean()) ** 4).sum() / (
        (1 / n * (x - x.mean()) ** 2).sum()
    ) ** 2 - 3


# %%
np.corrcoef(variant_scores["deltacor"], variant_scores["retained"])

# %%
ratios = []
kurtoses = []
retained_cors = []
retained_matched = []
for haplotype, snps in haplotypes["snps"].items():
    variant_scores_oi = variant_scores.loc[snps]
    # print(haplotype)
    # print(variant_scores_oi["deltacor"].min())
    if len(snps) > 5:
        extremity = (
            variant_scores_oi["deltacor"].min() - variant_scores_oi["deltacor"].mean()
        )
        kurtosis = calculate_sample_kurtosis(variant_scores_oi["deltacor"].values)
        retained_cors.append(
            np.corrcoef(variant_scores_oi["deltacor"], variant_scores_oi["retained"])[
                0, 1
            ]
        )
        retained_matched.append(
            np.argmin(variant_scores_oi["deltacor"])
            == np.argmin(variant_scores_oi["retained"])
        )
        extremity_random = []
        kurtoses_random = []
        for i in range(100):
            # variant_scores_oi = variant_scores.sample(len(snps))
            variant_scores_oi = variant_scores_oi.copy()
            variant_scores_oi["deltacor"] = np.random.choice(
                deltacor_interpolated[0], len(snps)
            )
            extremity_random.append(
                variant_scores_oi["deltacor"].min()
                - variant_scores_oi["deltacor"].mean()
            )
            kurtoses_random.append(
                calculate_sample_kurtosis(variant_scores_oi["deltacor"])
            )
        # ratios.append(extremity / np.mean(extremity_random))
        ratios.append(kurtosis / np.mean(kurtoses_random))
        kurtoses.append(kurtosis)
kurtoses = np.array(kurtoses)

# %%
np.mean(retained_cors)

# %%
np.mean(retained_matched)

# %%
sns.ecdfplot(ratios)

# %%
fig, ax = plt.subplots(figsize=(20, 3))

# ax.plot(positions_oi  * promoter["strand"] + promoter["tss"], deltacor_interpolated.mean(0))
# ax2 = ax.twinx()
# ax2.plot(positions_oi  * promoter["strand"] + promoter["tss"], retained_interpolated.mean(0), color = "red", alpha = 0.6)
# ax2.set_ylabel("retained")
print(1)

for variant_id, variant in variants.iterrows():
    ax.scatter(
        variant["position"] * promoter["strand"] + promoter["tss"],
        variant_scores.loc[variant_id, "deltacor"],
        color=haplotypes.loc[variant["snp_main_first"], "color"],
        s=20,
        marker="o",
    )

ax.invert_yaxis()
ax2.invert_yaxis()

# %% [markdown]
# ### Smooth effects

# %%
effects = variant_scoring.effects
losts = variant_scoring.losts
deltacors = variant_scoring.deltacors


# %%
def smooth(effects_raw):
    effects_raw = (
        effects_raw  # / (variant_filterer.design["window_size"] * 1000).values[:, None]
    )
    effects_raw = effects_raw
    effects_smooth = effects_raw.T.values[neighbors].mean(1)
    effects_smooth = pd.DataFrame(
        effects_smooth.T, index=effects_raw.index, columns=effects_raw.columns
    )
    return effects_smooth


# %%
effects_raw = (
    effects.sel(gene=gene).to_pandas().groupby(variant_filterer.design["snp"]).mean()
)
effects_smooth = smooth(effects_raw)

losts_raw = (
    losts.sel(gene=gene).to_pandas().groupby(variant_filterer.design["snp"]).mean()
)
losts_smooth = smooth(losts_raw)

deltacors_raw = (
    deltacors.sel(gene=gene).to_pandas().groupby(variant_filterer.design["snp"]).mean()
)
deltacors_smooth = smooth(deltacors_raw)

# %%
transcriptome.adata.obs = transcriptome.adata.obs.copy()
transcriptome.adata.obs = transcriptome.adata.obs.assign(
    **{"effect_" + snp: effect for snp, effect in effects_smooth.iterrows()}
)

# %%
haplotypes

# %%
snps_oi = pd.Series(haplotypes["snps"].iloc[0])
# snps_oi = pd.Series(haplotypes.loc["rs3087243", "snps"])
# snps_oi = pd.Series(haplotypes.loc[association_oi.query("snp == 'rs4987360'")["snp_main"].values[0], "snps"]) # main SELL SNP

# %%
variant_scores.loc[snps_oi].style.bar(subset=["deltacor", "effect", "lost"])

# %%
main = chd.grid.Grid()
fig = chd.grid.Figure(main)
for i, snp in enumerate(snps_oi):
    panel = fig.main[0, i] = chd.grid.Panel((5, 5))
    ax = panel.ax
    plotdata = effects_raw.loc[snp]
    cellorder = np.argsort(np.abs(plotdata))
    norm = mpl.colors.CenteredNorm(halfrange=0.001)
    cmap = mpl.colors.ListedColormap(["#FF4136", "#DDD", "#0074D9"])
    ax.scatter(
        *[*transcriptome.adata.obsm["X_umap"][cellorder].T],
        c=plotdata[cellorder],
        s=1,
        norm=norm,
        cmap=cmap,
    )
    ax.axis("off")

    ax.set_title(snp)

    panel = fig.main[1, i] = chd.grid.Panel((5, 5))
    ax = panel.ax
    plotdata = losts_raw.loc[snp]
    cellorder = np.argsort(np.abs(plotdata))
    norm = mpl.colors.CenteredNorm(halfrange=0.001)
    cmap = mpl.colors.ListedColormap(["#FF4136", "#DDD", "#0074D9"])
    ax.scatter(
        *[*transcriptome.adata.obsm["X_umap"][cellorder].T],
        c=plotdata[cellorder],
        s=1,
        norm=norm,
        cmap=cmap,
    )
    ax.axis("off")

    panel = fig.main[2, i] = chd.grid.Panel((5, 5))
    ax = panel.ax
    plotdata = deltacors_raw.loc[snp]
    cellorder = np.argsort(np.abs(plotdata))
    norm = mpl.colors.CenteredNorm(halfrange=0.001)
    cmap = mpl.colors.ListedColormap(["#FF4136", "#DDD", "#0074D9"])
    ax.scatter(
        *[*transcriptome.adata.obsm["X_umap"][cellorder].T],
        c=plotdata[cellorder],
        s=1,
        norm=norm,
        cmap=cmap,
    )
    ax.scatter(
        *[*transcriptome.adata.obsm["X_umap"][cellorder[plotdata != 0]].T],
        c=plotdata[cellorder[plotdata != 0]],
        s=10,
        norm=norm,
        cmap=cmap,
    )
    ax.axis("off")

fig.plot()
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

# %% [markdown]
# ### Prioritized variants more connected

# %%
haplotype_scores = []
for haplotype, snps in haplotypes["snps"].items():
    variant_scores_oi = variant_scores.loc[snps]
    chosen_variant = variant_scores_oi["deltacor"].idxmin()
    haplotype_scores.append(
        {
            "snp": chosen_variant,
            "haplotype": haplotype,
            "score": variant_scores_oi["deltacor"].min(),
            "position": variant_scores_oi.loc[chosen_variant, "window_mid"],
        }
    )
haplotype_scores = pd.DataFrame(haplotype_scores).set_index("snp")

# %%
windowpair_scores = (
    windowpair_scoring.genescores.sel(gene=gene, phase="validation")
    .to_pandas()
    .join(windowpair_scoring.design)
)

# %%
window_scores = (
    window_scoring.genescores.sel(gene=gene, phase="validation")
    .to_pandas()
    .join(window_scoring.design)
)
window_scores["matched"] = False

# %%

for haplotype, haplotype_score in haplotype_scores.iterrows():
    # filter windowpair_scores on haplotype
    matched_windows = (haplotype_score["position"] > windows_oi["window_start"]) & (
        haplotype_score["position"] < windows_oi["window_end"]
    )
    if matched_windows.sum() == 0:
        continue
    window_oi = matched_windows[matched_windows].index[0]
    windowpairs_oi = windowpair_scores.query(
        "(window1 == @window_oi) or (window2 == @window_oi)"
    ).index
    window_scores.loc[window_oi, "matched"] = haplotype
    haplotype_scores.loc[haplotype, "interacting"] = (
        (deltacor_interacting)
        .sel(phase="test")
        .sel(windowpair=windowpairs_oi)
        .mean()
        .item()
    )
    haplotype_scores.loc[haplotype, "relinteracting"] = (
        (reldeltacor_interacting)
        .sel(phase="test")
        .sel(windowpair=windowpairs_oi)
        .mean()
        .item()
    )

# %% [markdown]
# Are QTLs more interacting than the average position (*which was somewhat predictive given that it was included within the pairwindow_scoring)

# %%
(
    haplotype_scores["relinteracting"].mean()
    / (reldeltacor_interacting).sel(phase="test").mean().item(),
    haplotype_scores["interacting"].mean()
    / (deltacor_interacting).sel(phase="test").mean().item(),
)

# %%
window_oi = 22500.0
windowpairs_oi = windowpair_scores.query(
    "(window1 == @window_oi) or (window2 == @window_oi)"
).index
sns.histplot(
    (reldeltacor_interacting)
    .sel(phase="test")
    .sel(windowpair=windowpairs_oi)
    .to_pandas(),
    stat="density",
    color="k",
    alpha=0.5,
    bins=10,
)
sns.histplot(
    (reldeltacor_interacting).sel(phase="test").to_pandas(),
    stat="density",
    color="k",
    alpha=0.5,
    bins=10,
)

# %%
window_scores.loc[windows_oi.index].sort_values("deltacor")

# %%
haplotype_scores

# %%

# %% [markdown]
# ## Directionality

# %% [markdown]
# ### Load

# %%
scores_folder = prediction.path / "scoring" / "windowdirection_gene" / gene
windowdirection_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %% [markdown]
# ### Interpret

# %%
plotdata = (
    windowdirection_scoring.genescores["retained"]
    .sel(gene=gene)
    .sel(phase="test")
    .mean("model")
    .to_pandas()
    .to_frame("retained")
)
plotdata = plotdata.join(windowdirection_scoring.design)

# %%
lost = windowdirection_scoring.genescores["lost"]
lost.coords["window_direction"] = pd.MultiIndex.from_frame(
    windowdirection_scoring.design[["window", "direction"]]
)
lost = lost.unstack("window_direction")

deltacor = windowdirection_scoring.genescores["deltacor"]
deltacor.coords["window_direction"] = pd.MultiIndex.from_frame(
    windowdirection_scoring.design[["window", "direction"]]
)
deltacor = deltacor.unstack("window_direction")

# %%
shrinkage_rate = 1 / 5
# shrinkage_rate = 10
shrinkage = np.exp(-lost.sum("direction") * shrinkage_rate)
plt.plot(
    np.linspace(0, 100, 100),
    np.exp(-np.linspace(0, 100, 100) * shrinkage_rate),
    color="k",
    linestyle="--",
)

# %%
numerator = deltacor.sel(direction="forward")
denominator = deltacor.sel(direction="reverse")
deltacor_bias = numerator - denominator


# %%
def shrink(numerator, denominator, shrinkage):
    out = ((1 - shrinkage) * (numerator / denominator)) + (shrinkage * 1)
    out.values[np.isinf(out.values)] = 1.0
    return out


# %%
deltacor_bias_ratio = shrink(
    deltacor_bias, np.clip(window_scoring.genescores["deltacor"], -np.inf, 0), shrinkage
)

# %%
fig, ax = plt.subplots()
deltacor_bias.sel(phase="test", gene=gene).mean("model").to_pandas().plot(ax=ax)
ax2 = ax.twiny()
# deltacor_bias_ratio.sel(phase = "test", gene = gene).mean("model").to_pandas().plot(ax = ax2)
window_scoring.genescores.sel(gene=gene, phase="test").mean("model").to_pandas()[
    "deltacor"
].plot()

# %%
numerator = lost.sel(direction="forward")
denominator = lost.sel(direction="reverse")
lost_ratio = shrink(numerator, denominator, shrinkage)
# ((1 - shrinkage) * (numerator / denominator)) + (shrinkage * 1)
# lost_ratio.values[np.isnan(lost_ratio) | np.isinf(lost_ratio)] = 1.

# %%
fig, ax = plt.subplots()
lost_ratio.sel(phase="test", gene=gene).mean("model").to_pandas().plot(ax=ax)
ax.set_yscale("log")

# %%
plotdata = pd.DataFrame(
    {
        "lost_ratio": lost_ratio.sel(phase="validation", gene=gene)
        .mean("model")
        .to_pandas(),
        "deltacor_bias": deltacor_bias.sel(phase="validation", gene=gene)
        .mean("model")
        .to_pandas(),
    }
).reset_index()
plotdata["deltacor"] = (
    window_scoring.genescores.sel(phase="validation", gene=gene)
    .mean("model")
    .to_pandas()["deltacor"]
    .values
)


# %%
focus = window
focus = [0, 6000]
# focus = [-40000, -10000]
# focus = [-50000, 10000]
# focus = [-100000, 100000]

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0))
fig.main[0, 0] = panel = chd.grid.Panel((10, 2))
ax = panel.ax
ax.plot(plotdata["window"], plotdata["lost_ratio"], marker=".")
ax.set_yscale("log")
ax.set_ylim(0.1, 10)
ax.axhline(1, dashes=(2, 2), color="#333")
ax.yaxis.set_inverted(True)
fig.plot()
ax2 = ax.twinx()
ax2.plot(plotdata["window"], plotdata["deltacor_bias"], color="red", marker=".")
ax2.set_ylim(-0.01, 0.01)
ax.set_xlim(*focus)
ax.axvline(4988)

fig.main[1, 0] = panel = chd.grid.Panel((10, 2))
ax = panel.ax
ax.plot(plotdata["window"], plotdata["deltacor"])
ax.set_xlim(*focus)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(100))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
fig.plot()

# %%
variant_prioritized_scores[["effect", "deltacor"]]

# %%
haplotypes.loc[""]

# %% [markdown]
# ## Window + size

# %%
scores_folder = prediction.path / "scoring" / "windowsize_gene" / gene
windowsize_scoring = chd.scoring.prediction.Scoring.load(scores_folder)

# %%
deltacor = windowsize_scoring.genescores.mean("model").sel(
    phase="validation", gene=gene
)["deltacor"]
lost = windowsize_scoring.genescores.mean("model").sel(phase="validation", gene=gene)[
    "lost"
]
reldeltacor = windowsize_scoring.genescores.mean("model").sel(
    phase="validation", gene=gene
)["deltacor"] / (
    1
    - windowsize_scoring.genescores.mean("model").sel(phase="validation", gene=gene)[
        "retained"
    ]
)

# %%
deltacor.coords["window_size"] = pd.MultiIndex.from_frame(
    windowsize_scoring.design[["window", "size"]]
)
lost.coords["window_size"] = pd.MultiIndex.from_frame(
    windowsize_scoring.design[["window", "size"]]
)

# %%
sns.heatmap(deltacor.to_pandas().unstack())

# %%
plotdata

# %%
fig, ax = plt.subplots(figsize=(30, 5))
ax.plot(
    lost.to_pandas().unstack().index, np.log1p(lost).to_pandas().unstack(), marker="."
)
# ax.set_xlim([-20000, -8000])
ax.set_xlim([3000, 6000])

fig, ax = plt.subplots(figsize=(30, 5))
plotdata = deltacor.to_pandas().unstack()
for size, plotdata_size in plotdata.items():
    ax.plot(plotdata_size.index, plotdata_size, marker=".", label=size)
ax.legend()
# ax.set_xlim([-20000, -8000])
ax.set_xlim([3000, 6000])

# %%
chd.utils.paircor(deltacor.unstack().values, np.log(0.1 + lost.unstack().values))

# %%
sns.heatmap(
    np.corrcoef(
        deltacor.to_pandas()
        .unstack()
        .T.loc[:, (lost.unstack().sum("size") > 10).values]
    )
)

# %%
deltacor

# %%
sns.heatmap(np.corrcoef(deltacor.to_pandas().unstack().T))

# %%

# %%
