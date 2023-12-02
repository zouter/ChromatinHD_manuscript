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

import torch_scatter
import torch

import tqdm.auto as tqdm

device = "cuda:0"

# %%
import peakfreeatac as pfa
import peakfreeatac.data

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "pbmc3k-pbmc10k"
# dataset_name = "lymphoma+pbmc10k"
dataset_name = "lymphoma-pbmc10k"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
prediction_design = pd.DataFrame({"model_name":[
    "v4_dummy",
    "v4",
    # "v4_1k-1k",
    # "v4_10-10",
    # "v4_150-0",
    # "v4_0-150",
    "v4_nn",
    # "v4_nn_dummy1",
    # "v4_nn_1k-1k",
    # "v4_split",
    # "v4_nn_split",
    # "v4_lw_split",
    # "v4_nn_lw",
    # "v4_nn_lw_split",
    # "v4_nn_lw_split_mean",
    # "v4_nn2_lw_split",
    # "v4_150-0_nn",
    # "v4_150-100-50-0_nn",
    # "v4_150-100-50-0",
    # "v4_1k-150-0_nn",
    # "v4_cutoff001",
    # "v4_nn_cutoff001",
    # "v4_prom",
    # "v4_prom_nn",
]})
prediction_design["dataset_name"] = dataset_name
prediction_design.index = prediction_design["model_name"]
prediction_design.index.name = "run"


# %%
class Prediction(pfa.flow.Flow):
    pass


# %%
scores = {}
genescores = {}
for run_name, design_row in prediction_design.iterrows():
    prediction = Prediction(pfa.get_output() / "prediction_sequence" / design_row["dataset_name"] / promoter_name / design_row["model_name"])
    
    scores_dir = (prediction.path / "scoring" / "overall")
    
    scores_prediction = pd.read_pickle(scores_dir / "scores.pkl")
    genescores_prediction = pd.read_pickle(scores_dir / "genescores.pkl")
    
    scores[run_name] = scores_prediction
    genescores[run_name] = genescores_prediction
scores = pd.concat(scores, names = ["run", "phase"])
genescores = pd.concat(genescores, names = ["run", "phase", "gene"])

# %%
scores = scores.join((scores - scores.loc["v4_dummy"]).rename(columns = lambda x: x+ "_relative"))

# %%
genescores = genescores.join((genescores - genescores.loc["v4_dummy"]).rename(columns = lambda x: x+ "_relative"))
genescores["label"] = pd.Categorical(transcriptome.symbol(genescores.index.get_level_values("gene")).values)

# %%
scores

# %%
scores_oi = scores.xs(
    # "validation",
    "test", 
level = "phase")[["cor_relative"]]
scores_oi.style.bar()
scores_oi["cor_relative"].plot(kind = "barh")

# %%
scores_oi = scores.xs("test", level = "phase")[["cor_relative"]]
scores_oi.style.bar()

# %%
scores_oi = scores.xs("validation_cell", level = "phase")[["mse_relative"]]
scores_oi.style.bar()

# %%
genescores

# %%
genescores["cor_ratio"] = genescores["cor_relative"]/genescores["cor"]

# %%
genescores.loc["v4_nn"].loc["validation"].query("cor_relative > 0").sort_values("cor_relative", ascending = False).head(30)

# %%
fig, ax = plt.subplots()
plt.scatter(genescores.loc["v4"].loc["validation"]["cor_relative"], genescores.loc["v4_nn"].loc["validation"]["cor_relative"])
ax.axline((0, 0), slope = 1)

# %%
fig, ax = plt.subplots()
plt.scatter(genescores.loc["v4"]["cor_relative"], genescores.loc["v4_1k-1k"]["cor_relative"])
ax.axline((0, 0), slope = 1)

# %%
genescores_oi = genescores.loc["v4"].loc["validation"].sort_values("mse_relative", ascending = False)
genescores_oi

# %%
# genes_oi = genescores_oi.index[:10]
genes_oi = transcriptome.gene_id(["MYC"])
sc.pl.umap(transcriptome.adata, color = genes_oi, title = transcriptome.symbol(genes_oi))

# %%
genescores_oi.index[:10]

# %%
# plt.scatter(genescores.loc["v4"].loc["validation_cell"]["cor_relative"], genescores.loc["v4"].loc["validation"]["cor_relative"])

# %%
genes_oi = genescores.loc["v4"].loc["validation"].query("cor_relative > 0.01").index

# %%

# %% [markdown]
# ### Investigate linear model

# %%
import pickle
prediction = Prediction(pfa.get_output() / "prediction_sequence_v5" / dataset_name / promoter_name / "v4")
model = pickle.load(open(prediction.path / ("model_" + str(0) + ".pkl"), "rb"))

# %%
model.embedding_gene_pooler

# %%
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name
motifscan_folder.mkdir(parents=True, exist_ok=True)
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"

# %%
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))

# %%
motif_linear_scores = pd.Series(
    model.embedding_gene_pooler.nn1[0].weight.detach().cpu().numpy()[0],
    # model.embedding_gene_pooler.motif_bias.detach().cpu().numpy(),
    index = motifs.index
).sort_values(ascending = False)
# motif_linear_scores = pd.Series(model.embedding_gene_pooler.linear_weight[0].detach().cpu().numpy(), index = motifs_oi.index).sort_values(ascending = False)

# %%
motif_linear_scores.plot(kind = "hist")

# %%
# motif_linear_scores["E2F3_HUMAN.H11MO.0.A"]

# %%
motif_linear_scores.head(10)

# %%
motif_linear_scores.tail(10)

# %%
motif_linear_scores.loc[motif_linear_scores.index.str.startswith("CTCF")]

# %%
motif_linear_scores.loc[motif_linear_scores.index.str.startswith("MBD")]

# %%
motif_linear_scores.loc[motif_linear_scores.index.str.startswith("MECP2")]

# %%
motifs_oi = motifs.loc[motifs["gene_label"].isin(transcriptome.var["symbol"])]

# %%
plotdata = pd.DataFrame({
    "is_variable":motifs["gene_label"].isin(transcriptome.var["symbol"]),
    "linear_score":motif_linear_scores
})
plotdata["dispersions_norm"] = pd.Series(
    transcriptome.var["dispersions_norm"][transcriptome.gene_id(motifs_oi["gene_label"]).values].values,
    index = motifs_oi.index
)

# %%
plotdata.groupby("is_variable").std()

# %%
sns.scatterplot(plotdata.dropna(), y = "linear_score", x = "dispersions_norm")

# %%
sns.stripplot(plotdata, y = "linear_score", x = "is_variable")

# %% [markdown]
# ### Investigate linear models within one dataset

# %%
prediction_design = pd.DataFrame({"name":[
    "v4_dummy",
    "v4",
    "v4_1k-1k",
]}).set_index("name")

# %%
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name
motifscan_folder.mkdir(parents=True, exist_ok=True)
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"

# %%
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))

# %%
motifweights = {}
for model_name in prediction_design.index:
    prediction = Prediction(pfa.get_output() / "prediction_sequence" / dataset_name / promoter_name / model_name)
    model = pickle.load(open(prediction.path / ("model_" + str(0) + ".pkl"), "rb"))
    
    motifweights[model_name] = pd.Series(model.embedding_gene_pooler.nn1[0].weight.detach().cpu().numpy()[0], motifs.index)
motifweights = pd.concat(motifweights, names = ["model", "motif"])
# genescores = pd.concat(genescores, names = ["model", "phase", "gene"])

# %%
np.corrcoef(motifweights.unstack())

# %%
plt.scatter(motifweights.loc["v4"], motifweights.loc["v4_1k-1k"])

# %% [markdown]
# ### Investigate linear models across datasets

# %%
prediction_design = pd.DataFrame({"model":"v4", "dataset":["pbmc10k", "lymphoma"]}).set_index("dataset", drop = False)

# %%
motifscan_folder = pfa.get_output() / "motifscans" / prediction_design["dataset"][0] / promoter_name
motifscan_folder.mkdir(parents=True, exist_ok=True)
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"

# %%
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))

# %%
motifweights = {}
for run_name, design_row in prediction_design.iterrows():
    prediction = Prediction(pfa.get_output() / "prediction_sequence" / design_row["dataset"] / promoter_name / design_row["model"])
    model = pickle.load(open(prediction.path / ("model_" + str(0) + ".pkl"), "rb"))
    
    motifweights[design_row["dataset"]] = pd.Series(model.embedding_gene_pooler.nn1[0].weight.detach().cpu().numpy()[0], motifs.index)
motifweights = pd.concat(motifweights, names = ["dataset", "motif"])
# genescores = pd.concat(genescores, names = ["model", "phase", "gene"])

# %%
motifweights

# %%
np.corrcoef(motifweights.unstack())

# %%
sns.pairplot(data = motifweights.unstack().T)

# %%
(motifweights.loc["lymphoma"] - motifweights.loc["pbmc10k"]).sort_values()

# %%
# !pip install adjustText

# %%
import adjustText

# %%
fig, ax = plt.subplots()
plotdata = motifweights.unstack()
plt.scatter(plotdata.loc["pbmc10k"], plotdata.loc["lymphoma"])
ax.set_xlabel("pbmc10k")
ax.set_ylabel("lymphoma")

texts = []
symbols_oi = pd.DataFrame([
    ["MECP2"],
    ["MBD2"],
    ["SPI1"],
    ["SPIB"],
    ["ALL4"],
    ["FOXO4"],
    ["IRF9"],
    ["SALL4"],
    ["BATF3"],
    ["ETV2"],
    ["CTCF"],
    ["E2F3"]
], columns = ["symbol"])
    
for symbol in symbols_oi["symbol"]:
    motifs_oi = motifs.loc[motifs.index.str.startswith(symbol)]
    for motif_oi in motifs_oi.index:
        ax.scatter(plotdata.loc["pbmc10k", motif_oi], plotdata.loc["lymphoma", motif_oi], color = "red")
        texts.append(ax.text(plotdata.loc["pbmc10k", motif_oi], plotdata.loc["lymphoma", motif_oi], s = motifs.loc[motif_oi, "gene_label"]))
adjustText.adjust_text(texts)

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data / "pbmc10k" / "transcriptome")
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["FOXO4", "ETV1"]))

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data / "lymphoma" / "transcriptome")
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["FOXO1", "FOXO3", "PAX5", "E2F2"]))

# %%
symbols_oi = ["E2F8", "E2F7"]

transcriptome = peakfreeatac.data.Transcriptome(folder_data / "lymphoma" / "transcriptome")
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(symbols_oi))
transcriptome = peakfreeatac.data.Transcriptome(folder_data / "pbmc10k" / "transcriptome")
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(symbols_oi))

# %%
transcriptome.var.loc[transcriptome.adata.var["symbol"].str.startswith("FOX")].sort_values("means")

# %%
texts
